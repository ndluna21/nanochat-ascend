"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
# Utilities

def _scalar_to_device(
    t: Tensor,
    ref: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Move a 0-D tensor (often stored on CPU to avoid recompilation) onto the same
    device as `ref` for device-local fused ops (NPU/CUDA).
    """
    if dtype is None:
        dtype = ref.dtype
    # Keep it as a tensor (not Python float) to preserve original intent.
    return t.to(device=ref.device, dtype=dtype)

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

def adamw_step_fused(
    p: Tensor,              # parameter tensor
    grad: Tensor,           # gradient, same shape as p
    exp_avg: Tensor,        # first moment, same shape as p
    exp_avg_sq: Tensor,     # second moment, same shape as p
    step_t: Tensor,         # () - 0-D tensor (often CPU), step count
    lr_t: Tensor,           # () - 0-D tensor (often CPU), learning rate
    beta1_t: Tensor,        # () - 0-D tensor (often CPU), beta1
    beta2_t: Tensor,        # () - 0-D tensor (often CPU), beta2
    eps_t: Tensor,          # () - 0-D tensor (often CPU), epsilon
    wd_t: Tensor,           # () - 0-D tensor (often CPU), weight decay
) -> None:
    """
    Fused AdamW step.
    NOTE: On Ascend NPU, scalar tensors participating in device ops must be on the
    same device. We therefore move the 0-D tensors to p.device before use.
    """
    # Move scalars to device to avoid CPU/NPU mixing errors
    lr = _scalar_to_device(lr_t, p)
    wd = _scalar_to_device(wd_t, p)
    beta1 = _scalar_to_device(beta1_t, p)
    beta2 = _scalar_to_device(beta2_t, p)
    eps = _scalar_to_device(eps_t, p)
    # Step participates in pow; keep as float32 for stability but on-device
    step = _scalar_to_device(step_t, p, dtype=torch.float32)

    one = torch.ones((), device=p.device, dtype=p.dtype)

    # Weight decay (decoupled, applied before the update)
    p.mul_(one - lr * wd)
    # Update running averages
    exp_avg.lerp_(grad, one - beta1)
    exp_avg_sq.lerp_(grad.square(), one - beta2)
    # Bias corrections
    bias1 = one - beta1.to(dtype=torch.float32) ** step
    bias2 = one - beta2.to(dtype=torch.float32) ** step
    # Compute update and apply
    denom = (exp_avg_sq / bias2.to(dtype=exp_avg_sq.dtype)).sqrt() + eps
    step_size = lr / bias1.to(dtype=lr.dtype)
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def muon_step_fused(
    stacked_grads: Tensor,          # (K, m, n) stacked gradients
    stacked_params: Tensor,         # (K, m, n) stacked parameters
    momentum_buffer: Tensor,        # (K, m, n) momentum buffer
    second_momentum_buffer: Tensor, # (K, m, 1) or (K, 1, n) factored second moment
    momentum_t: Tensor,             # () - 0-D tensor (often CPU)
    lr_t: Tensor,                   # () - 0-D tensor (often CPU)
    wd_t: Tensor,                   # () - 0-D tensor (often CPU)
    beta2_t: Tensor,                # () - 0-D tensor (often CPU)
    ns_steps: int,                  # number of Polar Express iterations
    red_dim: int,                   # reduction dim for variance
) -> None:
    """
    Fused Muon step.
    NOTE: On Ascend NPU, scalar tensors participating in device ops must be on the
    same device. We therefore move the 0-D tensors to stacked_grads.device before use.
    """
    dev_ref = stacked_grads
    one = torch.ones((), device=dev_ref.device, dtype=dev_ref.dtype)

    momentum = _scalar_to_device(momentum_t, dev_ref)
    lr = _scalar_to_device(lr_t, dev_ref)
    wd = _scalar_to_device(wd_t, dev_ref)
    beta2 = _scalar_to_device(beta2_t, dev_ref)

    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, one - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express (orthogonalization)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), one - beta2.to(dtype=one.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single-device version

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU/NPU version.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors (kept for original intent). We move them to device inside fused ops.
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        params: list[Tensor] = group["params"]
        if not params:
            return

        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors (kept for original intent). We move them to device inside fused ops.
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0, (
                    f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                )
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        params = group["params"]
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p0 = params[0]
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        param_infos = info["param_infos"]
        for p in group["params"]:
            if p not in param_infos:
                continue
            pinfo = param_infos[p]
            pinfo["future"].wait()
            grad_slice = pinfo["grad_slice"]
            state = self.state[p]

            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            adamw_step_fused(
                p_slice, grad_slice, state["exp_avg"], state["exp_avg_sq"],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, stacked_params=None, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"]
        p0 = params[0]
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p0]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])

            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]  # reuse buffer
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(
                    info["params"],
                    list(info["stacked_params"][:len(info["params"])].unbind(0)),
                )

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon":
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon":
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        self._finish_gathers(gather_list)
