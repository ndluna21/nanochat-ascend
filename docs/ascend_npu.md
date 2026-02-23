# Ascend NPU (Experimental) Support

This fork adds experimental training support for Huawei Ascend NPU (for example Ascend 910B) in `nanochat`, covering:

- device autodetection (`npu`)
- distributed training initialization via `HCCL`
- BF16 autocast paths for train/inference
- optimizer compatibility fixes for `torch_npu` scalar/device ops
- SFT script support
- basic reporting for NPU hardware + CANN version

The code paths are designed to keep CUDA behavior unchanged while making Ascend training runnable.

## Scope

Supported paths in this fork:

- `scripts.base_train` (pretraining)
- `scripts.chat_sft` (SFT)
- `scripts.base_eval` / `scripts.chat_eval` (via shared init and report paths)
- `nanochat.engine` inline test path (`__main__`) on NPU

## Environment Requirements

You need a working Ascend software stack before running the repo:

- CANN runtime/toolkit installed and configured
- PyTorch build compatible with your `torch_npu` version
- `torch_npu` installed and importable
- `torch.distributed` with `HCCL` backend available

Quick check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("has torch.npu:", hasattr(torch, "npu"))
if hasattr(torch, "npu"):
    print("npu available:", torch.npu.is_available())
    if torch.npu.is_available():
        print("device count:", torch.npu.device_count())
        print("device 0:", torch.npu.get_device_name(0))
print("cann:", getattr(torch.version, "cann", None))
PY
```

## Quick Start

An example end-to-end script is provided:

- `runs/run_ascend.sh`

The script is now a public template and intentionally avoids hardcoded local paths, proxies, and credentials.

Example usage:

```bash
export SWANLAB_API_KEY=your_key_here
export SWANLAB_WORKSPACE=your_workspace   # optional
export NNPU=4
export DEPTH=26
export DEVICE_BATCH_SIZE=16
bash runs/run_ascend.sh
```

Optional environment variables used by `runs/run_ascend.sh`:

- `NANOCHAT_BASE_DIR` (default: `<repo>/out`)
- `HTTP_PROXY`, `HTTPS_PROXY`, `HF_ENDPOINT`, `PIP_INDEX_URL`
- `RUN_REPORT_RESET` (`1`/`0`)
- `DOWNLOAD_DATASETS` (`1`/`0`)
- `TRAIN_TOKENIZER` (`1`/`0`)
- `NNPU`, `DEPTH`, `TARGET_PARAM_DATA_RATIO`, `DEVICE_BATCH_SIZE`, `EVAL_DEVICE_BATCH_SIZE`, `WINDOW_PATTERN`
- `RUN_NAME`, `SFT_RUN_NAME`

## Key Implementation Changes

### 1. Device and distributed init

- `nanochat/common.py`
  - adds `npu` autodetect path
  - validates `torch.npu` availability
  - seeds NPU RNG
  - initializes `torch.distributed` with `backend="hccl"` for multi-NPU runs

### 2. BF16 paths for NPU

- `scripts/base_train.py`
- `scripts/chat_sft.py`
- `nanochat/engine.py`
- `nanochat/gpt.py`

NPU is treated like CUDA for BF16 autocast and relevant BF16 storage paths (embeddings / value embeddings).

### 3. Optimizer compatibility on NPU

- `nanochat/optim.py`

The fused AdamW/Muon code originally keeps hyperparameters as CPU 0-D tensors. On Ascend NPU, some fused/device-local ops require scalar tensors to be on the same device as the target tensors.

This fork moves those scalars to the target device inside the fused step functions before computation.

### 4. Reporting and observability

- `nanochat/report.py`

Adds NPU hardware reporting:

- device count
- device names
- total NPU memory
- CANN version (`torch.version.cann`)

## MFU and Throughput Metrics

SFT MFU is computed using the detected accelerator peak BF16 FLOPS (instead of a hardcoded H100 constant):

- CUDA: uses the existing `get_peak_flops(...)` lookup
- NPU: Ascend 910B is mapped to `313e12` BF16 FLOPS (approximate)

This makes MFU numbers more meaningful on Ascend hardware.

## Cost Estimation in Reports

The report header now supports NPU cost estimation and uses two pricing layers:

1. Local reference prices (preferred, CNY/hour/card)
   - Ascend 910B: `3 CNY/hour/card`
2. GPU fallback (USD/hour/card, rough Lambda pricing)
   - H100: `3.00`
   - A100: `1.79`
   - V100: `0.55`

If an NPU model is detected but no local reference price matches, the report skips cost estimation instead of guessing.

## Known Limitations

- `torch.compile` is disabled on the NPU training paths in this fork for compatibility/stability.
- MFU is currently lower than well-tuned CUDA/H100 baselines (correctness/reproducibility first).
- FlashAttention acceleration is not available on Ascend in this fork; attention uses fallback paths.
- Performance tuning for Ascend kernels/operators is not exhaustive yet.
- The `Ascend 910B -> 313e12 BF16 FLOPS` mapping is approximate and intended for MFU estimation, not exact benchmarking.
- Some scripts may still assume CUDA-centric defaults in comments/log wording even though runtime logic supports NPU.

## Publish Checklist (Recommended)

Before tagging a public release, fill in the exact validated environment matrix in your release notes:

- server model / CPU
- Ascend model and card count
- CANN version
- PyTorch version
- `torch_npu` version
- single-node / multi-node scope tested
- max stable `--device-batch-size` used for your target model depth

It is also helpful to include:

- one successful pretraining command
- one successful SFT command
- one short throughput sample (`tok/sec`, MFU)
- known failure modes (OOM, unsupported ops, compile limitations)
