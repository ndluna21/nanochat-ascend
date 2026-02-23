import importlib
import os

from nanochat.common import DummyWandb


class _ModuleLoggerAdapter:
    """Wrap module-level logging APIs (e.g. swanlab.log / swanlab.finish)."""

    def __init__(self, module):
        self._module = module

    def log(self, *args, **kwargs):
        return self._module.log(*args, **kwargs)

    def finish(self, *args, **kwargs):
        finish_fn = getattr(self._module, "finish", None)
        if callable(finish_fn):
            return finish_fn(*args, **kwargs)
        return None


def _resolve_backend(logger_choice: str) -> str:
    choice = (logger_choice or "auto").strip().lower()
    if choice != "auto":
        return choice

    env_choice = os.environ.get("NANOCHAT_LOGGER", "").strip().lower()
    if env_choice in {"wandb", "swanlab", "none"}:
        return env_choice

    # Prefer SwanLab when an explicit SwanLab credential/workspace is provided.
    if os.environ.get("SWANLAB_API_KEY") or os.environ.get("SWANLAB_WORKSPACE"):
        return "swanlab"
    return "wandb"


def _init_wandb(run_name, config, default_project, print_fn):
    try:
        wandb = importlib.import_module("wandb")
    except Exception as e:
        raise RuntimeError("Failed to import wandb. Install `wandb` or choose --logger=swanlab/none.") from e

    wandb_init_kwargs = dict(
        project=os.environ.get("WANDB_PROJECT", default_project),
        name=run_name,
        config=config,
    )
    wandb_entity = os.environ.get("WANDB_ENTITY", "").strip()
    if wandb_entity:
        wandb_init_kwargs["entity"] = wandb_entity
    if print_fn:
        print_fn(f"Experiment logger: wandb (project={wandb_init_kwargs['project']})")
    return wandb.init(**wandb_init_kwargs), "wandb"


def _init_swanlab(run_name, config, default_project, print_fn):
    try:
        swanlab = importlib.import_module("swanlab")
    except Exception as e:
        raise RuntimeError("Failed to import swanlab. Install `swanlab` or choose --logger=wandb/none.") from e

    swanlab_api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if swanlab_api_key:
        swanlab.login(api_key=swanlab_api_key, save=True)
    elif print_fn:
        print_fn("SWANLAB_API_KEY is not set, using existing SwanLab login session if available.")

    swanlab_init_kwargs = dict(
        project=os.environ.get("SWANLAB_PROJECT", default_project),
        experiment_name=run_name,
        config=config,
    )
    swanlab_workspace = os.environ.get("SWANLAB_WORKSPACE", "").strip()
    if swanlab_workspace:
        swanlab_init_kwargs["workspace"] = swanlab_workspace
    if print_fn:
        print_fn(f"Experiment logger: swanlab (project={swanlab_init_kwargs['project']})")
    swanlab.init(**swanlab_init_kwargs)
    return _ModuleLoggerAdapter(swanlab), "swanlab"


def init_experiment_logger(
    *,
    run_name: str,
    config: dict,
    master_process: bool,
    logger_choice: str = "auto",
    wandb_project: str,
    swanlab_project: str,
    print_fn=None,
):
    """
    Initialize an experiment logger compatible with `.log()` and `.finish()`.

    Backends:
    - auto (default): resolve from env / installed packages
    - wandb
    - swanlab
    - none
    """
    if run_name == "dummy" or not master_process:
        if print_fn and master_process:
            print_fn("Experiment logger: none (disabled by --run=dummy)")
        return DummyWandb(), "none"

    backend = _resolve_backend(logger_choice)
    if backend == "none":
        if print_fn:
            print_fn("Experiment logger: none")
        return DummyWandb(), "none"
    if backend == "wandb":
        return _init_wandb(run_name, config, wandb_project, print_fn)
    if backend == "swanlab":
        return _init_swanlab(run_name, config, swanlab_project, print_fn)

    raise ValueError(f"Unsupported logger backend: {logger_choice!r}")

