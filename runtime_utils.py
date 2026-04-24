from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - optional during setup checks
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent


def safe_set_resource_limit(limit: int = 8192) -> str:
    try:
        import resource
    except ImportError:
        return "resource module unavailable on this platform"

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    return str(resource.getrlimit(resource.RLIMIT_NOFILE))


def configure_runtime_warnings() -> None:
    noisy_messages = [
        r"Importing from timm\.models\.layers is deprecated",
        r".*Orientationd\.__init__:labels: Current default value.*",
        r"`torch\.cuda\.amp\..*` is deprecated.*",
        r"torch\.meshgrid: in an upcoming release.*",
        r"Detected pickle protocol 5 in the checkpoint.*",
        r"Corrupt cache file detected: .*",
        r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        r"Padding: moving img .* from cuda to cpu for dtype=.* mode=constant\.",
    ]
    for message in noisy_messages:
        warnings.filterwarnings("ignore", message=message, category=Warning)


def ensure_cuda_available() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this project. CPU is not supported in this build.")


def validate_gpu_id(gpu_id: int) -> None:
    ensure_cuda_available()
    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise ValueError(f"Invalid gpu_id={gpu_id}. Available GPU count: {device_count}.")


def get_device(gpu_id: int):
    validate_gpu_id(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def get_default_distributed_backend() -> str:
    return "gloo" if platform.system() == "Windows" else "nccl"


def resolve_datalist_path(data_dir: str | None, json_list: str | None, datalist_json: str | None = None) -> str:
    if datalist_json:
        return str(Path(datalist_json).expanduser().resolve())
    if not data_dir or not json_list:
        raise ValueError("Either datalist_json or both data_dir and json_list must be provided.")
    return str((Path(data_dir).expanduser() / json_list).resolve())


def dumps_pretty(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
