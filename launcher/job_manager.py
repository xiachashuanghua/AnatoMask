from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_ROOT = PROJECT_ROOT / "webui_runs"
JOBS_ROOT = WEBUI_ROOT / "jobs"
DEFAULT_CACHE_ROOT = WEBUI_ROOT / "cache"
_RUNNING_JOBS: dict[str, subprocess.Popen[Any]] = {}


def _ensure_dirs() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _job_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_path(value: str | None, default: Path | None = None) -> str:
    if value:
        return str(Path(value).expanduser().resolve())
    if default is None:
        return ""
    return str(default.resolve())


def _require_text(value: str | None, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    return text


def _require_existing_dir(path_str: str, field_name: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{field_name} does not exist or is not a directory: {path}")
    return str(path)


def _require_existing_file(path_str: str, field_name: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist or is not a file: {path}")
    return str(path)


def _stringify_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _launch_process(command: list[str], job_id: str, job_dir: Path, config: dict) -> dict:
    _ensure_dirs()
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "stdout.log"
    state_path = job_dir / "state.json"
    command_path = job_dir / "command.txt"
    config_path = job_dir / "config.json"

    _write_json(config_path, config)
    command_path.write_text(_stringify_command(command), encoding="utf-8")

    with log_path.open("ab") as log_handle:
        kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)

    _RUNNING_JOBS[job_id] = process
    state = {
        "job_id": job_id,
        "status": "running",
        "pid": process.pid,
        "returncode": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_dir": str(job_dir),
        "log_path": str(log_path),
    }
    _write_json(state_path, state)
    return state


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def build_train_job(config: dict) -> tuple[list[str], dict]:
    job_id = _job_id("train")
    job_dir = _job_dir(job_id)
    artifacts_dir = job_dir / "artifacts"
    cache_dir = _normalize_path(config.get("cache_dir"), DEFAULT_CACHE_ROOT / job_id)
    pretrained_root = _require_existing_file(config.get("pretrained_root"), "pretrained_root") if config.get("pretrained_root") else ""
    checkpoint = _require_existing_file(config.get("checkpoint"), "checkpoint") if config.get("checkpoint") else ""
    data_dir = _require_existing_dir(_require_text(config.get("data_dir"), "data_dir"), "data_dir")
    json_list = _require_text(config.get("json_list"), "json_list")
    dataset_json_path = Path(data_dir) / json_list
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"dataset json does not exist: {dataset_json_path}")

    resolved = {
        "job_type": "train",
        "gpu_id": int(config["gpu_id"]),
        "data_dir": data_dir,
        "json_list": json_list,
        "cache_dir": cache_dir,
        "pretrained_root": pretrained_root,
        "logdir": str(artifacts_dir.resolve()),
        "feature_size": int(config["feature_size"]),
        "in_channels": int(config["in_channels"]),
        "out_channels": int(config["out_channels"]),
        "max_epochs": int(config["max_epochs"]),
        "batch_size": int(config["batch_size"]),
        "sw_batch_size": int(config["sw_batch_size"]),
        "optim_lr": float(config["optim_lr"]),
        "val_every": int(config["val_every"]),
        "workers": int(config["workers"]),
        "use_persistent_dataset": _as_bool(config.get("use_persistent_dataset"), default=True),
        "rand_flipd_prob": float(config["rand_flipd_prob"]),
        "rand_rotate90d_prob": float(config["rand_rotate90d_prob"]),
        "rand_scale_intensityd_prob": float(config["rand_scale_intensityd_prob"]),
        "rand_shift_intensityd_prob": float(config["rand_shift_intensityd_prob"]),
        "infer_overlap": float(config["infer_overlap"]),
        "roi_x": int(config["roi_x"]),
        "roi_y": int(config["roi_y"]),
        "roi_z": int(config["roi_z"]),
        "space_x": float(config["space_x"]),
        "space_y": float(config["space_y"]),
        "space_z": float(config["space_z"]),
        "a_min": float(config["a_min"]),
        "a_max": float(config["a_max"]),
        "b_min": float(config["b_min"]),
        "b_max": float(config["b_max"]),
        "checkpoint": checkpoint,
        "noamp": bool(config.get("noamp", False)),
        "job_id": job_id,
        "job_dir": str(job_dir.resolve()),
    }

    command = [
        sys.executable,
        "main.py",
        "--gpu_id",
        str(resolved["gpu_id"]),
        "--data_dir",
        resolved["data_dir"],
        "--json_list",
        resolved["json_list"],
        "--cache_dir",
        resolved["cache_dir"],
        "--logdir",
        resolved["logdir"],
        "--feature_size",
        str(resolved["feature_size"]),
        "--in_channels",
        str(resolved["in_channels"]),
        "--out_channels",
        str(resolved["out_channels"]),
        "--max_epochs",
        str(resolved["max_epochs"]),
        "--batch_size",
        str(resolved["batch_size"]),
        "--sw_batch_size",
        str(resolved["sw_batch_size"]),
        "--optim_lr",
        str(resolved["optim_lr"]),
        "--val_every",
        str(resolved["val_every"]),
        "--workers",
        str(resolved["workers"]),
        "--use_persistent_dataset",
        "true" if resolved["use_persistent_dataset"] else "false",
        "--RandFlipd_prob",
        str(resolved["rand_flipd_prob"]),
        "--RandRotate90d_prob",
        str(resolved["rand_rotate90d_prob"]),
        "--RandScaleIntensityd_prob",
        str(resolved["rand_scale_intensityd_prob"]),
        "--RandShiftIntensityd_prob",
        str(resolved["rand_shift_intensityd_prob"]),
        "--infer_overlap",
        str(resolved["infer_overlap"]),
        "--roi_x",
        str(resolved["roi_x"]),
        "--roi_y",
        str(resolved["roi_y"]),
        "--roi_z",
        str(resolved["roi_z"]),
        "--space_x",
        str(resolved["space_x"]),
        "--space_y",
        str(resolved["space_y"]),
        "--space_z",
        str(resolved["space_z"]),
        "--a_min",
        str(resolved["a_min"]),
        "--a_max",
        str(resolved["a_max"]),
        "--b_min",
        str(resolved["b_min"]),
        "--b_max",
        str(resolved["b_max"]),
    ]
    if resolved["pretrained_root"]:
        command.extend(["--pretrained_root", resolved["pretrained_root"]])
    if resolved["checkpoint"]:
        command.extend(["--checkpoint", resolved["checkpoint"]])
    if resolved["noamp"]:
        command.append("--noamp")

    return command, resolved


def build_infer_job(config: dict) -> tuple[list[str], dict]:
    job_id = _job_id("infer")
    job_dir = _job_dir(job_id)
    predictions_dir = job_dir / "predictions"
    data_dir = _require_existing_dir(_require_text(config.get("data_dir"), "data_dir"), "data_dir")
    json_list = _require_text(config.get("json_list"), "json_list")
    dataset_json_path = Path(data_dir) / json_list
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"dataset json does not exist: {dataset_json_path}")

    resolved = {
        "job_type": "infer",
        "gpu_id": int(config["gpu_id"]),
        "data_dir": data_dir,
        "json_list": json_list,
        "trained_pth": _require_existing_file(_require_text(config.get("trained_pth"), "trained_pth"), "trained_pth"),
        "save_prediction_path": _normalize_path(config.get("save_prediction_path"), predictions_dir),
        "feature_size": int(config["feature_size"]),
        "in_channels": int(config["in_channels"]),
        "out_channels": int(config["out_channels"]),
        "sw_batch_size": int(config["sw_batch_size"]),
        "workers": int(config["workers"]),
        "infer_overlap": float(config["infer_overlap"]),
        "roi_x": int(config["roi_x"]),
        "roi_y": int(config["roi_y"]),
        "roi_z": int(config["roi_z"]),
        "space_x": float(config["space_x"]),
        "space_y": float(config["space_y"]),
        "space_z": float(config["space_z"]),
        "a_min": float(config["a_min"]),
        "a_max": float(config["a_max"]),
        "b_min": float(config["b_min"]),
        "b_max": float(config["b_max"]),
        "noamp": bool(config.get("noamp", False)),
        "job_id": job_id,
        "job_dir": str(job_dir.resolve()),
    }

    command = [
        sys.executable,
        "eval.py",
        "--gpu_id",
        str(resolved["gpu_id"]),
        "--data_dir",
        resolved["data_dir"],
        "--json_list",
        resolved["json_list"],
        "--trained_pth",
        resolved["trained_pth"],
        "--save_prediction_path",
        resolved["save_prediction_path"],
        "--feature_size",
        str(resolved["feature_size"]),
        "--in_channels",
        str(resolved["in_channels"]),
        "--out_channels",
        str(resolved["out_channels"]),
        "--sw_batch_size",
        str(resolved["sw_batch_size"]),
        "--workers",
        str(resolved["workers"]),
        "--infer_overlap",
        str(resolved["infer_overlap"]),
        "--roi_x",
        str(resolved["roi_x"]),
        "--roi_y",
        str(resolved["roi_y"]),
        "--roi_z",
        str(resolved["roi_z"]),
        "--space_x",
        str(resolved["space_x"]),
        "--space_y",
        str(resolved["space_y"]),
        "--space_z",
        str(resolved["space_z"]),
        "--a_min",
        str(resolved["a_min"]),
        "--a_max",
        str(resolved["a_max"]),
        "--b_min",
        str(resolved["b_min"]),
        "--b_max",
        str(resolved["b_max"]),
    ]
    if resolved["noamp"]:
        command.append("--noamp")

    return command, resolved


def start_train_job(config: dict) -> tuple[str, str, str, str]:
    command, resolved = build_train_job(config)
    job_id = resolved["job_id"]
    job_dir = Path(resolved["job_dir"])
    state = _launch_process(command, job_id, job_dir, resolved)
    return job_id, str(job_dir), state["status"], _stringify_command(command)


def start_infer_job(config: dict) -> tuple[str, str, str, str]:
    command, resolved = build_infer_job(config)
    job_id = resolved["job_id"]
    job_dir = Path(resolved["job_dir"])
    state = _launch_process(command, job_id, job_dir, resolved)
    return job_id, str(job_dir), state["status"], _stringify_command(command)


def _finalize_state(job_id: str, state: dict) -> dict:
    process = _RUNNING_JOBS.get(job_id)
    if process is None:
        return state

    returncode = process.poll()
    if returncode is None:
        return state

    state["returncode"] = returncode
    state["status"] = "completed" if returncode == 0 else "failed"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_job_dir(job_id) / "state.json", state)
    _RUNNING_JOBS.pop(job_id, None)
    return state


def tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def list_jobs() -> list[str]:
    _ensure_dirs()
    return sorted([item.name for item in JOBS_ROOT.iterdir() if item.is_dir()], reverse=True)


def list_train_checkpoints() -> list[tuple[str, str]]:
    _ensure_dirs()
    choices: list[tuple[str, str]] = []
    for job_id in list_jobs():
        job_dir = _job_dir(job_id)
        state = _read_json(job_dir / "state.json")
        config = _read_json(job_dir / "config.json")
        if config.get("job_type") != "train":
            continue
        status = state.get("status", "unknown")
        artifacts_dir = job_dir / "artifacts"
        candidate_files = [
            ("best", artifacts_dir / "model.pt"),
            ("final", artifacts_dir / "model_final.pt"),
        ]
        for tag, checkpoint_path in candidate_files:
            if checkpoint_path.is_file():
                label = f"{job_id} [{status}] - {tag}"
                choices.append((label, str(checkpoint_path.resolve())))
    return choices


def get_job_details(job_id: str) -> tuple[dict, dict, str, str]:
    if not job_id:
        return {}, {}, "", ""
    job_dir = _job_dir(job_id)
    state = _read_json(job_dir / "state.json")
    config = _read_json(job_dir / "config.json")
    state = _finalize_state(job_id, state)
    log_tail = tail_text(job_dir / "stdout.log")

    artifacts = []
    if job_dir.exists():
        for path in sorted(job_dir.rglob("*")):
            if path.is_file():
                artifacts.append(str(path.relative_to(PROJECT_ROOT)))
    return state, config, log_tail, "\n".join(artifacts)


def cancel_job(job_id: str) -> dict:
    if not job_id:
        return {}
    job_dir = _job_dir(job_id)
    state_path = job_dir / "state.json"
    state = _read_json(state_path)
    process = _RUNNING_JOBS.get(job_id)
    if process is None:
        state["status"] = state.get("status", "unknown")
        _write_json(state_path, state)
        return state

    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    state["status"] = "cancelled"
    state["returncode"] = process.returncode
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(state_path, state)
    _RUNNING_JOBS.pop(job_id, None)
    return state
