# AnatoMask

This repository now includes a first web-based wrapper for training and inference.

## Scope

- Single-machine GPU training
- Single-machine GPU inference
- Windows and Linux single-card usage
- CPU is not supported
- Distributed training is not supported in this first release

## Quick Start With uv

Use the bootstrap script below. It will:

- Detect whether `uv` exists
- Install `uv` automatically when missing
- Create a dedicated uv environment at `.uv-envs/AnatoMask`
- Install base dependencies
- Detect the NVIDIA driver / CUDA compatibility
- Install a supported GPU build of PyTorch
- Launch the web UI

```bash
sh start_anatomask.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_anatomask.ps1
```

Windows cmd / double-click:

```bat
start_anatomask.bat
```

If the environment is already complete, the same command will skip installation and directly launch the web UI.

Environment variables:

```bash
ANATOMASK_HOST=0.0.0.0 ANATOMASK_PORT=7860 sh start_anatomask.sh
ANATOMASK_PYTHON_VERSION=3.10 sh start_anatomask.sh
```

Windows note:

- `start_anatomask.sh` is a shell script. Run it in `Git Bash`, `MSYS2`, `Cygwin`, or `WSL`.
- It will call the official PowerShell installer when `uv` is missing on Windows.
- If you prefer native Windows startup, use `start_anatomask.ps1`.
- If you want a double-click entrypoint for regular Windows users, use `start_anatomask.bat`.

## Manual Install

1. Detect the recommended PyTorch CUDA channel.

```bash
sh scripts/detect_cuda.sh --summary
python scripts/recommend_torch_install.py
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\detect_cuda.ps1 -Summary
```

2. Install a supported GPU build of PyTorch manually if needed.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

3. Install the remaining dependencies.

```bash
python -m pip install -r requirements.web.txt
```

## Start The Web UI

`sh start_anatomask.sh` is the recommended path.

Native Windows PowerShell startup:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_anatomask.ps1
```

Windows cmd startup:

```bat
start_anatomask.bat
```

Manual start:

```bash
python launch_webui.py --host 0.0.0.0 --port 7860
```

Open `http://127.0.0.1:7860` or replace the host with your server address.

## What The Web UI Does

- Training tab:
  - Fill in dataset path, dataset JSON file name, GPU ID, epochs, batch size, workers, preprocessing parameters.
  - Click `Start training`.
  - A job directory is created under `webui_runs/jobs/<job-id>/`.

- Inference tab:
  - Fill in dataset path, dataset JSON file name, trained checkpoint path, GPU ID, and output path.
  - Click `Start inference`.
  - Prediction files are written to the job directory or your chosen output path.

- Jobs tab:
  - Inspect `state.json`, `config.json`, log tail, and generated artifacts.
  - Cancel running jobs.

## Important Notes

- Training uses `main.py`.
- Inference uses `eval.py`.
- The web layer launches background subprocess jobs instead of embedding the training loop into the UI process.
- Each job persists its command, config, and logs so the flow is traceable.
- CPU is intentionally unsupported in the bootstrap flow.
