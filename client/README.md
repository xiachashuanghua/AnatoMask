# AnatoMask Desktop Client

This folder contains the desktop client wrapper around the existing local Web UI.

## Entry files

- `desktop_app.py`: launches the embedded Gradio server and opens it inside a native window through `pywebview`
- `requirements.txt`: desktop client dependencies
- `start_anatomask_client.bat`
- `start_anatomask_client.ps1`
- `start_anatomask_client.sh`

## Notes

- The desktop client reuses the same `AnatoMask` uv environment as the Web version.
- Training, inference, jobs, checkpoints, and saved UI parameters still use the existing project data under `webui_runs`.
- Windows usually works directly after installing `pywebview`.
- Linux may still require a local GUI backend such as GTK or Qt depending on the machine.
