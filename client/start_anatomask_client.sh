#!/usr/bin/env sh
set -eu

CLIENT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PROJECT_ROOT="$(CDPATH= cd -- "$CLIENT_ROOT/.." && pwd -P)"
ENV_NAME="AnatoMask"
ENV_DIR="$PROJECT_ROOT/.uv-envs/$ENV_NAME"
PYTHON_VERSION="${ANATOMASK_PYTHON_VERSION:-3.10}"
cd "$PROJECT_ROOT"

UNAME_S="$(uname -s 2>/dev/null || printf unknown)"
case "$UNAME_S" in
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM_FAMILY="windows"
    ;;
  Darwin*)
    PLATFORM_FAMILY="macos"
    ;;
  Linux*)
    PLATFORM_FAMILY="linux"
    ;;
  *)
    PLATFORM_FAMILY="unknown"
    ;;
esac

USER_HOME="${HOME:-}"
if [ -z "$USER_HOME" ] && [ -n "${USERPROFILE:-}" ] && command -v cygpath >/dev/null 2>&1; then
  USER_HOME="$(cygpath -u "$USERPROFILE")"
fi
if [ -z "$USER_HOME" ]; then
  USER_HOME="${USERPROFILE:-}"
fi

if [ "$PLATFORM_FAMILY" = "windows" ]; then
  UV_HOME="$USER_HOME/.local/bin"
  ENV_PYTHON="$ENV_DIR/Scripts/python.exe"
else
  UV_HOME="$USER_HOME/.local/bin"
  ENV_PYTHON="$ENV_DIR/bin/python"
fi

PATH="$UV_HOME:$PATH"
export PATH

REQUIREMENTS_FILE="$CLIENT_ROOT/requirements.txt"

log() {
  printf '%s\n' "[AnatoMask] $*"
}

die() {
  printf '%s\n' "[AnatoMask] ERROR: $*" >&2
  exit 1
}

[ -n "$USER_HOME" ] || die "Unable to determine the current user home directory."

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi

  if [ "$PLATFORM_FAMILY" = "windows" ] && [ -x "$UV_HOME/uv.exe" ]; then
    UV_BIN="$UV_HOME/uv.exe"
    return 0
  fi

  if [ -x "$UV_HOME/uv" ]; then
    UV_BIN="$UV_HOME/uv"
    return 0
  fi

  return 1
}

install_uv() {
  log "uv was not found. Installing it now."
  if [ "$PLATFORM_FAMILY" = "windows" ]; then
    command -v powershell.exe >/dev/null 2>&1 || die "Installing uv on Windows requires powershell.exe."
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || die "uv installation failed."
  else
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv installation failed."
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | sh || die "uv installation failed."
    else
      die "Installing uv requires curl or wget."
    fi
  fi
  find_uv || die "uv was installed, but the current shell could not find the uv executable."
}

ensure_uv() {
  if ! find_uv; then
    install_uv
  fi
  log "Using uv: $UV_BIN"
}

detect_cuda() {
  CUDA_SCRIPT="$PROJECT_ROOT/scripts/detect_cuda.sh"
  [ -f "$CUDA_SCRIPT" ] || die "Missing CUDA detection script: $CUDA_SCRIPT"

  if ! TORCH_CHANNEL="$(sh "$CUDA_SCRIPT" --channel 2>/dev/null)"; then
    sh "$CUDA_SCRIPT" --summary || true
    die "This machine does not meet the GPU runtime requirements. Install a supported NVIDIA driver first. CPU is not supported."
  fi

  DRIVER_MAJOR="$(sh "$CUDA_SCRIPT" --driver-major)"
  log "Detected NVIDIA driver major version: $DRIVER_MAJOR"
  log "Selected PyTorch channel: $TORCH_CHANNEL"
}

ensure_environment() {
  if [ -x "$ENV_PYTHON" ]; then
    log "Found existing uv environment: $ENV_DIR"
    return 0
  fi

  log "Creating uv environment $ENV_NAME -> $ENV_DIR"
  "$UV_BIN" python install "$PYTHON_VERSION"
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$ENV_DIR"
}

runtime_check() {
  if [ ! -x "$ENV_PYTHON" ]; then
    printf '%s\n' "Environment python executable was not found."
    return 1
  fi

  ANATOMASK_PROJECT_ROOT="$PROJECT_ROOT" "$ENV_PYTHON" - <<'PY'
import importlib
import os
import sys

project_root = os.environ.get("ANATOMASK_PROJECT_ROOT", "")
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

errors = []
modules = [
    "gradio",
    "webview",
    "monai",
    "nibabel",
    "SimpleITK",
    "matplotlib",
    "tensorboardX",
    "einops",
    "PIL",
    "scipy",
    "timm",
    "torch",
    "launcher.webui",
    "client.desktop_app",
]
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")

if not errors:
    import torch
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available(): False")
    else:
        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"gpu_count={torch.cuda.device_count()}")

if errors:
    print("RUNTIME_CHECK_FAILED")
    for item in errors:
        print(item)
    raise SystemExit(1)
print("RUNTIME_CHECK_OK")
PY
}

install_base_packages() {
  log "Installing desktop client dependencies from client/requirements.txt"
  "$UV_BIN" pip install --python "$ENV_PYTHON" --link-mode copy -r "$REQUIREMENTS_FILE"
}

install_gpu_pytorch() {
  log "Installing GPU PyTorch build ($TORCH_CHANNEL)"
  "$UV_BIN" pip install --python "$ENV_PYTHON" --link-mode copy --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL" torch torchvision torchaudio
}

recreate_environment() {
  if [ -d "$ENV_DIR" ]; then
    log "Removing broken environment: $ENV_DIR"
    rm -rf "$ENV_DIR"
  fi
  ensure_environment
}

ensure_runtime() {
  LAST_RUNTIME_OUTPUT="$(runtime_check 2>&1 || true)"
  if printf '%s' "$LAST_RUNTIME_OUTPUT" | grep -q "RUNTIME_CHECK_OK"; then
    log "AnatoMask desktop environment is ready. Launching the client."
    return 0
  fi

  if [ -n "$LAST_RUNTIME_OUTPUT" ]; then
    log "Runtime check reported:"
    printf '%s\n' "$LAST_RUNTIME_OUTPUT"
  fi

  install_gpu_pytorch
  install_base_packages

  LAST_RUNTIME_OUTPUT="$(runtime_check 2>&1 || true)"
  if printf '%s' "$LAST_RUNTIME_OUTPUT" | grep -q "RUNTIME_CHECK_OK"; then
    return 0
  fi

  if [ -n "$LAST_RUNTIME_OUTPUT" ]; then
    log "Runtime check still failed after install:"
    printf '%s\n' "$LAST_RUNTIME_OUTPUT"
  fi

  log "Recreating the uv environment and reinstalling once."
  recreate_environment
  install_gpu_pytorch
  install_base_packages
  LAST_RUNTIME_OUTPUT="$(runtime_check 2>&1 || true)"
  if printf '%s' "$LAST_RUNTIME_OUTPUT" | grep -q "RUNTIME_CHECK_OK"; then
    log "Environment rebuild succeeded."
    return 0
  fi
  if [ -n "$LAST_RUNTIME_OUTPUT" ]; then
    log "Runtime check after rebuild still failed:"
    printf '%s\n' "$LAST_RUNTIME_OUTPUT"
  fi

  die "Environment setup finished, but runtime validation still failed. Check the diagnostics above."
}

launch_client() {
  HOST="${ANATOMASK_CLIENT_HOST:-${ANATOMASK_HOST:-127.0.0.1}}"
  PORT="${ANATOMASK_CLIENT_PORT:-${ANATOMASK_PORT:-7860}}"

  if [ "$#" -eq 0 ]; then
    set -- --host "$HOST" --port "$PORT"
  fi

  log "Launching desktop client: $ENV_PYTHON -m client $*"
  exec "$ENV_PYTHON" -m client "$@"
}

main() {
  ensure_uv
  detect_cuda
  ensure_environment
  ensure_runtime
  launch_client "$@"
}

main "$@"
