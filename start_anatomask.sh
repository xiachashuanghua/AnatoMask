#!/usr/bin/env sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
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

log() {
  printf '%s\n' "[AnatoMask] $*"
}

die() {
  printf '%s\n' "[AnatoMask] ERROR: $*" >&2
  exit 1
}

[ -n "$USER_HOME" ] || die "无法确定当前用户主目录。"

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
  log "未检测到 uv，开始自动安装。"
  if [ "$PLATFORM_FAMILY" = "windows" ]; then
    command -v powershell.exe >/dev/null 2>&1 || die "Windows 下安装 uv 需要 powershell.exe。"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || die "uv 安装失败。"
  else
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv 安装失败。"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | sh || die "uv 安装失败。"
    else
      die "安装 uv 需要 curl 或 wget。"
    fi
  fi
  find_uv || die "uv 已安装，但当前 shell 无法定位 uv 可执行文件。"
}

ensure_uv() {
  if ! find_uv; then
    install_uv
  fi
  log "使用 uv: $UV_BIN"
}

detect_cuda() {
  CUDA_SCRIPT="$PROJECT_ROOT/scripts/detect_cuda.sh"
  [ -f "$CUDA_SCRIPT" ] || die "缺少 CUDA 检测脚本: $CUDA_SCRIPT"

  if ! TORCH_CHANNEL="$(sh "$CUDA_SCRIPT" --channel 2>/dev/null)"; then
    sh "$CUDA_SCRIPT" --summary || true
    die "当前机器未满足 GPU 版运行条件。请先安装 NVIDIA 驱动 / CUDA 环境。CPU 不支持。"
  fi

  DRIVER_MAJOR="$(sh "$CUDA_SCRIPT" --driver-major)"
  log "检测到 NVIDIA 驱动主版本: $DRIVER_MAJOR"
  log "将安装 PyTorch 通道: $TORCH_CHANNEL"
}

ensure_environment() {
  if [ -x "$ENV_PYTHON" ]; then
    log "已找到 uv 环境: $ENV_DIR"
    return 0
  fi

  log "创建 uv 环境 $ENV_NAME -> $ENV_DIR"
  "$UV_BIN" python install "$PYTHON_VERSION"
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$ENV_DIR"
}

runtime_check() {
  if [ ! -x "$ENV_PYTHON" ]; then
    printf '%s\n' "Environment python executable was not found."
    return 1
  fi

  "$ENV_PYTHON" - <<'PY'
import importlib
errors = []
modules = [
    "gradio",
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
  log "安装基础依赖 requirements.web.txt"
  "$UV_BIN" pip install --python "$ENV_PYTHON" --link-mode copy -r "$PROJECT_ROOT/requirements.web.txt"
}

install_gpu_pytorch() {
  log "安装 GPU 版 PyTorch ($TORCH_CHANNEL)"
  "$UV_BIN" pip install --python "$ENV_PYTHON" --link-mode copy --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL" torch torchvision torchaudio
}

recreate_environment() {
  if [ -d "$ENV_DIR" ]; then
    log "移除损坏环境: $ENV_DIR"
    rm -rf "$ENV_DIR"
  fi
  ensure_environment
}

ensure_runtime() {
  LAST_RUNTIME_OUTPUT="$(runtime_check 2>&1 || true)"
  if printf '%s' "$LAST_RUNTIME_OUTPUT" | grep -q "RUNTIME_CHECK_OK"; then
    log "AnatoMask 环境已就绪，直接启动 Web UI。"
    return 0
  fi

  if [ -n "$LAST_RUNTIME_OUTPUT" ]; then
    log "运行时检查输出:"
    printf '%s\n' "$LAST_RUNTIME_OUTPUT"
  fi

  install_gpu_pytorch
  install_base_packages

  LAST_RUNTIME_OUTPUT="$(runtime_check 2>&1 || true)"
  if printf '%s' "$LAST_RUNTIME_OUTPUT" | grep -q "RUNTIME_CHECK_OK"; then
    return 0
  fi

  if [ -n "$LAST_RUNTIME_OUTPUT" ]; then
    log "安装后运行时检查仍失败:"
    printf '%s\n' "$LAST_RUNTIME_OUTPUT"
  fi

  log "开始重建 uv 环境并重装一次。"
  recreate_environment
  install_gpu_pytorch
  install_base_packages
  LAST_RUNTIME_OUTPUT="$(runtime_check 2>&1 || true)"
  if printf '%s' "$LAST_RUNTIME_OUTPUT" | grep -q "RUNTIME_CHECK_OK"; then
    log "环境重建完成。"
    return 0
  fi
  if [ -n "$LAST_RUNTIME_OUTPUT" ]; then
    log "重建后运行时检查仍失败:"
    printf '%s\n' "$LAST_RUNTIME_OUTPUT"
  fi

  die "环境初始化完成，但运行时校验失败。请检查上面的诊断输出。"
}

launch_webui() {
  cd "$PROJECT_ROOT"
  HOST="${ANATOMASK_HOST:-0.0.0.0}"
  PORT="${ANATOMASK_PORT:-7860}"

  if [ "$#" -eq 0 ]; then
    set -- --host "$HOST" --port "$PORT"
  fi

  log "启动 Web UI: $ENV_PYTHON $PROJECT_ROOT/launch_webui.py $*"
  exec "$ENV_PYTHON" "$PROJECT_ROOT/launch_webui.py" "$@"
}

main() {
  ensure_uv
  detect_cuda
  ensure_environment
  ensure_runtime
  launch_webui "$@"
}

main "$@"
