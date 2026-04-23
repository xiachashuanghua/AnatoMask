#!/usr/bin/env sh
set -eu

MODE="summary"
if [ "${1-}" = "--channel" ]; then
  MODE="channel"
elif [ "${1-}" = "--driver-major" ]; then
  MODE="driver-major"
elif [ "${1-}" = "--quiet" ]; then
  MODE="quiet"
elif [ "${1-}" = "--summary" ] || [ -z "${1-}" ]; then
  MODE="summary"
else
  printf '%s\n' "Usage: sh scripts/detect_cuda.sh [--summary|--channel|--driver-major|--quiet]" >&2
  exit 2
fi

find_nvidia_smi() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    command -v nvidia-smi
    return 0
  fi
  if command -v nvidia-smi.exe >/dev/null 2>&1; then
    command -v nvidia-smi.exe
    return 0
  fi
  return 1
}

fail() {
  if [ "$MODE" != "quiet" ] && [ "$MODE" != "channel" ] && [ "$MODE" != "driver-major" ]; then
    printf '%s\n' "$1" >&2
  fi
  exit 1
}

NVIDIA_SMI="$(find_nvidia_smi || true)"
[ -n "$NVIDIA_SMI" ] || fail "[AnatoMask] 未检测到 nvidia-smi。当前版本仅支持 NVIDIA GPU + CUDA 驱动环境。"

DRIVER_VERSION="$("$NVIDIA_SMI" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r')"
[ -n "$DRIVER_VERSION" ] || fail "[AnatoMask] 无法从 nvidia-smi 读取驱动版本。"

DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
case "$DRIVER_MAJOR" in
  ''|*[!0-9]*)
    fail "[AnatoMask] 解析 NVIDIA 驱动版本失败: $DRIVER_VERSION"
    ;;
esac

TORCH_CHANNEL=""
if [ "$DRIVER_MAJOR" -ge 580 ]; then
  TORCH_CHANNEL="cu130"
elif [ "$DRIVER_MAJOR" -ge 525 ]; then
  TORCH_CHANNEL="cu126"
fi

[ -n "$TORCH_CHANNEL" ] || fail "[AnatoMask] NVIDIA 驱动版本过低: $DRIVER_VERSION。当前启动器仅支持 PyTorch CUDA 12.6 或 13.0 通道。"

if [ "$MODE" = "channel" ]; then
  printf '%s\n' "$TORCH_CHANNEL"
elif [ "$MODE" = "driver-major" ]; then
  printf '%s\n' "$DRIVER_MAJOR"
elif [ "$MODE" = "summary" ]; then
  printf '%s\n' "[AnatoMask] 检测到 NVIDIA 驱动版本: $DRIVER_VERSION"
  printf '%s\n' "[AnatoMask] 推荐 PyTorch CUDA 通道: $TORCH_CHANNEL"
  printf '%s\n' "[AnatoMask] 推荐安装命令: python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_CHANNEL"
fi
