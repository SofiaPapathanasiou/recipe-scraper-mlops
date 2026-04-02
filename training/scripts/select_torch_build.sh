#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="${VIRTUAL_ENV:-/app/.venv}/bin/python"

TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"

CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
GPU_INDEX_URL="${PYTORCH_GPU_INDEX_URL:-}"
TORCH_BUILD_MODE="${TORCH_BUILD_MODE:-auto}"

gpu_visible=0
if [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]]; then
    gpu_visible=1
elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "void" && "${NVIDIA_VISIBLE_DEVICES}" != "none" ]]; then
    gpu_visible=1
fi

target_build="cpu"
index_url="${CPU_INDEX_URL}"
torch_spec="torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu"

case "${TORCH_BUILD_MODE}" in
    gpu)
        target_build="gpu"
        ;;
    cpu)
        target_build="cpu"
        ;;
    auto)
        if [[ "${gpu_visible}" -eq 1 ]]; then
            target_build="gpu"
        fi
        ;;
    *)
        echo "Unsupported TORCH_BUILD_MODE='${TORCH_BUILD_MODE}'. Use auto, cpu, or gpu." >&2
        exit 1
        ;;
esac

if [[ "${target_build}" == "gpu" ]]; then
    if [[ -z "${GPU_INDEX_URL}" ]]; then
        echo "GPU requested but PYTORCH_GPU_INDEX_URL is not set." >&2
        exit 1
    fi
    index_url="${GPU_INDEX_URL}"
    torch_spec="torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}"
fi

echo "Selecting PyTorch build: ${target_build}"
echo "Using wheel index: ${index_url}"

"${VENV_PYTHON}" -m pip install --no-cache-dir --upgrade --index-url "${index_url}" ${torch_spec}

"${VENV_PYTHON}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_built={torch.backends.cuda.is_built()}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
PY
