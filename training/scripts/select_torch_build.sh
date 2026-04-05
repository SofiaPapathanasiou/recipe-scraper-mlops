#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
VENV_PATH="${VENV_PATH:-/opt/venv}"
VENV_PYTHON="${VENV_PATH}/bin/python"
TORCH_BUILD_MODE="${TORCH_BUILD_MODE:-auto}"

gpu_visible=0
if [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]]; then
    gpu_visible=1
elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "void" && "${NVIDIA_VISIBLE_DEVICES}" != "none" ]]; then
    gpu_visible=1
fi

target_build="cpu"

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

target_extra="torch-cpu"
if [[ "${target_build}" == "gpu" ]]; then
    target_extra="torch-gpu"
fi

echo "Selecting PyTorch build: ${target_build}"
echo "Syncing uv extra: ${target_extra}"

cd "${PROJECT_ROOT}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Recreating missing virtual environment at ${VENV_PATH}"
    uv venv "${VENV_PATH}" --python 3.11
fi

export VIRTUAL_ENV="${VENV_PATH}"
export PATH="${VENV_PATH}/bin:${PATH}"
uv sync --frozen --active --group dev --extra "${target_extra}"

"${VENV_PYTHON}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_built={torch.backends.cuda.is_built()}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

if torch.__version__.endswith("+cpu") and torch.backends.cuda.is_built():
    raise SystemExit("Torch install verification failed: unexpected mixed CPU/CUDA state.")
PY
