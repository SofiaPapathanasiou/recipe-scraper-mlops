#!/bin/sh
set -eu

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/config.yaml}"
TRAIN_MODE="${TRAIN_MODE:-train}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

resolve_num_processes() {
    /opt/venv/bin/python - "${TRAIN_CONFIG}" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(f"TRAIN_CONFIG not found: {config_path}")

with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
requested = config.get("accelerate", {}).get("num_processes", "auto")

if requested is None:
    requested = "auto"

requested_text = str(requested).strip().lower()
if requested_text in {"", "auto"}:
    print(available_gpu_count if available_gpu_count > 0 else 1)
    raise SystemExit(0)

try:
    parsed = int(requested_text)
except ValueError as error:
    raise SystemExit(
        f"Invalid accelerate.num_processes value {requested!r}. Use an integer or 'auto'."
    ) from error

if parsed < 1:
    raise SystemExit(f"accelerate.num_processes must be at least 1, got {parsed}.")

if available_gpu_count == 0:
    print(1)
else:
    print(min(parsed, available_gpu_count))
PY
}

resolve_accelerate_config_file() {
    /opt/venv/bin/python - "${TRAIN_CONFIG}" <<'PY'
import sys
import tempfile
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(f"TRAIN_CONFIG not found: {config_path}")

with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

accelerate_cfg = config.get("accelerate")
if isinstance(accelerate_cfg, dict) and accelerate_cfg:
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="accelerate-config-",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(accelerate_cfg, tmp, sort_keys=False, allow_unicode=False)
    print(tmp.name)
else:
    raise SystemExit(
        "TRAIN_CONFIG must define a non-empty 'accelerate' mapping for Accelerate launches."
    )
PY
}

if [ "${TRAIN_MODE}" = "tune" ]; then
    TRAIN_EXTRA_ARGS_FORWARDED=1 exec /opt/venv/bin/python "${TRAIN_SCRIPT}" \
        --config "${TRAIN_CONFIG}" \
        --mode "${TRAIN_MODE}" \
        ${TRAIN_EXTRA_ARGS}
fi

ACCELERATE_CONFIG_PATH="$(resolve_accelerate_config_file)"
RESOLVED_NUM_PROCESSES="${NUM_PROCESSES:-$(resolve_num_processes)}"

TRAIN_EXTRA_ARGS_FORWARDED=1 exec /opt/venv/bin/python -m accelerate.commands.launch \
    --num_processes="${RESOLVED_NUM_PROCESSES}" \
    --config_file="${ACCELERATE_CONFIG_PATH}" \
    "${TRAIN_SCRIPT}" \
    --config "${TRAIN_CONFIG}" \
    --mode "${TRAIN_MODE}" \
    ${TRAIN_EXTRA_ARGS}
