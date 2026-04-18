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

import yaml

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(f"TRAIN_CONFIG not found: {config_path}")

with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

print(config.get("accelerate", {}).get("num_processes", 1))
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
