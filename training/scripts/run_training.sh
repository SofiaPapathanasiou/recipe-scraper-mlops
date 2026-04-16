#!/bin/sh
set -eu

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/config.yaml}"
TRAIN_MODE="${TRAIN_MODE:-train}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

resolve_accelerate_config_file() {
    if [ -n "${ACCELERATE_CONFIG_FILE:-}" ]; then
        printf "%s\n" "${ACCELERATE_CONFIG_FILE}"
        return
    fi

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
        "TRAIN_CONFIG must define a non-empty 'accelerate' mapping when "
        "ACCELERATE_CONFIG_FILE is not set."
    )
PY
}

if [ "${TRAIN_MODE}" = "tune" ]; then
    exec /opt/venv/bin/python "${TRAIN_SCRIPT}" \
        --config "${TRAIN_CONFIG}" \
        --mode "${TRAIN_MODE}" \
        ${TRAIN_EXTRA_ARGS}
fi

ACCELERATE_CONFIG_PATH="$(resolve_accelerate_config_file)"

exec /opt/venv/bin/python -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES:-1}" \
    --config_file="${ACCELERATE_CONFIG_PATH}" \
    "${TRAIN_SCRIPT}" \
    --config "${TRAIN_CONFIG}" \
    --mode "${TRAIN_MODE}" \
    ${TRAIN_EXTRA_ARGS}
