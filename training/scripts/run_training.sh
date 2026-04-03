#!/bin/sh
set -eu

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config.yaml}"
TRAIN_MODE="${TRAIN_MODE:-train}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

if [ "${TRAIN_MODE}" = "tune" ]; then
    exec /opt/venv/bin/python "${TRAIN_SCRIPT}" \
        --config "${TRAIN_CONFIG}" \
        --mode "${TRAIN_MODE}" \
        ${TRAIN_EXTRA_ARGS}
fi

exec /opt/venv/bin/python -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES:-1}" \
    --config_file="${ACCELERATE_CONFIG_FILE:-accelerate_config.yaml}" \
    "${TRAIN_SCRIPT}" \
    --config "${TRAIN_CONFIG}" \
    --mode "${TRAIN_MODE}" \
    ${TRAIN_EXTRA_ARGS}
