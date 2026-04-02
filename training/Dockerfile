# =============================================================================
# Training Container — T5 Recipe Correction
# =============================================================================
# Purpose:
# - Production/runtime training container with only required runtime tools
# - Multi-stage build keeps the final image smaller than the dev container
#
# Build:   docker compose build training
# Run:     docker compose run --rm training python train.py
# =============================================================================

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

LABEL maintainer="mlops-class"
LABEL description="T5 recipe correction training container"

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (same mechanism as dev Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

# Install Python dependencies (cached independently of source changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra torch-gpu

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

LABEL maintainer="mlops-class"
LABEL description="T5 recipe correction training container"

# Minimal runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        # libgomp1 is required by some HuggingFace tokenizer backends
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN useradd -m -u 1000 trainer

# Copy training source code
# .dockerignore excludes runtime data and cache artifacts.
COPY --chown=trainer:trainer . .

# Keep defaults consistent with dev image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# Create expected dirs when volumes are not mounted yet
RUN mkdir -p /opt/venv /app/.cache/huggingface /app/checkpoints /app/data/raw /app/data/processed \
    && chown -R trainer:trainer /app
USER trainer

# Health check verifies the Python environment and key imports
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import accelerate, boto3, mlflow, torch, transformers, yaml; print('OK')" || exit 1

# Default entrypoint for training runs
CMD ["sh", "-c", "accelerate launch --num_processes=${NUM_PROCESSES:-1} --config_file=${ACCELERATE_CONFIG_FILE:-accelerate_config.yaml} train.py"]
