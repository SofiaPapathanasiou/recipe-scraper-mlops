# =============================================================================
# Training Container — T5 Recipe Correction
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) — install Python deps into a virtualenv
#   Stage 2 (runtime) — copy only the venv + source, no build tools
#
# Base image: pytorch/pytorch with CUDA 12.1 + cuDNN 8
# Matches the CUDA version on Chameleon GPU nodes (RTX 6000 / A100)
#
# Build:   docker compose build training
# Run:     docker compose run --rm training python train.py
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS builder

# Build-time only tools (not copied to final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtualenv so the runtime stage gets a clean copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first — older pip misresolves some HuggingFace deps
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install Python dependencies
# Copied before source code so Docker layer cache is reused on code-only changes
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS runtime

LABEL maintainer="mlops-class"
LABEL description="T5 recipe correction training container"

# Minimal runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        # libgomp1 is required by some HuggingFace tokenizer backends
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from builder — no pip/gcc/make in final image
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Ensures Python uses the venv even if called as `python3`
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /app

# Copy training source code
# .dockerignore excludes: data/, checkpoints/, __pycache__/, *.pyc, .env
COPY . .

# ── Runtime environment defaults ──────────────────────────────────────────────
# These are overridden by docker-compose.yml environment: block at runtime.
# Set here so the container is runnable standalone (e.g. for local testing).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    # HuggingFace cache — mount a volume here to persist downloaded model weights
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    # MLFlow — overridden by compose but needed as a fallback
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# Pre-create cache directory (avoids permission errors if volume not mounted)
RUN mkdir -p /app/.cache/huggingface /app/checkpoints /app/data/raw /app/data/processed

# Non-root user for security — matches UID 1000 (default Chameleon cc user)
RUN useradd -m -u 1000 trainer && chown -R trainer:trainer /app
USER trainer

# Health check — verifies Python env and key imports are intact
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch, transformers, mlflow, redis, boto3; print('OK')" || exit 1

# Default entrypoint — override with `docker compose run training python <script>`
CMD ["python", "train.py"]
