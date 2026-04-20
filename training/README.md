# Training

This directory contains the containerized training and experimentation stack for the
recipe-correction model. The local Docker Compose setup now runs only:

- A Jupyter Lab container for interactive development
- A separate training job container for scripted train/tune runs

Both containers use `MLFLOW_TRACKING_URI` for the remote MLflow server. The repo
default is a placeholder URI, so set the real endpoint in your `.env` file.

## Prerequisites

- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit
- A GPU-capable Docker host for the `training` service. The production training container is GPU-only by design.
- A populated `.env` file in this directory

The Compose file automatically reads [`.env`](/home/cc/recipe-scraper-mlops/training/.env).
Important defaults there include:

- `JUPYTER_TOKEN` for Jupyter auth
- `TORCH_BUILD_MODE` with `auto`, `cpu`, or `gpu`
- `NUM_PROCESSES` for Accelerate or tune worker count
  Defaults to `accelerate.num_processes`, which may be set to `auto`
- `MLFLOW_TRACKING_URI` to point at your actual remote tracking server

## Container Layout

The long-running development stack is:

- `jupyter`

The `training` service is a job container behind the `training` profile, so you
typically run it on demand instead of keeping it up in the background.

## Build Images

Build the Jupyter image:

```bash
docker compose build jupyter
```

Build the training image:

```bash
docker compose --profile training build training
```

Build and publish the training image to the cluster-local registry:

```bash
# required once per build host because the registry serves plain HTTP
sudo mkdir -p /etc/docker
cat <<'EOF' | sudo tee /etc/docker/daemon.json
{
  "insecure-registries": ["192.168.1.11:5000"]
}
EOF
sudo systemctl restart docker

docker build -t 192.168.1.11:5000/recipe-scraper-training:latest -f training/Dockerfile training
docker push 192.168.1.11:5000/recipe-scraper-training:latest
```

Build both images:

```bash
docker compose --profile training build
```

## Argo Workflow Integration

The cluster workflow resources use the same training container entrypoint and
config contract as the Kubernetes training Job template. The reusable workflow
definitions live under
[`devops/workflows`](/home/cc/recipe-scraper-mlops/devops/workflows/README.md:1).

Current default behavior:

- The scheduled retraining `CronWorkflow` is created suspended.
- This is intentional because, on April 20, 2026, the only GPU node is already
  fully reserved by Triton serving.
- After the training image is pushed and GPU capacity is available, you can
  unsuspend the `recipe-model-retraining` CronWorkflow.

## Bring Up Jupyter

Start Jupyter in the background:

```bash
docker compose up -d jupyter
```

Watch startup logs:

```bash
docker compose logs -f jupyter
```

Check container status:

```bash
docker compose ps
```

Open Jupyter Lab:

- Jupyter Lab: `http://localhost:8888`
- Remote MLflow UI: the `MLFLOW_TRACKING_URI` value from your `.env` file

If `JUPYTER_TOKEN` is blank, Docker/Jupyter may generate one at startup. Retrieve it with:

```bash
docker compose logs jupyter
```

## Interactive Development In Jupyter

The `jupyter` service is the main interactive development environment.

It bind-mounts this repo into `/app`, exposes Jupyter Lab on port `8888`, and runs
[`scripts/select_torch_build.sh`](/home/cc/recipe-scraper-mlops/training/scripts/select_torch_build.sh)
on startup so the container installs the PyTorch extra that matches `TORCH_BUILD_MODE`.

Open a shell inside the running Jupyter container:

```bash
docker compose exec jupyter bash
```

Run Python interactively inside the Jupyter container:

```bash
docker compose exec jupyter python
```

Run a one-off command in the Jupyter environment:

```bash
docker compose exec jupyter python train.py --config /app/config/config.yaml --mode train
```

Launch the notebook workbench from Jupyter Lab:

- Open `notebooks/train_workbench.ipynb`
- Use it to inspect the runtime config and launch the same `train.py` and Accelerate code paths used by the container workflows
- Leave `EXACT_RUNTIME_NUM_PROCESSES = None` to mirror `train.py` worker selection, or set it explicitly for a notebook-only override

Useful interactive development commands:

```bash
docker compose exec jupyter python -m pip list
docker compose exec jupyter python -c "import torch; print(torch.cuda.is_available())"
docker compose exec jupyter python -c "import mlflow; print(mlflow.get_tracking_uri())"
docker compose exec jupyter ls -la /app
```

If you change dependencies or want the torch variant reselected, recreate Jupyter:

```bash
docker compose up -d --build --force-recreate jupyter
```

If you want to force the GPU build:

```bash
TORCH_BUILD_MODE=gpu docker compose up -d --build jupyter
```

## Training Container Overview

The `training` service uses [`scripts/run_training.sh`](/home/cc/recipe-scraper-mlops/training/scripts/run_training.sh)
as its entrypoint.

The production `training` container requires Docker GPU runtime support and is expected
to run with `gpus: all`. Treat the Compose training service as GPU-only.

Behavior by mode:

- `TRAIN_MODE=train`: launches `train.py` through `accelerate`
- `TRAIN_MODE=tune`: runs `train.py --mode tune` directly, and tune mode launches its own Accelerate trial workers internally

If `train.py` is invoked directly in `train` mode without Accelerate rank env vars, it now re-execs itself through `accelerate.commands.launch` using the `accelerate:` block from the training config.

Key environment variables:

- `TRAIN_MODE`: `train` or `tune`
- `TRAIN_CONFIG`: config path inside the container, default `config/config.yaml`
- `TRAIN_EXTRA_ARGS`: extra default CLI flags applied before explicit command-line arguments
- `DATA_DIR` or `TRAINING_DATA_DIR`: optional base directory for JSONL training data discovery
- `data.num_workers`: DataLoader worker count per process; `auto` picks a bounded host-aware default
- `data.prefetch_factor`: batches prefetched by each DataLoader worker
- `evaluation.every_n_epochs`: run full validation every N epochs; final epoch can still be forced
- `TRAINING_CHECKPOINT_DIR` or `CHECKPOINT_DIR`: optional override for `checkpointing.checkpoint_dir`
- `TRAINING_HF_CACHE_DIR`, `HF_CACHE_DIR`, or `HUGGINGFACE_CACHE_DIR`: optional override for `huggingface.cache_dir`
- `NUM_PROCESSES`: number of Accelerate processes or tune trial worker count; `auto` uses visible GPU count
- `MLFLOW_TRACKING_URI`: remote tracking server override

Precision is also configurable in the YAML file through `accelerate.mixed_precision`.
Supported values are `no`, `fp16`, `bf16`, and `fp8`. The config value is now used by
the training worker directly, while `ACCELERATE_MIXED_PRECISION` can still override it
when you need an environment-level override for a specific run.

MLflow artifact uploads happen after the best checkpoint is already saved locally. If the
artifact server resets the connection during a large upload, training now keeps the run
successful by default and leaves the best checkpoint on disk. Set
`mlflow.fail_on_artifact_logging_error: true` if you want those upload failures to fail
the run instead.

With `checkpointing.save_intermediate_checkpoints: false`, training now persists only the
current best checkpoint locally under `<checkpoint_dir>/<run_id-or-manual-run>/best` so it
survives container or pod crashes before the final MLflow artifact upload.

The Hugging Face model cache should live on persistent storage. For Kubernetes, the
expected path is `/data/checkpoints/huggingface-cache`, which the Helm Job mounts from
the shared training volume. The first run may need outbound access to Hugging Face to
download the base model; later runs reuse that persistent cache. If egress is blocked,
prewarm the cache, enable the Helm cache-prewarm init container, or provide a local
model path instead of relying on runtime download.

The training container is not started by `docker compose up -d` unless you explicitly
run it with the `training` profile.

## Direct Docker Run Commands

Build the standalone training image from the repo root:

```bash
docker build -t recipe-training -f training/Dockerfile training
```

Use this base `docker run` command for direct container launches:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  recipe-training
```

In this repo, the JSONL datasets live under the repo-level [`data/`](/home/cc/recipe-scraper-mlops/data) directory,
so the `docker run` examples mount `$(pwd)/data` to `/app/data`. They also mount the
repo `.git` directory to `/app/.git` so MLflow tagging can resolve the current commit hash.

The entrypoint is [`scripts/run_training.sh`](/home/cc/recipe-scraper-mlops/training/scripts/run_training.sh), so the
container is configured through environment variables. The user-facing controls are:

- `-e TRAIN_MODE=train|tune`
- `-e TRAIN_CONFIG=/app/config/<file>.yaml`
- `-e TRAINING_DATA_DIR=/app/data`
- `-e TRAIN_JSONL_PATH=/app/data/train.jsonl`
- `-e EVAL_JSONL_PATH=/app/data/eval.jsonl`
- `-e NUM_PROCESSES=<n>`
- `-e MLFLOW_TRACKING_URI=<uri>`
- `-e TRAIN_EXTRA_ARGS="--experiment-name <name>"`

Examples for each supported runtime flag:

Run standard training:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  recipe-training
```

Run tune mode:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  -e TRAIN_MODE=tune \
  recipe-training
```

Use a different config file:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  -e TRAIN_CONFIG=/app/config/config.t5-small.yaml \
  recipe-training
```

Override Accelerate process count:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  -e NUM_PROCESSES=1 \
  recipe-training
```

Override the MLflow tracking server:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  recipe-training
```

Pass the forwarded `train.py` flag:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  -e 'TRAIN_EXTRA_ARGS=--experiment-name my-experiment' \
  recipe-training
```

Combine multiple flags in one run:

```bash
docker run --rm -it \
  --gpus all \
  -v "$(pwd)/training:/app" \
  -v "$(pwd)/.git:/app/.git:ro" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/training/checkpoints:/app/checkpoints" \
  --env-file training/.env \
  -e TRAIN_MODE=tune \
  -e TRAIN_CONFIG=/app/config/config.t5-base.yaml \
  -e NUM_PROCESSES=2 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e 'TRAIN_EXTRA_ARGS=--experiment-name recipe-tuning' \
  recipe-training
```

Run tune mode with explicit dataset file paths:

```bash
docker run --rm -it \
  --gpus all \
  -v /home/cc/recipe-scraper-mlops/training:/app \
  -v /home/cc/recipe-scraper-mlops/.git:/app/.git:ro \
  -v /home/cc/recipe-scraper-mlops/data:/app/data \
  -v /home/cc/recipe-scraper-mlops/training/checkpoints:/app/checkpoints \
  --env-file /home/cc/recipe-scraper-mlops/training/.env \
  -e TRAIN_MODE=tune \
  -e TRAIN_CONFIG=/app/config/config.t5-base.yaml \
  -e TRAIN_JSONL_PATH=/app/data/train.jsonl \
  -e EVAL_JSONL_PATH=/app/data/eval.jsonl \
  -e 'TRAIN_EXTRA_ARGS=--experiment-name T5-Small-BF16-Tune' \
  recipe-training
```

Use `TRAINING_DATA_DIR` when `train.jsonl` and `eval.jsonl` live together in the same
directory. Use `TRAIN_JSONL_PATH` and `EVAL_JSONL_PATH` when you want the paths to be
fully explicit or when the files are not being discovered where you expect. If you
cannot mount `.git`, you can also set `GIT_COMMIT_HASH` explicitly to preserve commit tagging.

The container entrypoint does not forward positional arguments from `docker run image ...`.
If you want to pass `train.py` CLI flags, use `TRAIN_EXTRA_ARGS`. The public `train.py`
flags are:

- `--config`
- `--mode`
- `--experiment-name`

## Common Training Commands

Run a standard training job:

```bash
docker compose --profile training run --rm training
```

Run training with an explicit mode override:

```bash
docker compose --profile training run --rm -e TRAIN_MODE=train training
```

Run hyperparameter tuning:

```bash
docker compose --profile training run --rm -e TRAIN_MODE=tune training
```

Run training with a custom MLflow experiment name:

```bash
docker compose --profile training run --rm \
  -e 'TRAIN_EXTRA_ARGS=--experiment-name my-experiment' \
  training
```

Run tune mode with a custom MLflow experiment name:

```bash
docker compose --profile training run --rm \
  -e TRAIN_MODE=tune \
  -e 'TRAIN_EXTRA_ARGS=--experiment-name my-tuning-experiment' \
  training
```

Use a different config file:

```bash
docker compose --profile training run --rm \
  -e TRAIN_CONFIG=/app/config/config.yaml \
  training
```

Set Accelerate launch options directly in your train config:

```yaml
accelerate:
  compute_environment: LOCAL_MACHINE
  distributed_type: MULTI_GPU
  num_processes: auto
  mixed_precision: bf16
  dynamo_backend: "no"
```

```yaml
evaluation:
  every_n_epochs: 2
  run_on_last_epoch: true
```

Override the number of processes:

```bash
docker compose --profile training run --rm \
  -e NUM_PROCESSES=1 \
  training
```

Run a multi-GPU training job:

```bash
docker compose --profile training run --rm \
  -e NUM_PROCESSES=2 \
  training
```

Run tune mode with a specific worker count:

```bash
docker compose --profile training run --rm \
  -e TRAIN_MODE=tune \
  -e NUM_PROCESSES=2 \
  training
```

Open an interactive shell in the training image instead of running the entrypoint:

```bash
docker compose --profile training run --rm --entrypoint bash training
```

From that shell, run the script directly:

```bash
python train.py --config config/config.yaml --mode train
```

Or run tune mode directly:

```bash
python train.py --config config/config.yaml --mode tune
```

## What Train And Tune Actually Do

Standard training:

```bash
python train.py --config config/config.yaml --mode train
```

- Uses [`config.yaml`](/home/cc/recipe-scraper-mlops/training/config/config.yaml)
- Uses mock-generated data only in this refactor
- Tracks runs in the remote MLflow server
- Saves local checkpoints under `/app/checkpoints/...`
- These storage paths can be redirected for Kubernetes or external volumes with `DATA_DIR`, `TRAINING_CHECKPOINT_DIR`, and `HF_CACHE_DIR`
- Reuses a persistent Hugging Face cache on later runs after the initial model download
- Logs the best checkpoint directory to MLflow artifacts under `checkpoints/best`
- Optionally registers the best model in the MLflow Model Registry
- In the training container, this path is launched via `accelerate`

Hyperparameter tuning:

```bash
python train.py --config config/config.yaml --mode tune
```

- Uses Optuna settings and search space from [`optuna.yaml`](/home/cc/recipe-scraper-mlops/training/config/optuna.yaml)
- Uses the tuning experiment name as the Optuna study name
- Creates one top-level MLflow run per Optuna trial within the tuning experiment
- Launches each trial with `accelerate` across the requested visible GPUs
- Writes the winning resolved config to:

```text
<checkpoint_dir>/optuna/<tuning_experiment_name>/best_config.yaml
```

Override the experiment name for either mode:

```bash
python train.py --config config/config.yaml --mode train --experiment-name my-experiment
python train.py --config config/config.yaml --mode tune --experiment-name my-experiment
```

Without a CLI override:

- `train` uses `mlflow.experiment_name`
- `tune` uses `mlflow.tuning_experiment_name`

## Typical Workflows

Bring up Jupyter, then iterate interactively:

```bash
docker compose up -d jupyter
docker compose exec jupyter bash
```

Run an interactive experiment from inside Jupyter:

```bash
python train.py --config config/config.yaml --mode train --experiment-name notebook-dev
```

Or use the notebook workbench for the same flows from Jupyter Lab:

- `MODE="train"` with `EXECUTION_MODE="interactive"` runs `train_worker(...)` in-kernel
- `MODE="train"` with `EXECUTION_MODE="exact-runtime"` launches `accelerate.commands.launch` against `train.py`
- `MODE="tune"` runs the real Optuna study controller path from `train.py`

Run a clean scripted training job in the separate training container:

```bash
docker compose --profile training run --rm training
```

Run a clean tuning study:

```bash
docker compose --profile training run --rm -e TRAIN_MODE=tune training
```

## Logs, Status, And Cleanup

Follow logs:

```bash
docker compose logs -f jupyter
docker compose logs -f training
```

Stop Jupyter:

```bash
docker compose stop jupyter
```

Remove containers:

```bash
docker compose down
```

Remove and rebuild from scratch:

```bash
docker compose down
docker compose --profile training build --no-cache
docker compose up -d jupyter
```

## Notes

- The Jupyter container bind-mounts this repo to `/app`, so code edits on the host are visible immediately.
- The Jupyter service also mounts `../.git` read-only into `/app/.git`, which helps preserve repo context in the container.
- The `training` service mounts `./data` and `./checkpoints` so outputs persist on the host.
- If no GPUs are available, tune mode falls back to a single process.
