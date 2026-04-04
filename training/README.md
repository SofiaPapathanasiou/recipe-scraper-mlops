# Training

This directory contains the containerized training and experimentation stack for the
recipe-correction model. The Docker Compose setup gives you:

- MinIO for dataset/model artifact storage
- PostgreSQL for the MLflow backend store
- MLflow for experiment tracking
- Redis and Redis Insight
- A Jupyter Lab container for interactive development
- A separate training job container for scripted train/tune runs

## Prerequisites

- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit if you want GPU access in `jupyter` or `training`
- A populated `.env` file in this directory

The Compose file automatically reads [`.env`](/home/cc/recipe-scraper-mlops/training/.env).
Important defaults there include:

- `JUPYTER_TOKEN` for Jupyter auth
- `TORCH_BUILD_MODE` with `auto`, `cpu`, or `gpu`
- `NUM_PROCESSES` for Accelerate/trial worker count
- `MLFLOW_EXTRA_ALLOWED_HOSTS` and `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` for MLflow UI access

## Container Layout

The long-running development stack is:

- `minio`
- `minio-init`
- `postgres`
- `mlflow`
- `redis`
- `redis-insight`
- `jupyter`

The `training` service is different: it is a job container behind the `training` profile,
so you typically run it on demand instead of keeping it up in the background.

## Build Images

Build the always-on stack:

```bash
docker compose build
```

Build the training image too:

```bash
docker compose --profile training build
```

Rebuild only the interactive dev image:

```bash
docker compose build jupyter
```

Rebuild only the training image:

```bash
docker compose --profile training build training
```

## Bring Up The Stack

Start all long-running containers in the background:

```bash
docker compose up -d
```

Watch startup logs:

```bash
docker compose logs -f
```

Check container status:

```bash
docker compose ps
```

Open the main UIs:

- MLflow: `http://localhost:5000`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`
- Redis Insight: `http://localhost:5540`
- Jupyter Lab: `http://localhost:8888`

If `JUPYTER_TOKEN` is blank, Docker/Jupyter may generate one at startup. Retrieve it with:

```bash
docker compose logs jupyter
```

If you want to start only the infra without Jupyter:

```bash
docker compose up -d minio minio-init postgres mlflow redis redis-insight
```

If you want to restart just one service after config changes:

```bash
docker compose up -d --force-recreate mlflow
```

## Interactive Development In Jupyter

The `jupyter` service is the main interactive development environment.

It bind-mounts this repo into `/app`, exposes Jupyter Lab on port `8888`, and runs
[`scripts/select_torch_build.sh`](/home/cc/recipe-scraper-mlops/training/scripts/select_torch_build.sh)
on startup so the container installs the CPU or GPU PyTorch extra that matches
`TORCH_BUILD_MODE`.

Start Jupyter if the rest of the stack is already up:

```bash
docker compose up -d jupyter
```

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
docker compose exec jupyter python train.py --config /app/config.yaml --mode train
```

Launch the notebook workbench from Jupyter Lab:

- Open `notebooks/train_workbench.ipynb`
- Use it to inspect the runtime config and launch the same training paths used by the container workflows

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

If you want a CPU-only interactive session even on a GPU host:

```bash
TORCH_BUILD_MODE=cpu docker compose up -d --build jupyter
```

If you want to force the GPU build:

```bash
TORCH_BUILD_MODE=gpu docker compose up -d --build jupyter
```

## Training Container Overview

The `training` service uses [`scripts/run_training.sh`](/home/cc/recipe-scraper-mlops/training/scripts/run_training.sh)
as its entrypoint.

Behavior by mode:

- `TRAIN_MODE=train`: launches `train.py` through `accelerate`
- `TRAIN_MODE=tune`: runs `train.py --mode tune` directly, and tune mode launches its own Accelerate trial workers internally

Key environment variables:

- `TRAIN_MODE`: `train` or `tune`
- `TRAIN_CONFIG`: config path inside the container, default `config.yaml`
- `TRAIN_EXTRA_ARGS`: extra CLI flags appended to the training command
- `NUM_PROCESSES`: number of Accelerate processes or tune trial worker count
- `ACCELERATE_CONFIG_FILE`: Accelerate config path, default `accelerate_config.yaml`

The training container is not started by `docker compose up -d` unless you explicitly
run it with the `training` profile.

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
  -e TRAIN_CONFIG=/app/config.yaml \
  training
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
python train.py --config config.yaml --mode train
```

Or run tune mode directly:

```bash
python train.py --config config.yaml --mode tune
```

## What Train And Tune Actually Do

Standard training:

```bash
python train.py --config config.yaml --mode train
```

- Uses [`config.yaml`](/home/cc/recipe-scraper-mlops/training/config.yaml)
- Defaults to `data.source: mock` so local smoke-test runs work without uploading MinIO dataset objects
- Tracks runs in MLflow
- Saves checkpoints under `/app/checkpoints/...`
- In the training container, this path is launched via `accelerate`

To train against MinIO-backed JSONL data instead, change [`config.yaml`](/home/cc/recipe-scraper-mlops/training/config.yaml) to:

```yaml
data:
  source: minio
  minio_bucket: recipe-datasets
  minio_train_key: train.jsonl
  minio_val_key: val.jsonl
```

Those objects must already exist in the bucket. The built-in `minio-init` service creates buckets only; it does not upload dataset files.

Hyperparameter tuning:

```bash
python train.py --config config.yaml --mode tune
```

- Uses Optuna settings and search space from `config.yaml`
- Uses the tuning experiment name as the Optuna study name
- Creates one top-level MLflow run per Optuna trial within the tuning experiment
- Launches each trial with `accelerate` across the requested visible GPUs
- Writes the winning resolved config to:

```text
<checkpoint_dir>/optuna/<tuning_experiment_name>/best_config.yaml
```

Override the experiment name for either mode:

```bash
python train.py --config config.yaml --mode train --experiment-name my-experiment
python train.py --config config.yaml --mode tune --experiment-name my-experiment
```

Without a CLI override:

- `train` uses `mlflow.experiment_name`
- `tune` uses `mlflow.tuning_experiment_name`

## Typical Workflows

Bring up the full dev stack, then iterate in Jupyter:

```bash
docker compose up -d
docker compose exec jupyter bash
```

Run an interactive experiment from inside Jupyter:

```bash
python train.py --config /app/config.yaml --mode train --experiment-name notebook-dev
```

Run a clean scripted training job in the separate training container:

```bash
docker compose --profile training run --rm training
```

Run a clean tuning study:

```bash
docker compose --profile training run --rm -e TRAIN_MODE=tune training
```

## Logs, Status, And Cleanup

Follow logs for one service:

```bash
docker compose logs -f jupyter
docker compose logs -f mlflow
docker compose logs -f training
```

Stop the long-running stack:

```bash
docker compose down
```

Stop the stack and remove named volumes:

```bash
docker compose down -v
```

Remove and rebuild from scratch:

```bash
docker compose down -v
docker compose --profile training build --no-cache
docker compose up -d
```

## Notes

- The Jupyter container bind-mounts this repo to `/app`, so code edits on the host are visible immediately.
- The Jupyter service also mounts `../.git` read-only into `/app/.git`, which helps preserve repo context in the container.
- The `training` service mounts `./data` and `./checkpoints` so outputs persist on the host.
- If no GPUs are available, tune mode falls back to a single process.
