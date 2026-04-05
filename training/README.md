# Training

This directory contains the containerized training and experimentation stack for the
recipe-correction model. The local Docker Compose setup now runs only:

- A Jupyter Lab container for interactive development
- A separate training job container for scripted train/tune runs

Both containers log to the remote MLflow server at `http://129.114.26.23:8000/` by
default. You can override that at runtime with `MLFLOW_TRACKING_URI`.

## Prerequisites

- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit if you want GPU access in `jupyter` or `training`
- A populated `.env` file in this directory

The Compose file automatically reads [`.env`](/home/cc/recipe-scraper-mlops/training/.env).
Important defaults there include:

- `JUPYTER_TOKEN` for Jupyter auth
- `TORCH_BUILD_MODE` with `auto`, `cpu`, or `gpu`
- `NUM_PROCESSES` for Accelerate or tune worker count
- `MLFLOW_TRACKING_URI` to override the remote tracking server

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

Build both images:

```bash
docker compose --profile training build
```

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
- Remote MLflow UI: `http://129.114.26.23:8000/`

If `JUPYTER_TOKEN` is blank, Docker/Jupyter may generate one at startup. Retrieve it with:

```bash
docker compose logs jupyter
```

## Interactive Development In Jupyter

The `jupyter` service is the main interactive development environment.

It bind-mounts this repo into `/app`, exposes Jupyter Lab on port `8888`, and runs
[`scripts/select_torch_build.sh`](/home/cc/recipe-scraper-mlops/training/scripts/select_torch_build.sh)
on startup so the container installs the CPU or GPU PyTorch extra that matches
`TORCH_BUILD_MODE`.

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
- `MLFLOW_TRACKING_URI`: remote tracking server override

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
- Uses mock-generated data only in this refactor
- Tracks runs in the remote MLflow server
- Saves local checkpoints under `/app/checkpoints/...`
- Logs the best checkpoint directory to MLflow artifacts under `checkpoints/best`
- Optionally registers the best model in the MLflow Model Registry
- In the training container, this path is launched via `accelerate`

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

Bring up Jupyter, then iterate interactively:

```bash
docker compose up -d jupyter
docker compose exec jupyter bash
```

Run an interactive experiment from inside Jupyter:

```bash
python train.py --config config.yaml --mode train --experiment-name notebook-dev
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
