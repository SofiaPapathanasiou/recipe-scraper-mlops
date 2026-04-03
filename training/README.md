# Training

Run a single training job with the config-driven defaults:

```bash
python train.py --config config.yaml --mode train
```

Override the MLflow experiment name for a one-off run:

```bash
python train.py --config config.yaml --mode train --experiment-name my-experiment
```

# Hyperparameter Tuning

Run an Optuna study with all trial settings and search spaces sourced from `config.yaml`:

```bash
python train.py --config config.yaml --mode tune
```

To group the parent study run and every nested Optuna trial under a specific MLflow
experiment, pass the same CLI override:

```bash
python train.py --config config.yaml --mode tune --experiment-name my-experiment
```

Tune mode now runs as a single Optuna controller that launches each trial with `accelerate`
across the visible GPUs. In Docker, the training service already defaults `NUM_PROCESSES=2`,
so a tune trial will use both GPUs unless you override that value.

Without the CLI override, tune mode uses `mlflow.tuning_experiment_name`.
Each study creates one parent MLflow run with nested child runs for individual trials,
and all runs are renamed to the `YYYY-MM-DD-<mlflow_run_id>` convention.

The winning resolved configuration is exported to:

```text
<checkpoint_dir>/optuna/<study_name>/best_config.yaml
```
