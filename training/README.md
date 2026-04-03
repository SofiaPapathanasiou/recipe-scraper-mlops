# Training

Run a single training job with the config-driven defaults:

```bash
python train.py --config config.yaml --mode train
```

# Hyperparameter Tuning

Run an Optuna study with all trial settings and search spaces sourced from `config.yaml`:

```bash
python train.py --config config.yaml --mode tune
```

Tuning runs are logged to the MLflow experiment named by `mlflow.tuning_experiment_name`.
Each study creates one parent MLflow run with nested child runs for individual trials.

The winning resolved configuration is exported to:

```text
<checkpoint_dir>/optuna/<study_name>/best_config.yaml
```
