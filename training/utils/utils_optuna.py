import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import optuna
import torch

from .utils_core import (
    TrainingContext,
    TrainingResult,
    append_jsonl_file,
    deserialize_training_result,
    ensure_hf_cache_env,
    get_nested_value,
    get_optuna_study_name,
    resolve_accelerate_config_path,
    resolve_hf_cache_dir,
    sanitize_study_name,
    serialize_training_context,
    set_nested_value,
    write_json_file,
    write_yaml_file,
)
from .utils_logging import emit_console_summary

def validate_optuna_config(cfg: dict[str, Any]) -> None:
    optuna_cfg = cfg.get("optuna")
    if not isinstance(optuna_cfg, dict):
        raise ValueError("Tune mode requires an optuna section in config.yaml.")

    if optuna_cfg.get("direction") not in {"maximize", "minimize"}:
        raise ValueError("optuna.direction must be either 'maximize' or 'minimize'.")

    if int(optuna_cfg.get("n_trials", 0)) <= 0:
        raise ValueError("optuna.n_trials must be a positive integer.")

    search_space = optuna_cfg.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("optuna.search_space must be a non-empty mapping.")

    for dotted_path, spec in search_space.items():
        get_nested_value(cfg, dotted_path)
        if not isinstance(spec, dict):
            raise ValueError(f"Search space for {dotted_path} must be a mapping.")
        spec_type = spec.get("type")
        if spec_type not in {"float", "int", "categorical"}:
            raise ValueError(f"Unsupported search-space type for {dotted_path}: {spec_type!r}")
        if spec_type in {"float", "int"}:
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"Search space for {dotted_path} must define low and high.")
            if spec.get("log") and "step" in spec:
                raise ValueError(f"Search space for {dotted_path} cannot use both log and step.")
        if spec_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Categorical search space for {dotted_path} must define choices.")


def build_optuna_sampler(sampler_cfg: dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_type = sampler_cfg.get("type", "tpe")
    seed = sampler_cfg.get("seed")
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_type == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported Optuna sampler type: {sampler_type!r}")


def build_optuna_pruner(pruner_cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_type = pruner_cfg.get("type", "median")
    if pruner_type in {"none", "nop"}:
        return optuna.pruners.NopPruner()
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 0)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 0)),
        )
    raise ValueError(f"Unsupported Optuna pruner type: {pruner_type!r}")


def sample_trial_params(
    trial: optuna.trial.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for dotted_path, spec in search_space.items():
        spec_type = spec["type"]
        if spec_type == "float":
            sampled[dotted_path] = trial.suggest_float(
                dotted_path,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
                step=spec.get("step"),
            )
        elif spec_type == "int":
            sampled[dotted_path] = trial.suggest_int(
                dotted_path,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
                step=int(spec.get("step", 1)),
            )
        else:
            sampled[dotted_path] = trial.suggest_categorical(dotted_path, spec["choices"])
    return sampled


def apply_trial_params(base_cfg: dict[str, Any], trial_params: dict[str, Any]) -> dict[str, Any]:
    resolved_cfg = copy.deepcopy(base_cfg)
    for dotted_path, value in trial_params.items():
        set_nested_value(resolved_cfg, dotted_path, value)
    return resolved_cfg


def resolve_study_output_dir(cfg: dict[str, Any]) -> Path:
    checkpoint_root = Path(cfg["checkpointing"]["checkpoint_dir"])
    return checkpoint_root / "optuna" / sanitize_study_name(get_optuna_study_name(cfg))


def build_trial_summary(study: optuna.study.Study) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for trial in study.trials:
        summary.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    return summary


def summarize_trial_counts(study: optuna.study.Study) -> dict[str, int]:
    counts = {
        "complete": 0,
        "failed": 0,
        "pruned": 0,
        "running": 0,
        "waiting": 0,
    }
    for trial in study.trials:
        state_name = trial.state.name.lower()
        counts[state_name] = counts.get(state_name, 0) + 1
    counts["total"] = len(study.trials)
    return counts


def load_progress_updates(progress_path: Path, seen_epochs: set[int]) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []

    updates: list[dict[str, Any]] = []
    with open(progress_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            epoch = int(payload["epoch"])
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            updates.append(payload)
    return updates


def run_distributed_trial(
    cfg: dict[str, Any],
    context: TrainingContext,
    trial: optuna.trial.Trial,
) -> TrainingResult:
    num_processes = resolve_tune_num_processes(cfg)
    script_path = Path(__file__).resolve().parent.parent / "train.py"

    with tempfile.TemporaryDirectory(prefix=f"optuna-trial-{trial.number:04d}-") as temp_dir:
        temp_root = Path(temp_dir)
        accelerate_config_path = resolve_accelerate_config_path(
            cfg,
            output_path=temp_root / "accelerate_config.yaml",
        )
        config_path = temp_root / "resolved_config.yaml"
        context_path = temp_root / "training_context.json"
        progress_path = temp_root / "progress.jsonl"
        prune_signal_path = temp_root / "prune.signal"
        result_path = temp_root / "result.json"

        child_context = copy.deepcopy(context)
        child_context.progress_file = str(progress_path)
        child_context.prune_signal_file = str(prune_signal_path)
        child_context.result_file = str(result_path)
        child_context.trial = None

        write_yaml_file(config_path, cfg)
        write_json_file(context_path, serialize_training_context(child_context))

        cache_dir = resolve_hf_cache_dir(cfg)
        ensure_hf_cache_env(cache_dir)
        child_env = os.environ.copy()

        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(num_processes),
            "--config_file",
            accelerate_config_path,
            str(script_path),
            "--config",
            str(config_path),
            "--mode",
            "train",
            "--context-file",
            str(context_path),
        ]
        emit_console_summary(
            print,
            f"TUNE TRIAL {trial.number:04d} START",
            {
                "study_name": context.mlflow_tags.get("study_name"),
                "trial_number": trial.number,
                "num_processes": num_processes,
                "objective_metric": context.objective_metric_name,
                "experiment_name": context.mlflow_experiment_name,
            },
        )
        print(f"[tune] command: {' '.join(command)}", flush=True)
        process = subprocess.Popen(command, env=child_env)

        seen_epochs: set[int] = set()
        while process.poll() is None:
            for update in load_progress_updates(progress_path, seen_epochs):
                current_metric = float(update["current_metric"])
                epoch = int(update["epoch"])
                emit_console_summary(
                    print,
                    f"TUNE TRIAL {trial.number:04d} EPOCH {epoch}",
                    {
                        "global_step": update["global_step"],
                        "objective_metric": update["objective_metric_name"],
                        "current_value": current_metric,
                    },
                )
                trial.report(current_metric, step=epoch)
                if trial.should_prune() and not prune_signal_path.exists():
                    print(f"[tune] pruning requested for trial {trial.number} at epoch {epoch}.", flush=True)
                    prune_signal_path.write_text("1\n", encoding="utf-8")
            time.sleep(1.0)

        return_code = process.wait()
        for update in load_progress_updates(progress_path, seen_epochs):
            current_metric = float(update["current_metric"])
            epoch = int(update["epoch"])
            emit_console_summary(
                print,
                f"TUNE TRIAL {trial.number:04d} EPOCH {epoch}",
                {
                    "global_step": update["global_step"],
                    "objective_metric": update["objective_metric_name"],
                    "current_value": current_metric,
                },
            )
            trial.report(current_metric, step=epoch)
            if trial.should_prune() and not prune_signal_path.exists():
                print(f"[tune] pruning requested for trial {trial.number} at epoch {epoch}.", flush=True)
                prune_signal_path.write_text("1\n", encoding="utf-8")

        if not result_path.exists():
            raise RuntimeError(
                f"Distributed trial {trial.number} exited with code {return_code} before writing a result file."
            )

        with open(result_path, "r", encoding="utf-8") as handle:
            result_payload = json.load(handle)

        status = result_payload.get("status")
        if status == "complete":
            result = deserialize_training_result(result_payload["result"])
            emit_console_summary(
                print,
                f"TUNE TRIAL {trial.number:04d} COMPLETE",
                {
                    "run_id": result.run_id,
                    "best_metric_name": result.best_metric_name,
                    "best_metric": result.best_metric,
                    "best_checkpoint": result.best_checkpoint,
                    "exit_code": return_code,
                },
            )
            return result
        if status == "pruned":
            emit_console_summary(
                print,
                f"TUNE TRIAL {trial.number:04d} PRUNED",
                {
                    "exit_code": return_code,
                    "message": result_payload.get("message", f"Trial {trial.number} was pruned."),
                },
            )
            raise optuna.TrialPruned(result_payload.get("message", f"Trial {trial.number} was pruned."))

        error_message = result_payload.get("error_message", "unknown error")
        error_type = result_payload.get("error_type", "RuntimeError")
        emit_console_summary(
            print,
            f"TUNE TRIAL {trial.number:04d} FAILED",
            {
                "error_type": error_type,
                "error_message": error_message,
                "exit_code": return_code,
            },
        )
        raise RuntimeError(
            f"Distributed trial {trial.number} failed with {error_type}: {error_message} "
            f"(process exit code {return_code})."
        )
