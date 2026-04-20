import copy
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import accelerate as accelerate_pkg
import mlflow
import mlflow.pytorch
import psutil
import torch
import transformers
import yaml
from accelerate import Accelerator
from mlflow.tracking import MlflowClient
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
)
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM

from .utils_core import (
    DEFAULT_MLFLOW_TRACKING_URI,
    TrainingContext,
    ensure_hf_cache_env,
    flatten_dict,
    get_mlflow_experiment_name,
    infer_metric_direction,
    resolve_mlflow_tracking_uri,
    sanitize_study_name,
    write_yaml_file,
)

BEST_CHECKPOINT_ARTIFACT_PATH = "checkpoints/best"


def filter_mlflow_run_params(cfg: dict[str, Any]) -> dict[str, Any]:
    filtered_cfg = copy.deepcopy(cfg)
    filtered_cfg.pop("optuna", None)
    flat_cfg = flatten_dict(filtered_cfg)
    return {key: flat_cfg[key] for key in MLFLOW_RUN_PARAM_ALLOWLIST if key in flat_cfg}


def sanitize_mlflow_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True)


def sanitize_mlflow_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: sanitize_mlflow_value(value) for key, value in params.items()}


MLFLOW_RUN_PARAM_ALLOWLIST = [
    "model.name",
    "tokenization.max_input_length",
    "tokenization.max_target_length",
    "training.learning_rate",
    "training.weight_decay",
    "training.num_epochs",
    "training.per_device_train_batch_size",
    "training.per_device_eval_batch_size",
    "training.gradient_accumulation_steps",
    "training.warmup_ratio",
    "training.seed",
    "evaluation.every_n_epochs",
    "evaluation.full_generation_every_n_epochs",
    "evaluation.interim_max_eval_batches",
    "evaluation.full_max_eval_batches",
    "evaluation.generation_max_new_tokens",
    "evaluation.metric_for_best_model",
]

MLFLOW_CONTEXT_TAG_ALLOWLIST = {
    "study_name",
    "trial_number",
}


def filter_mlflow_context_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in tags.items()
        if key in MLFLOW_CONTEXT_TAG_ALLOWLIST and value is not None
    }


def format_summary_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return "none"
    return str(value)


def emit_console_block(printer: Any, title: str, lines: list[str]) -> None:
    printer("")
    printer(LOG_DELIMITER)
    printer(title)
    for line in lines:
        printer(line)
    printer(LOG_DELIMITER)
    printer("")


def emit_console_summary(printer: Any, title: str, values: dict[str, Any]) -> None:
    lines = [f"{key:<24}: {format_summary_value(value)}" for key, value in values.items()]
    emit_console_block(printer, title, lines)


def debug_log(
    accelerator: Accelerator,
    message: str,
    *,
    main_process_only: bool = True,
    section: str | None = None,
) -> None:
    if main_process_only and not accelerator.is_main_process:
        return
    prefix = f"[rank {accelerator.process_index}]"
    if section:
        accelerator.print("")
        accelerator.print(f"{prefix} {LOG_SUBDELIMITER}")
        accelerator.print(f"{prefix} {section}")
    accelerator.print(f"{prefix} {message}")


def log_temp_artifact(content: str, filename: str, artifact_path: str | None = None) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_file = Path(temp_dir) / filename
        artifact_file.write_text(content, encoding="utf-8")
        if artifact_path is None:
            mlflow.log_artifact(str(artifact_file))
        else:
            mlflow.log_artifact(str(artifact_file), artifact_path=artifact_path)


def sanitize_artifact_value(value: Any) -> Any:
    if value is None:
        return value
    if type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, dict):
        return {str(key): sanitize_artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_artifact_value(item) for item in value]
    return str(value)


def log_yaml_artifact(data: dict[str, Any], filename: str, artifact_path: str | None = None) -> None:
    log_temp_artifact(
        yaml.safe_dump(sanitize_artifact_value(data), sort_keys=False, allow_unicode=False),
        filename=filename,
        artifact_path=artifact_path,
    )


def log_json_artifact(data: Any, filename: str, artifact_path: str | None = None) -> None:
    log_temp_artifact(
        json.dumps(sanitize_artifact_value(data), indent=2, sort_keys=True),
        filename=filename,
        artifact_path=artifact_path,
    )


def log_optuna_search_space_artifacts(cfg: dict[str, Any], experiment_name: str) -> None:
    optuna_cfg = cfg.get("optuna")
    if not isinstance(optuna_cfg, dict):
        return

    search_space = optuna_cfg.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        return

    filename = f"{sanitize_study_name(experiment_name)}.yaml"
    log_yaml_artifact(search_space, filename, artifact_path="Optuna")


def write_optuna_search_space_file(cfg: dict[str, Any], experiment_name: str) -> Path | None:
    optuna_cfg = cfg.get("optuna")
    if not isinstance(optuna_cfg, dict):
        return None

    search_space = optuna_cfg.get("search_space")
    if not isinstance(search_space, dict) or not search_space:
        return None

    checkpoint_root = Path(cfg["checkpointing"]["checkpoint_dir"])
    destination = checkpoint_root / "Optuna" / f"{sanitize_study_name(experiment_name)}.yaml"
    write_yaml_file(destination, search_space)
    return destination



def ensure_mlflow_experiment(
    cfg: dict[str, Any],
    experiment_name: str,
) -> str:
    tracking_uri = resolve_mlflow_tracking_uri(cfg)
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id

    try:
        experiment_id = client.create_experiment(experiment_name)
        return experiment_id
    except Exception:
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            raise
        return existing.experiment_id


def format_mlflow_run_name(run_id: str) -> str:
    return f"{time.strftime('%Y-%m-%d')}-{run_id}"



def mark_mlflow_run_pruned(
    epoch: int,
    global_step: int,
    objective_metric_name: str,
    objective_direction: str,
    current_metric: float | None = None,
) -> None:
    if mlflow.active_run() is None:
        return
    mlflow.set_tag("status", "pruned")
    if current_metric is not None:
        mlflow.log_metric("objective_value", current_metric, step=global_step)
    mlflow.log_metric("pruned_epoch", epoch, step=global_step)


def find_git_repo_root(start_path: Path) -> Path | None:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / ".git").exists():
            return candidate

    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(start_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = result.stdout.strip()
        return Path(repo_root) if repo_root else None
    except Exception:
        return None


def log_environment_info() -> dict[str, str]:
    def git_commit_hash() -> str:
        env_commit = (
            os.getenv("GIT_COMMIT_HASH")
            or os.getenv("GIT_COMMIT")
            or os.getenv("COMMIT_SHA")
            or os.getenv("CI_COMMIT_SHA")
        )
        if env_commit:
            return env_commit.strip() or "unknown"

        try:
            repo_root = find_git_repo_root(Path(__file__).resolve().parent)
            git_cmd = ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"]
            if repo_root is not None:
                git_cmd = ["git", "-c", "safe.directory=*", "-C", str(repo_root), "rev-parse", "HEAD"]
            result = subprocess.run(
                git_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            commit = result.stdout.strip()
            return commit or "unknown"
        except Exception:
            return "unknown"

    info: dict[str, str] = {
        "env.python_version": platform.python_version(),
        "env.platform": platform.platform(),
        "env.os": platform.system(),
        "env.hostname": socket.gethostname(),
        "env.cpu_count": str(psutil.cpu_count(logical=True)),
        "env.ram_total_gb": f"{psutil.virtual_memory().total / 1e9:.1f}",
        "env.cuda_version": str(torch.version.cuda or "none"),
        "env.torch_version": str(torch.__version__),
        "env.transformers_version": str(transformers.__version__),
        "env.accelerate_version": str(accelerate_pkg.__version__),
        "env.mlflow_version": str(mlflow.__version__),
        "env.git_commit_hash": git_commit_hash(),
    }

    gpu_names: list[str] = []
    gpu_memories: list[str] = []
    initialized = False
    try:
        # Prefer NVML when available because it works even if PyTorch has not touched every GPU.
        nvmlInit()
        initialized = True
        count = nvmlDeviceGetCount()
        for index in range(count):
            handle = nvmlDeviceGetHandleByIndex(index)
            gpu_names.append(nvmlDeviceGetName(handle).decode("utf-8"))
            memory = nvmlDeviceGetMemoryInfo(handle)
            gpu_memories.append(f"{memory.total / 1e9:.1f}")
    except Exception:
        # Fall back to PyTorch's device APIs if NVML is unavailable in the environment.
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(index))
                props = torch.cuda.get_device_properties(index)
                gpu_memories.append(f"{props.total_memory / 1e9:.1f}")
    finally:
        if initialized:
            nvmlShutdown()

    info["env.gpu_count"] = str(len(gpu_names))
    info["env.gpu_names"] = ", ".join(gpu_names) if gpu_names else "none"
    info["env.gpu_total_memory_gb"] = ", ".join(gpu_memories) if gpu_memories else "none"
    return info


def build_mlflow_run_params(
    cfg: dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Dataset,
    accelerator: Accelerator,
    mixed_precision: str,
    total_steps: int,
    warmup_steps: int,
    trainable_params: int,
) -> dict[str, Any]:
    params = filter_mlflow_run_params(cfg)
    params.update(
        {
            "accelerate.mixed_precision": mixed_precision,
            "effective_batch_size": (
                cfg["training"]["per_device_train_batch_size"]
                * cfg["training"]["gradient_accumulation_steps"]
                * accelerator.num_processes
            ),
            "total_optimizer_steps": total_steps,
            "warmup_steps": warmup_steps,
            "trainable_params": trainable_params,
            "num_train_examples": len(train_dataset),
            "num_val_examples": len(val_dataset),
        }
    )
    return params


def build_mlflow_run_tags(
    cfg: dict[str, Any],
    context: TrainingContext,
    status: str,
    environment_info: dict[str, str] | None = None,
) -> dict[str, str]:
    tags: dict[str, str] = {
        "mode": context.mode,
        "status": status,
        "model_name": str(cfg["model"]["name"]),
        "data_source": str(cfg["data"]["source"]),
    }
    if environment_info is not None:
        git_commit = environment_info.get("env.git_commit_hash")
        if git_commit:
            tags["git_commit_hash"] = git_commit
    if context.mlflow_tags:
        tags.update(filter_mlflow_context_tags(context.mlflow_tags))
    return tags


def log_model_summary(model: torch.nn.Module) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(str(model))
        summary_path = handle.name
    try:
        mlflow.log_artifact(summary_path, artifact_path="model_summary")
    finally:
        os.unlink(summary_path)


def resolve_run_checkpoint_dir(checkpoint_dir: Path, run_id: str | None) -> Path:
    if run_id:
        return checkpoint_dir / run_id
    return checkpoint_dir / f"manual-run-{int(time.time())}"


def resolve_best_checkpoint_dir(checkpoint_dir: Path, run_id: str | None) -> Path:
    return resolve_run_checkpoint_dir(checkpoint_dir, run_id) / "best"


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    epoch: int,
    metrics: dict[str, float],
) -> str:
    path = checkpoint_dir / f"epoch-{epoch:02d}"
    return save_checkpoint_to_path(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=path,
        metrics=metrics,
    )


def save_checkpoint_to_path(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_path: Path,
    metrics: dict[str, float],
) -> str:
    path = checkpoint_path
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            path,
            is_main_process=True,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(path)
        with open(path / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    accelerator.wait_for_everyone()
    return str(path)


def maybe_start_mlflow_run(
    cfg: dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Dataset,
    accelerator: Accelerator,
    mixed_precision: str,
    total_steps: int,
    warmup_steps: int,
    trainable_params: int,
    model: torch.nn.Module,
    context: TrainingContext,
) -> str | None:
    if not accelerator.is_main_process:
        return None

    experiment_name = context.mlflow_experiment_name or get_mlflow_experiment_name(cfg, context.mode)
    tracking_uri = resolve_mlflow_tracking_uri(cfg)
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = ensure_mlflow_experiment(cfg, experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run = mlflow.start_run(
        run_name=context.mlflow_run_name,
        nested=context.mlflow_nested,
        parent_run_id=context.mlflow_parent_run_id,
    )
    mlflow.set_tag("mlflow.runName", format_mlflow_run_name(run.info.run_id))

    environment_info = log_environment_info()
    flat_params = build_mlflow_run_params(
        cfg,
        train_dataset,
        val_dataset,
        accelerator,
        mixed_precision,
        total_steps,
        warmup_steps,
        trainable_params,
    )
    mlflow.log_params(sanitize_mlflow_params(flat_params))
    if context.trial_params:
        mlflow.log_params(
            sanitize_mlflow_params(
                {f"trial_param.{key}": value for key, value in context.trial_params.items()}
            )
        )
    mlflow.set_tags(build_mlflow_run_tags(cfg, context, status="running", environment_info=environment_info))
    log_model_summary(accelerator.unwrap_model(model))
    log_yaml_artifact(environment_info, "environment.yaml", artifact_path="runtime")
    log_yaml_artifact(cfg, "resolved_config.yaml", artifact_path="config")
    if context.mode == "tune":
        log_optuna_search_space_artifacts(cfg, experiment_name)
    return run.info.run_id


def log_best_checkpoint_artifacts(best_checkpoint: str | None) -> dict[str, str] | None:
    if not best_checkpoint:
        return None

    checkpoint_path = Path(best_checkpoint)
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise ValueError(f"Cannot log missing checkpoint directory to MLflow: {checkpoint_path}")

    mlflow.log_artifacts(str(checkpoint_path), artifact_path=BEST_CHECKPOINT_ARTIFACT_PATH)
    return {
        "artifact_path": BEST_CHECKPOINT_ARTIFACT_PATH,
        "local_path": str(checkpoint_path),
    }


def maybe_log_best_model_to_mlflow_registry(
    cfg: dict[str, Any],
    best_checkpoint: str | None,
) -> dict[str, str] | None:
    if not best_checkpoint:
        return None

    model_registry_cfg = cfg.get("model_registry", {})
    if not bool(model_registry_cfg.get("log_to_mlflow_model_registry", True)):
        return None

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Cannot register the best model in MLflow without an active MLflow run.")

    run_id = active_run.info.run_id
    registered_model_name = str(model_registry_cfg.get("model_name") or cfg["model"]["name"])
    tracking_uri = resolve_mlflow_tracking_uri(cfg)
    client = MlflowClient(tracking_uri=tracking_uri)
    best_model = AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint)

    try:
        client.get_registered_model(registered_model_name)
    except Exception:
        client.create_registered_model(registered_model_name)

    with tempfile.TemporaryDirectory(prefix="mlflow-best-model-") as temp_dir:
        local_model_dir = Path(temp_dir) / "best-model"
        mlflow.pytorch.save_model(
            pytorch_model=best_model,
            path=str(local_model_dir),
        )
        mlflow.log_artifacts(str(local_model_dir), artifact_path="best-model")

    model_source = f"runs:/{run_id}/best-model"
    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_source,
        run_id=run_id,
    )
    return {
        "registered_model_name": registered_model_name,
        "model_uri": f"models:/{registered_model_name}/{model_version.version}",
    }


def evaluate_model_registry_gate(
    cfg: dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
) -> dict[str, Any]:
    model_registry_cfg = cfg.get("model_registry", {})
    gate_metric_name = str(
        model_registry_cfg.get("registry_threshold_metric") or best_metric_name
    )
    if gate_metric_name != best_metric_name:
        raise ValueError(
            "model_registry.registry_threshold_metric must match the tracked best metric "
            f"({best_metric_name}) for registry gating."
        )
    threshold = model_registry_cfg.get("registry_threshold")

    decision: dict[str, Any] = {
        "enabled": threshold is not None,
        "passed": True,
        "metric_name": gate_metric_name,
        "metric_value": best_metric_value,
        "threshold": threshold,
        "direction": infer_metric_direction(gate_metric_name),
        "reason": "threshold_not_configured",
    }

    if threshold is None:
        return decision

    threshold_value = float(threshold)
    direction = str(
        model_registry_cfg.get("registry_threshold_direction") or infer_metric_direction(gate_metric_name)
    ).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError(
            "model_registry.registry_threshold_direction must be 'maximize' or 'minimize' when provided."
        )

    if direction == "minimize":
        passed = best_metric_value <= threshold_value
        comparator = "less_than_or_equal"
    else:
        passed = best_metric_value >= threshold_value
        comparator = "greater_than_or_equal"

    decision.update(
        {
            "passed": passed,
            "threshold": threshold_value,
            "direction": direction,
            "comparator": comparator,
            "reason": "passed" if passed else "threshold_not_met",
        }
    )
    return decision
