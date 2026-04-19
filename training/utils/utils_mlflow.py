import copy
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
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
    resolve_mlflow_tracking_uri,
    sanitize_study_name,
    write_yaml_file,
)

BEST_CHECKPOINT_ARTIFACT_PATH = "checkpoints/best"
DEFAULT_MLFLOW_API_PRECHECK_TIMEOUT_SECONDS = 5.0
DEFAULT_MLFLOW_OPERATION_TIMEOUT_SECONDS = 30.0


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


def resolve_mlflow_timeout_seconds(env_var: str, default: float) -> float:
    raw_value = str(os.getenv(env_var, default)).strip()
    try:
        parsed = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{env_var} must be a positive number, got {raw_value!r}.") from error
    if parsed <= 0:
        raise ValueError(f"{env_var} must be greater than 0, got {parsed}.")
    return parsed


def resolve_mlflow_api_precheck_timeout_seconds() -> float:
    return resolve_mlflow_timeout_seconds(
        "MLFLOW_API_PRECHECK_TIMEOUT_SECONDS",
        DEFAULT_MLFLOW_API_PRECHECK_TIMEOUT_SECONDS,
    )


def resolve_mlflow_operation_timeout_seconds() -> float:
    return resolve_mlflow_timeout_seconds(
        "MLFLOW_OPERATION_TIMEOUT_SECONDS",
        DEFAULT_MLFLOW_OPERATION_TIMEOUT_SECONDS,
    )


def run_with_timeout(operation_name: str, timeout_seconds: float, fn: Any, *args: Any, **kwargs: Any) -> Any:
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def target() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=target, name=f"mlflow-timeout-{operation_name}", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(
            f"Timed out after {timeout_seconds:.1f}s while waiting for MLflow operation {operation_name!r}."
        )
    if "value" in error:
        raise error["value"]
    return result.get("value")


def build_mlflow_api_probe_url(cfg: dict[str, Any], experiment_name: str) -> str:
    tracking_uri = resolve_mlflow_tracking_uri(cfg).rstrip("/")
    encoded_name = urllib.parse.quote(experiment_name, safe="")
    return f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name?experiment_name={encoded_name}"


def probe_mlflow_api(cfg: dict[str, Any], experiment_name: str, timeout_seconds: float) -> tuple[int, str]:
    request = urllib.request.Request(build_mlflow_api_probe_url(cfg, experiment_name), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            body = response.read(256).decode("utf-8", errors="replace").strip()
            return status, body
    except urllib.error.HTTPError as exc:
        body = exc.read(256).decode("utf-8", errors="replace").strip()
        if exc.code < 500:
            return exc.code, body
        raise


def format_mlflow_probe_summary(status_code: int, body: str) -> str:
    compact_body = " ".join(body.split())
    if len(compact_body) > 160:
        compact_body = f"{compact_body[:157]}..."
    return f"http_status={status_code}, body={compact_body or '<empty>'}"



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
        operation_timeout = resolve_mlflow_operation_timeout_seconds()
        if artifact_path is None:
            run_with_timeout(
                f"log_artifact:{filename}",
                operation_timeout,
                mlflow.log_artifact,
                str(artifact_file),
            )
        else:
            run_with_timeout(
                f"log_artifact:{artifact_path}/{filename}",
                operation_timeout,
                mlflow.log_artifact,
                str(artifact_file),
                artifact_path=artifact_path,
            )


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
    *,
    timeout_seconds: float | None = None,
    status_logger: Any | None = None,
) -> str:
    tracking_uri = resolve_mlflow_tracking_uri(cfg)
    client = MlflowClient(tracking_uri=tracking_uri)
    operation_timeout = timeout_seconds or resolve_mlflow_operation_timeout_seconds()
    if status_logger is not None:
        status_logger(f"MLflow: calling get_experiment_by_name({experiment_name!r})")
    existing = run_with_timeout(
        "get_experiment_by_name",
        operation_timeout,
        client.get_experiment_by_name,
        experiment_name,
    )
    if existing is not None:
        if status_logger is not None:
            status_logger(f"MLflow: found experiment id={existing.experiment_id}")
        return existing.experiment_id

    try:
        if status_logger is not None:
            status_logger(f"MLflow: calling create_experiment({experiment_name!r})")
        experiment_id = run_with_timeout(
            "create_experiment",
            operation_timeout,
            client.create_experiment,
            experiment_name,
        )
        if status_logger is not None:
            status_logger(f"MLflow: created experiment id={experiment_id}")
        return experiment_id
    except Exception:
        if status_logger is not None:
            status_logger("MLflow: create_experiment raised; retrying get_experiment_by_name()")
        existing = run_with_timeout(
            "get_experiment_by_name",
            operation_timeout,
            client.get_experiment_by_name,
            experiment_name,
        )
        if existing is None:
            raise
        if status_logger is not None:
            status_logger(f"MLflow: recovered existing experiment id={existing.experiment_id}")
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
            ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
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
        try:
            repo_root = find_git_repo_root(Path(__file__).resolve().parent)
            git_cmd = ["git", "rev-parse", "HEAD"]
            if repo_root is not None:
                git_cmd = ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
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
        run_with_timeout(
            "log_artifact:model_summary",
            resolve_mlflow_operation_timeout_seconds(),
            mlflow.log_artifact,
            summary_path,
            artifact_path="model_summary",
        )
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
    api_precheck_timeout = resolve_mlflow_api_precheck_timeout_seconds()
    operation_timeout = resolve_mlflow_operation_timeout_seconds()
    debug_log(
        accelerator,
        (
            f"MLflow startup diagnostics: experiment_name={experiment_name}, "
            f"api_precheck_timeout={api_precheck_timeout:.1f}s, operation_timeout={operation_timeout:.1f}s"
        ),
    )
    probe_status, probe_body = probe_mlflow_api(cfg, experiment_name, api_precheck_timeout)
    debug_log(
        accelerator,
        (
            "MLflow API probe completed before client startup: "
            f"{format_mlflow_probe_summary(probe_status, probe_body)}"
        ),
    )
    debug_log(accelerator, f"MLflow: set_tracking_uri({tracking_uri})")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = ensure_mlflow_experiment(
        cfg,
        experiment_name,
        timeout_seconds=operation_timeout,
        status_logger=lambda message: debug_log(accelerator, message),
    )
    debug_log(accelerator, f"MLflow: set_experiment(experiment_id={experiment_id})")
    mlflow.set_experiment(experiment_id=experiment_id)
    debug_log(
        accelerator,
        (
            "MLflow: start_run("
            f"run_name={context.mlflow_run_name!r}, nested={context.mlflow_nested}, "
            f"parent_run_id={context.mlflow_parent_run_id!r})"
        ),
    )
    run = run_with_timeout(
        "start_run",
        operation_timeout,
        mlflow.start_run,
        run_name=context.mlflow_run_name,
        nested=context.mlflow_nested,
        parent_run_id=context.mlflow_parent_run_id,
    )
    debug_log(accelerator, f"MLflow: start_run completed with run_id={run.info.run_id}")
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
    debug_log(accelerator, f"MLflow: log_params(config_params={len(flat_params)})")
    mlflow.log_params(sanitize_mlflow_params(flat_params))
    if context.trial_params:
        debug_log(accelerator, f"MLflow: log_params(trial_params={len(context.trial_params)})")
        mlflow.log_params(
            sanitize_mlflow_params(
                {f"trial_param.{key}": value for key, value in context.trial_params.items()}
            )
        )
    debug_log(accelerator, "MLflow: set_tags(runtime metadata)")
    mlflow.set_tags(build_mlflow_run_tags(cfg, context, status="running", environment_info=environment_info))
    debug_log(accelerator, "MLflow: log_artifact(model_summary)")
    log_model_summary(accelerator.unwrap_model(model))
    debug_log(accelerator, "MLflow: log_artifact(model_summary) completed")
    debug_log(accelerator, "MLflow: log_artifact(runtime/environment.yaml)")
    log_yaml_artifact(environment_info, "environment.yaml", artifact_path="runtime")
    debug_log(accelerator, "MLflow: log_artifact(runtime/environment.yaml) completed")
    debug_log(accelerator, "MLflow: log_artifact(config/resolved_config.yaml)")
    log_yaml_artifact(cfg, "resolved_config.yaml", artifact_path="config")
    debug_log(accelerator, "MLflow: log_artifact(config/resolved_config.yaml) completed")
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

    registered_model_name = str(model_registry_cfg.get("model_name") or cfg["model"]["name"])
    best_model = AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint)
    model_info = mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="best-model",
        registered_model_name=registered_model_name,
    )
    return {
        "registered_model_name": registered_model_name,
        "model_uri": model_info.model_uri,
    }
