import copy
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import optuna
import torch
import yaml

DEFAULT_MLFLOW_TRACKING_URI = "file:///app/mlruns"

@dataclass
class TrainingContext:
    mode: str = "train"
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_nested: bool = False
    mlflow_parent_run_id: str | None = None
    mlflow_tags: dict[str, Any] = field(default_factory=dict)
    register_model: bool | None = None
    trial: optuna.trial.Trial | None = None
    objective_direction: str | None = None
    objective_metric_name: str | None = None
    trial_params: dict[str, Any] = field(default_factory=dict)
    progress_file: str | None = None
    prune_signal_file: str | None = None
    result_file: str | None = None


@dataclass
class TrainingResult:
    best_metric: float
    best_metric_name: str
    best_checkpoint: str | None
    run_id: str | None
    final_metrics: dict[str, float]
    resolved_config: dict[str, Any]


def serialize_training_context(context: TrainingContext) -> dict[str, Any]:
    return {
        "mode": context.mode,
        "mlflow_experiment_name": context.mlflow_experiment_name,
        "mlflow_run_name": context.mlflow_run_name,
        "mlflow_nested": context.mlflow_nested,
        "mlflow_parent_run_id": context.mlflow_parent_run_id,
        "mlflow_tags": context.mlflow_tags,
        "register_model": context.register_model,
        "objective_direction": context.objective_direction,
        "objective_metric_name": context.objective_metric_name,
        "trial_params": context.trial_params,
        "progress_file": context.progress_file,
        "prune_signal_file": context.prune_signal_file,
        "result_file": context.result_file,
    }


def deserialize_training_context(payload: dict[str, Any]) -> TrainingContext:
    return TrainingContext(
        mode=payload.get("mode", "train"),
        mlflow_experiment_name=payload.get("mlflow_experiment_name"),
        mlflow_run_name=payload.get("mlflow_run_name"),
        mlflow_nested=bool(payload.get("mlflow_nested", False)),
        mlflow_parent_run_id=payload.get("mlflow_parent_run_id"),
        mlflow_tags=dict(payload.get("mlflow_tags", {})),
        register_model=payload.get("register_model"),
        objective_direction=payload.get("objective_direction"),
        objective_metric_name=payload.get("objective_metric_name"),
        trial_params=dict(payload.get("trial_params", {})),
        progress_file=payload.get("progress_file"),
        prune_signal_file=payload.get("prune_signal_file"),
        result_file=payload.get("result_file"),
    )


def serialize_training_result(result: TrainingResult) -> dict[str, Any]:
    return {
        "best_metric": result.best_metric,
        "best_metric_name": result.best_metric_name,
        "best_checkpoint": result.best_checkpoint,
        "run_id": result.run_id,
        "final_metrics": result.final_metrics,
        "resolved_config": result.resolved_config,
    }


def deserialize_training_result(payload: dict[str, Any]) -> TrainingResult:
    return TrainingResult(
        best_metric=float(payload["best_metric"]),
        best_metric_name=str(payload["best_metric_name"]),
        best_checkpoint=payload.get("best_checkpoint"),
        run_id=payload.get("run_id"),
        final_metrics=dict(payload["final_metrics"]),
        resolved_config=dict(payload["resolved_config"]),
    )


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened



def write_yaml_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def sanitize_study_name(study_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", study_name.strip())
    return cleaned.strip("-") or "optuna-study"


def get_mlflow_experiment_name(cfg: dict[str, Any], mode: str) -> str:
    if mode == "tune":
        return cfg["mlflow"].get("active_experiment_name") or cfg["mlflow"]["tuning_experiment_name"]
    return cfg["mlflow"].get("active_experiment_name") or cfg["mlflow"]["experiment_name"]


def get_optuna_study_name(cfg: dict[str, Any]) -> str:
    return get_mlflow_experiment_name(cfg, "tune")


def resolve_mlflow_tracking_uri(cfg: dict[str, Any]) -> str:
    configured = str(
        os.getenv("MLFLOW_TRACKING_URI")
        or cfg.get("mlflow", {}).get("tracking_uri")
        or DEFAULT_MLFLOW_TRACKING_URI
    ).strip()
    return configured or DEFAULT_MLFLOW_TRACKING_URI


def resolve_num_processes(cfg: dict[str, Any]) -> int:
    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = os.getenv("NUM_PROCESSES")
    if requested is None or requested.strip() == "":
        requested = cfg.get("accelerate", {}).get("num_processes", "auto")

    if requested is None:
        requested = "auto"

    requested_text = str(requested).strip().lower()
    if requested_text in {"", "auto"}:
        return available_gpu_count if available_gpu_count > 0 else 1

    try:
        parsed = int(requested_text)
    except ValueError as error:
        raise ValueError(
            f"NUM_PROCESSES must be an integer or 'auto', got {requested!r}."
        ) from error

    if parsed < 1:
        raise ValueError(f"NUM_PROCESSES must be at least 1, got {parsed}.")
    if available_gpu_count == 0:
        return 1
    return min(parsed, available_gpu_count)


def resolve_mixed_precision(cfg: dict[str, Any]) -> str:
    requested = os.getenv("ACCELERATE_MIXED_PRECISION")
    if requested is None or requested.strip() == "":
        requested = cfg.get("accelerate", {}).get("mixed_precision", "no")

    requested_text = str(requested).strip().lower()
    if requested_text == "":
        requested_text = "no"

    valid_values = {"no", "fp16", "bf16", "fp8"}
    if requested_text not in valid_values:
        raise ValueError(
            "mixed precision must be one of "
            f"{sorted(valid_values)}, got {requested!r}."
        )
    return requested_text


def ensure_mlflow_experiment(cfg: dict[str, Any], experiment_name: str) -> str:
    tracking_uri = resolve_mlflow_tracking_uri(cfg)
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id

    try:
        return client.create_experiment(experiment_name)
    except Exception:
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            raise
        return existing.experiment_id


def format_mlflow_run_name(run_id: str) -> str:
    return f"{time.strftime('%Y-%m-%d')}-{run_id}"


def get_nested_value(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown config path: {dotted_path}")
        current = current[part]
    return current


def set_nested_value(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current: dict[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            raise KeyError(f"Unknown config path: {dotted_path}")
        current = next_value
    if parts[-1] not in current:
        raise KeyError(f"Unknown config path: {dotted_path}")
    current[parts[-1]] = value


def infer_metric_direction(metric_name: str) -> str:
    return "minimize" if "loss" in metric_name.lower() else "maximize"


def is_better_metric(candidate: float, best: float, direction: str) -> bool:
    if direction == "minimize":
        return candidate < best
    return candidate > best


def initial_best_metric(direction: str) -> float:
    return float("inf") if direction == "minimize" else -float("inf")


def best_metric_to_log(best_metric: float, direction: str) -> float:
    if direction == "minimize":
        return best_metric if best_metric < float("inf") else 0.0
    return best_metric if best_metric > -float("inf") else 0.0


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_central_optuna_config(config_dir: Path, model_name: str) -> dict[str, Any]:
    optuna_config_path = config_dir / "optuna.yaml"
    if not optuna_config_path.exists():
        return {}

    with open(optuna_config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Optuna config in {optuna_config_path}: expected a mapping.")

    default_cfg = payload.get("default", {})
    model_cfgs = payload.get("models", {})
    if default_cfg is None:
        default_cfg = {}
    if model_cfgs is None:
        model_cfgs = {}
    if not isinstance(default_cfg, dict):
        raise ValueError(f"Invalid Optuna config in {optuna_config_path}: 'default' must be a mapping.")
    if not isinstance(model_cfgs, dict):
        raise ValueError(f"Invalid Optuna config in {optuna_config_path}: 'models' must be a mapping.")

    model_cfg = model_cfgs.get(model_name, {})
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        raise ValueError(
            f"Invalid Optuna config in {optuna_config_path}: models.{model_name} must be a mapping."
        )

    return deep_merge_dicts(default_cfg, model_cfg)


def _resolve_env_or_config_path(path_value: str, config_dir: Path) -> Path:
    raw_path = Path(os.path.expandvars(os.path.expanduser(path_value)))
    if raw_path.is_absolute():
        return raw_path
    return (config_dir / raw_path).resolve()


def load_config(yaml_path: str) -> dict[str, Any]:
    config_dir = Path(yaml_path).resolve().parent
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config.setdefault("data", {})
    checkpointing_cfg = config.setdefault("checkpointing", {})
    huggingface_cfg = config.setdefault("huggingface", {})

    env_data_dir = os.getenv("TRAINING_DATA_DIR") or os.getenv("DATA_DIR")
    if env_data_dir:
        data_cfg["data_dir"] = str(_resolve_env_or_config_path(env_data_dir, config_dir))

    env_checkpoint_dir = os.getenv("TRAINING_CHECKPOINT_DIR") or os.getenv("CHECKPOINT_DIR")
    if env_checkpoint_dir:
        checkpointing_cfg["checkpoint_dir"] = str(_resolve_env_or_config_path(env_checkpoint_dir, config_dir))

    env_hf_cache_dir = (
        os.getenv("TRAINING_HF_CACHE_DIR")
        or os.getenv("HF_CACHE_DIR")
        or os.getenv("HUGGINGFACE_CACHE_DIR")
    )
    if env_hf_cache_dir:
        huggingface_cfg["cache_dir"] = str(_resolve_env_or_config_path(env_hf_cache_dir, config_dir))

    data_source = data_cfg.get("source", "mock")
    if data_source not in {"mock", "jsonl"}:
        raise ValueError(
            f"Unsupported data.source {data_source!r}; expected one of ('mock', 'jsonl')."
        )
    if data_source == "jsonl":
        def resolve_config_path(path_value: str) -> Path:
            return _resolve_env_or_config_path(path_value, config_dir)

        env_train_path = os.getenv("TRAIN_JSONL_PATH") or os.getenv("TRAIN_DATA_PATH")
        env_eval_path = os.getenv("EVAL_JSONL_PATH") or os.getenv("EVAL_DATA_PATH")

        configured_data_dir = data_cfg.get("data_dir")
        configured_train_file = data_cfg.get("train_file", "train.jsonl")
        configured_eval_file = data_cfg.get("eval_file", "eval.jsonl")
        configured_train_path = data_cfg.get("train_path")
        configured_eval_path = data_cfg.get("eval_path")

        candidate_dirs: list[Path] = []
        if env_data_dir:
            candidate_dirs.append(Path(os.path.expandvars(os.path.expanduser(env_data_dir))))
        if configured_data_dir:
            candidate_dirs.append(resolve_config_path(str(configured_data_dir)))
        candidate_dirs.extend(
            [
                Path("/data"),
                (Path.cwd() / "data").resolve(),
                (config_dir.parent / "data").resolve(),
                (config_dir.parent.parent / "data").resolve(),
                Path("/app/data"),
            ]
        )

        unique_candidate_dirs: list[Path] = []
        for candidate_dir in candidate_dirs:
            if candidate_dir not in unique_candidate_dirs:
                unique_candidate_dirs.append(candidate_dir)

        def resolve_dataset_path(
            *,
            explicit_path: Any,
            env_path: str | None,
            default_file_name: str,
        ) -> Path:
            if env_path:
                return Path(os.path.expandvars(os.path.expanduser(env_path)))
            if explicit_path:
                return resolve_config_path(str(explicit_path))
            for candidate_dir in unique_candidate_dirs:
                candidate_path = candidate_dir / default_file_name
                if candidate_path.exists():
                    return candidate_path
            return unique_candidate_dirs[0] / default_file_name

        train_path = resolve_dataset_path(
            explicit_path=configured_train_path,
            env_path=env_train_path,
            default_file_name=str(configured_train_file),
        )
        eval_path = resolve_dataset_path(
            explicit_path=configured_eval_path,
            env_path=env_eval_path,
            default_file_name=str(configured_eval_file),
        )

        data_cfg["train_path"] = str(train_path)
        data_cfg["eval_path"] = str(eval_path)
        data_cfg["data_dir"] = str(train_path.parent)

    config["mlflow"].setdefault(
        "tuning_experiment_name",
        f"{config['mlflow']['experiment_name']}-optuna",
    )
    config["mlflow"].setdefault("fail_on_artifact_logging_error", False)
    config["mlflow"]["tracking_uri"] = resolve_mlflow_tracking_uri(config)
    file_optuna_cfg = config.get("optuna", {})
    if file_optuna_cfg is None:
        file_optuna_cfg = {}
    if not isinstance(file_optuna_cfg, dict):
        raise ValueError("optuna must be a mapping when defined in the training config.")
    model_name = str(config.get("model", {}).get("name", "")).strip()
    central_optuna_cfg = load_central_optuna_config(Path(yaml_path).resolve().parent, model_name)
    config["optuna"] = deep_merge_dicts(central_optuna_cfg, file_optuna_cfg)
    config["optuna"]["study_name"] = get_optuna_study_name(config)
    model_registry_cfg = config.setdefault("model_registry", {})
    model_registry_cfg.setdefault(
        "model_name",
        config.get("mlflow", {}).get("registered_model_name") or config["model"]["name"],
    )
    model_registry_cfg.setdefault("log_to_mlflow_model_registry", True)
    accelerate_cfg = config.get("accelerate")
    if accelerate_cfg is not None and not isinstance(accelerate_cfg, dict):
        raise ValueError("accelerate must be a mapping when defined in the training config.")
    return config


def resolve_default_config_path() -> str:
    training_root = Path(__file__).resolve().parent.parent
    new_path = training_root / "config" / "config.yaml"
    if new_path.exists():
        return str(new_path)
    return str(training_root / "config.yaml")


def load_training_context(context_path: str | None) -> TrainingContext | None:
    if context_path is None:
        return None
    with open(context_path, "r", encoding="utf-8") as handle:
        return deserialize_training_context(json.load(handle))


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl_file(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def prune_requested_via_file(context: TrainingContext) -> bool:
    if context.prune_signal_file is None:
        return False
    return Path(context.prune_signal_file).exists()


def write_training_result_payload(context: TrainingContext, payload: dict[str, Any]) -> None:
    if context.result_file is None:
        return
    write_json_file(Path(context.result_file), payload)

def ensure_supported_tune_runtime() -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            "Tune mode must be started as a single controller process. "
            "The controller launches each trial with Accelerate on the requested GPUs. "
            "Run tune mode directly, for example with TRAIN_MODE=tune in the Docker training container."
        )


def resolve_tune_num_processes(cfg: dict[str, Any]) -> int:
    return resolve_num_processes(cfg)


def build_accelerate_launch_command(
    cfg: dict[str, Any],
    *,
    script_path: str | Path,
    script_args: list[str],
    num_processes: int | None = None,
    accelerate_config_path: str | None = None,
) -> list[str]:
    resolved_num_processes = num_processes if num_processes is not None else resolve_num_processes(cfg)
    resolved_accelerate_config_path = accelerate_config_path or resolve_accelerate_config_path(cfg)
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(resolved_num_processes),
        "--config_file",
        str(resolved_accelerate_config_path),
        str(Path(script_path).resolve()),
        *script_args,
    ]


def build_accelerate_launch_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    child_env = dict(base_env or os.environ.copy())
    child_env["TRAIN_ACCELERATE_BOOTSTRAPPED"] = "1"
    child_env["TRAIN_EXTRA_ARGS_FORWARDED"] = "1"
    return child_env


def resolve_accelerate_config_path(
    cfg: dict[str, Any] | None = None,
    *,
    output_path: Path | None = None,
) -> str:
    embedded_cfg = cfg.get("accelerate") if isinstance(cfg, dict) else None
    if embedded_cfg is not None:
        if not isinstance(embedded_cfg, dict):
            raise ValueError("accelerate must be a mapping when defined in the training config.")
        if not embedded_cfg:
            raise ValueError("accelerate mapping is empty; remove it or provide Accelerate settings.")

        destination = output_path
        if destination is None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="accelerate-config-",
                suffix=".yaml",
                delete=False,
                encoding="utf-8",
            ) as handle:
                destination = Path(handle.name)
        write_yaml_file(destination, embedded_cfg)
        return str(destination)

    raise ValueError(
        "Config must define a non-empty 'accelerate' mapping for Accelerate launches."
    )


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



def resolve_hf_cache_dir(cfg: dict[str, Any]) -> Path:
    configured = cfg.get("huggingface", {}).get("cache_dir")
    if configured:
        return Path(configured).expanduser()
    checkpoint_dir = Path(cfg["checkpointing"]["checkpoint_dir"]).expanduser()
    return checkpoint_dir.parent / "huggingface-cache"


def ensure_hf_cache_env(cache_dir: Path) -> None:
    hub_dir = cache_dir / "hub"
    datasets_dir = cache_dir / "datasets"
    asset_dir = cache_dir / "assets"
    for directory in (cache_dir, hub_dir, datasets_dir, asset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_dir))
    os.environ.setdefault("HF_ASSETS_CACHE", str(asset_dir))


def resolve_model_source(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("local_path") or cfg["model"]["name"])


def prepare_model_cache(cfg: dict[str, Any]) -> str:
    model_source = resolve_model_source(cfg)
    source_path = Path(model_source).expanduser()
    if source_path.exists():
        return str(source_path)

    cache_dir = resolve_hf_cache_dir(cfg)
    ensure_hf_cache_env(cache_dir)
    model_name = cfg["model"]["name"]
    try:
        model_snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        print(
            f"[hf-cache] Reusing cached Hugging Face model '{model_name}' from {cache_dir}.",
            flush=True,
        )
    except Exception:
        print(
            f"[hf-cache] Cache miss for remote model '{model_name}'. "
            f"Downloading from Hugging Face into persistent cache {cache_dir}. "
            "This first run requires outbound network access; later runs reuse the cache.",
            flush=True,
        )
        model_snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
        )
    cfg.setdefault("model", {})["local_path"] = model_snapshot_path
    return model_snapshot_path
