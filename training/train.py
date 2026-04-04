import argparse
import copy
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
from huggingface_hub import snapshot_download
import mlflow
import mlflow.pytorch
import optuna
import psutil
import torch
import yaml
from accelerate import Accelerator
import accelerate as accelerate_pkg
from accelerate.utils import set_seed
from evaluate import load as load_metric
from mlflow.tracking import MlflowClient
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import transformers


# =============================================================================
# Configuration helpers
# =============================================================================

MOCK_RECIPE_BLUEPRINTS = [
    {
        "title": "Lemon Garlic Chicken Pasta",
        "yield": "4 servings",
        "prep_time": "15 minutes",
        "cook_time": "25 minutes",
        "ingredients": [
            "8 ounces spaghetti",
            "1 pound boneless chicken breast, diced",
            "2 tablespoons olive oil",
            "3 cloves garlic, minced",
            "1 lemon, zested and juiced",
            "1/2 cup grated parmesan",
            "2 cups baby spinach",
            "1/2 teaspoon kosher salt",
            "1/4 teaspoon black pepper",
        ],
        "instructions": [
            "Cook the spaghetti in salted water until al dente, then reserve 1/2 cup pasta water and drain.",
            "Heat the olive oil in a skillet and cook the chicken until browned and cooked through.",
            "Stir in the garlic, lemon zest, lemon juice, salt, and pepper, then cook for 1 minute.",
            "Add the spinach and cooked pasta, tossing with the parmesan and reserved pasta water until glossy.",
        ],
        "notes": "Serve with extra parmesan and lemon wedges.",
    },
    {
        "title": "One-Bowl Banana Muffins",
        "yield": "12 muffins",
        "prep_time": "10 minutes",
        "cook_time": "20 minutes",
        "ingredients": [
            "3 ripe bananas, mashed",
            "1/2 cup melted butter",
            "1/2 cup brown sugar",
            "1 egg",
            "1 teaspoon vanilla extract",
            "1 1/2 cups all-purpose flour",
            "1 teaspoon baking soda",
            "1/2 teaspoon cinnamon",
            "1/4 teaspoon salt",
        ],
        "instructions": [
            "Preheat the oven to 350F and line a 12-cup muffin tin.",
            "Whisk the bananas, melted butter, brown sugar, egg, and vanilla until smooth.",
            "Fold in the flour, baking soda, cinnamon, and salt just until no dry streaks remain.",
            "Divide the batter between the cups and bake until the tops spring back lightly.",
        ],
        "notes": "A few chocolate chips or chopped walnuts can be folded in with the dry ingredients.",
    },
    {
        "title": "Sheet Pan Sausage and Vegetables",
        "yield": "4 servings",
        "prep_time": "15 minutes",
        "cook_time": "30 minutes",
        "ingredients": [
            "12 ounces smoked sausage, sliced",
            "1 red bell pepper, chopped",
            "1 zucchini, sliced",
            "1 small red onion, cut into wedges",
            "12 ounces baby potatoes, halved",
            "2 tablespoons olive oil",
            "1 teaspoon paprika",
            "1/2 teaspoon garlic powder",
            "1/2 teaspoon salt",
        ],
        "instructions": [
            "Heat the oven to 425F and line a sheet pan with parchment.",
            "Toss the sausage, bell pepper, zucchini, onion, and potatoes with the olive oil and seasonings.",
            "Spread everything in an even layer and roast until the vegetables are tender and caramelized.",
            "Stir halfway through cooking so the potatoes brown on multiple sides.",
        ],
        "notes": "Finish with chopped parsley or a squeeze of lemon if you have it.",
    },
    {
        "title": "Tomato Basil Soup",
        "yield": "6 servings",
        "prep_time": "10 minutes",
        "cook_time": "35 minutes",
        "ingredients": [
            "2 tablespoons butter",
            "1 yellow onion, diced",
            "3 cloves garlic, minced",
            "2 tablespoons tomato paste",
            "2 cans crushed tomatoes",
            "2 cups vegetable broth",
            "1/3 cup heavy cream",
            "1/4 cup basil leaves",
            "3/4 teaspoon salt",
        ],
        "instructions": [
            "Melt the butter in a soup pot and cook the onion until softened.",
            "Add the garlic and tomato paste, stirring until fragrant and slightly darkened.",
            "Pour in the crushed tomatoes and broth, then simmer for 25 minutes.",
            "Blend until smooth, then stir in the cream, basil, and salt before serving.",
        ],
        "notes": "A grilled cheese sandwich makes a good side.",
    },
    {
        "title": "Honey Soy Salmon Bowls",
        "yield": "4 servings",
        "prep_time": "20 minutes",
        "cook_time": "15 minutes",
        "ingredients": [
            "4 salmon fillets",
            "3 tablespoons soy sauce",
            "2 tablespoons honey",
            "1 tablespoon rice vinegar",
            "1 teaspoon grated ginger",
            "2 cups cooked jasmine rice",
            "1 cucumber, sliced",
            "1 avocado, sliced",
            "2 green onions, thinly sliced",
        ],
        "instructions": [
            "Whisk the soy sauce, honey, rice vinegar, and ginger in a shallow dish.",
            "Marinate the salmon for 10 minutes while the oven heats to 400F.",
            "Bake the salmon until flaky, brushing with the leftover marinade halfway through.",
            "Build bowls with rice, cucumber, avocado, and salmon, then top with green onions.",
        ],
        "notes": "Sesame seeds add crunch if you want a garnish.",
    },
    {
        "title": "Creamy Chickpea Curry",
        "yield": "4 servings",
        "prep_time": "10 minutes",
        "cook_time": "25 minutes",
        "ingredients": [
            "1 tablespoon coconut oil",
            "1 onion, diced",
            "2 cloves garlic, minced",
            "1 tablespoon grated ginger",
            "2 tablespoons curry powder",
            "2 cans chickpeas, drained",
            "1 can coconut milk",
            "1 cup diced tomatoes",
            "1/2 teaspoon salt",
        ],
        "instructions": [
            "Warm the coconut oil in a skillet and cook the onion until translucent.",
            "Add the garlic, ginger, and curry powder, stirring until fragrant.",
            "Stir in the chickpeas, coconut milk, tomatoes, and salt.",
            "Simmer until slightly thickened, then serve over rice or with naan.",
        ],
        "notes": "Baby spinach can be stirred in during the last 2 minutes of cooking.",
    },
    {
        "title": "Classic Pancakes",
        "yield": "10 pancakes",
        "prep_time": "10 minutes",
        "cook_time": "15 minutes",
        "ingredients": [
            "1 1/2 cups all-purpose flour",
            "2 tablespoons sugar",
            "2 teaspoons baking powder",
            "1/4 teaspoon salt",
            "1 1/4 cups milk",
            "1 egg",
            "2 tablespoons melted butter",
            "1 teaspoon vanilla extract",
        ],
        "instructions": [
            "Whisk the flour, sugar, baking powder, and salt in a bowl.",
            "In a second bowl, whisk the milk, egg, melted butter, and vanilla.",
            "Pour the wet mixture into the dry ingredients and stir just until combined.",
            "Cook 1/4-cup portions on a greased skillet until bubbles form and the pancakes are golden on both sides.",
        ],
        "notes": "Do not overmix or the pancakes will be tough.",
    },
    {
        "title": "Roasted Broccoli Mac and Cheese",
        "yield": "6 servings",
        "prep_time": "15 minutes",
        "cook_time": "35 minutes",
        "ingredients": [
            "12 ounces elbow macaroni",
            "1 head broccoli, cut into florets",
            "2 tablespoons olive oil",
            "3 tablespoons butter",
            "3 tablespoons flour",
            "2 cups milk",
            "2 cups shredded cheddar cheese",
            "1/2 teaspoon salt",
            "1/4 teaspoon mustard powder",
        ],
        "instructions": [
            "Roast the broccoli with the olive oil at 425F until crisp-tender.",
            "Cook the macaroni until just shy of al dente and drain.",
            "Make a roux with the butter and flour, whisk in the milk, then melt in the cheddar, salt, and mustard powder.",
            "Fold in the macaroni and broccoli, then bake until bubbling if you want a casserole-style finish.",
        ],
        "notes": "For a stovetop version, skip the final bake and serve immediately.",
    },
]


def format_mock_recipe(recipe: dict[str, Any]) -> str:
    ingredient_lines = "\n".join(f"- {item}" for item in recipe["ingredients"])
    instruction_lines = "\n".join(f"{index}. {step}" for index, step in enumerate(recipe["instructions"], start=1))
    return (
        f"Title: {recipe['title']}\n"
        f"Yield: {recipe['yield']}\n"
        f"Prep time: {recipe['prep_time']}\n"
        f"Cook time: {recipe['cook_time']}\n"
        "Ingredients:\n"
        f"{ingredient_lines}\n"
        "Instructions:\n"
        f"{instruction_lines}\n"
        f"Notes: {recipe['notes']}"
    )


def apply_word_level_recipe_noise(text: str) -> str:
    replacements = {
        "Title:": "Ttle:",
        "Yield:": "Yeild:",
        "Prep time:": "Prep tm:",
        "Cook time:": "Cook tm:",
        "Ingredients:": "Ingrednts:",
        "Instructions:": "Instrctions:",
        "Notes:": "Note:",
        "ounces": "oz",
        "tablespoons": "tbsp",
        "teaspoons": "tsp",
        "minutes": "mins",
        "boneless": "bonless",
        "chicken": "chikcen",
        "parmesan": "parmasan",
        "vegetable": "vegtable",
        "broccoli": "brocoli",
        "through": "thru",
        "until": "till",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.replace(", then", " then").replace(", and", " and")


def collapse_recipe_sections(text: str) -> str:
    collapsed = text.replace("\n- ", ", ").replace("\n", " | ")
    collapsed = collapsed.replace("Instructions: | 1. ", "Directions: ")
    collapsed = collapsed.replace(" | 2. ", " Next, ")
    collapsed = collapsed.replace(" | 3. ", " Then ")
    collapsed = collapsed.replace(" | 4. ", " Finally ")
    collapsed = collapsed.replace(" | Notes: ", " | Note ")
    return collapsed


def remove_recipe_punctuation(text: str) -> str:
    stripped = text.replace(":", "").replace(",", "").replace(".", "")
    stripped = stripped.replace("1/2", "1-2").replace("1/4", "1-4")
    stripped = stripped.replace("350F", "350 f").replace("425F", "425 f").replace("400F", "400 f")
    return stripped


def add_shorthand_recipe_noise(text: str) -> str:
    replacements = {
        "Preheat": "Pre-heat",
        "Whisk": "Mix up",
        "stir": "mix",
        "until": "til",
        "with the": "w/",
        "and": "&",
        "because": "bc",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.replace("Instructions:", "Steps:").replace("Notes:", "Tips:")


def build_mock_recipe_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for recipe in MOCK_RECIPE_BLUEPRINTS:
        target = format_mock_recipe(recipe)
        pairs.extend(
            [
                (apply_word_level_recipe_noise(target), target),
                (collapse_recipe_sections(apply_word_level_recipe_noise(target)), target),
                (remove_recipe_punctuation(target), target),
                (add_shorthand_recipe_noise(collapse_recipe_sections(target)), target),
            ]
        )
    return pairs


CORRUPTIONS = build_mock_recipe_pairs()


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


def filter_mlflow_run_params(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    filtered_cfg = copy.deepcopy(cfg)
    if mode != "tune":
        filtered_cfg.pop("optuna", None)
    return flatten_dict(filtered_cfg)


def sanitize_mlflow_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True)


def sanitize_mlflow_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: sanitize_mlflow_value(value) for key, value in params.items()}


LOG_DELIMITER = "=" * 88
LOG_SUBDELIMITER = "-" * 88


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


def log_yaml_artifact(data: dict[str, Any], filename: str, artifact_path: str | None = None) -> None:
    log_temp_artifact(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        filename=filename,
        artifact_path=artifact_path,
    )


def log_json_artifact(data: Any, filename: str, artifact_path: str | None = None) -> None:
    log_temp_artifact(
        json.dumps(data, indent=2, sort_keys=True),
        filename=filename,
        artifact_path=artifact_path,
    )


def sanitize_storage_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


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


def get_mlflow_artifact_root_uri(cfg: dict[str, Any]) -> str:
    configured = str(cfg.get("mlflow", {}).get("artifact_root_uri") or os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow-artifacts"))
    return configured.rstrip("/")


def get_mlflow_experiment_artifact_location(cfg: dict[str, Any], experiment_name: str) -> str:
    base_uri = get_mlflow_artifact_root_uri(cfg)
    experiment_component = sanitize_storage_path_component(experiment_name)
    return f"{base_uri}/{experiment_component}"


def ensure_mlflow_experiment(cfg: dict[str, Any], experiment_name: str) -> str:
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    client = MlflowClient(tracking_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id

    artifact_location = get_mlflow_experiment_artifact_location(cfg, experiment_name)
    try:
        return client.create_experiment(experiment_name, artifact_location=artifact_location)
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


def load_config(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_source = config["data"]["source"]
    if data_source not in {"mock", "minio"}:
        raise ValueError(f"Unsupported data.source {data_source!r}; expected 'mock' or 'minio'.")

    config["mlflow"].setdefault(
        "tuning_experiment_name",
        f"{config['mlflow']['experiment_name']}-optuna",
    )
    config.setdefault("optuna", {})
    config["optuna"]["study_name"] = get_optuna_study_name(config)
    model_registry_cfg = config.setdefault("model_registry", {})
    model_registry_cfg.setdefault(
        "model_name",
        config.get("mlflow", {}).get("registered_model_name") or config["model"]["name"],
    )
    model_registry_cfg.setdefault("bucket", os.getenv("MINIO_BUCKET_MODELS", "model-registry"))
    model_registry_cfg.setdefault("promote_best_checkpoint", True)
    model_registry_cfg.setdefault("log_to_mlflow_model_registry", True)
    return config


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


def mark_mlflow_run_pruned(
    epoch: int,
    global_step: int,
    objective_metric_name: str,
    objective_direction: str,
    current_metric: float | None = None,
) -> None:
    if mlflow.active_run() is None:
        return
    mlflow.set_tags(
        {
            "status": "pruned",
            "objective_metric_name": objective_metric_name,
            "objective_direction": objective_direction,
        }
    )
    if current_metric is not None:
        mlflow.log_metric("objective_value", current_metric, step=global_step)
    mlflow.log_metric("pruned_epoch", epoch, step=global_step)


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


def ensure_supported_tune_runtime() -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            "Tune mode must be started as a single controller process. "
            "The controller launches each trial with Accelerate on the requested GPUs. "
            "Run tune mode directly, for example with TRAIN_MODE=tune in the Docker training container."
        )


def resolve_tune_num_processes() -> int:
    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = os.getenv("NUM_PROCESSES")
    if requested is None or requested.strip() == "":
        return available_gpu_count if available_gpu_count > 0 else 1

    try:
        parsed = int(requested)
    except ValueError as error:
        raise ValueError(f"NUM_PROCESSES must be an integer, got {requested!r}.") from error

    if parsed < 1:
        raise ValueError(f"NUM_PROCESSES must be at least 1, got {parsed}.")
    if available_gpu_count == 0:
        return 1
    return min(parsed, available_gpu_count)


def resolve_accelerate_config_path() -> str:
    configured_path = os.getenv("ACCELERATE_CONFIG_FILE")
    if configured_path:
        return configured_path
    return str(Path(__file__).with_name("accelerate_config.yaml"))


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
    num_processes = resolve_tune_num_processes()
    accelerate_config_path = resolve_accelerate_config_path()
    script_path = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(prefix=f"optuna-trial-{trial.number:04d}-") as temp_dir:
        temp_root = Path(temp_dir)
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


# =============================================================================
# Dataset construction
# =============================================================================

# Build a deterministic split so mock-data training runs are reproducible.
def make_split(pairs: list[tuple[str, str]], target_size: int) -> list[tuple[str, str]]:
    pool = pairs * (target_size // len(pairs) + 1)
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(pool), generator=generator).tolist()
    shuffled = [pool[index] for index in order]
    return shuffled[:target_size]


class RecipeTextDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: Any,
        task_prefix: str,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.inputs = [task_prefix + corrupted for corrupted, _ in pairs]
        self.targets = [target for _, target in pairs]
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        model_inputs = self.tokenizer(
            self.inputs[index],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            self.targets[index],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        # T5 ignores labels set to -100 when computing the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs.input_ids.squeeze(0),
            "attention_mask": model_inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


class MockRecipeDataset(RecipeTextDataset):
    def __init__(self, tokenizer: Any, cfg: dict[str, Any], split: str) -> None:
        if split == "train":
            size = cfg["data"]["mock_train_size"]
        elif split == "val":
            size = cfg["data"]["mock_val_size"]
        else:
            raise ValueError(f"Unsupported split {split!r}")

        super().__init__(
            pairs=make_split(CORRUPTIONS, size),
            tokenizer=tokenizer,
            task_prefix=cfg["model"]["task_prefix"],
            max_input_length=cfg["tokenization"]["max_input_length"],
            max_target_length=cfg["tokenization"]["max_target_length"],
        )


def build_s3_client() -> Any:
    endpoint = os.getenv("MINIO_ENDPOINT")
    endpoint_url = None
    if endpoint:
        endpoint_url = endpoint if endpoint.startswith("http") else f"http://{endpoint}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def load_minio_dataset(cfg: dict[str, Any], tokenizer: Any, split: str) -> RecipeTextDataset:
    key_name = "minio_train_key" if split == "train" else "minio_val_key"
    client = build_s3_client()
    bucket = cfg["data"]["minio_bucket"]
    key = cfg["data"][key_name]
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey as exc:
        raise FileNotFoundError(
            "MinIO dataset object not found for "
            f"{split!r} split: bucket={bucket!r}, key={key!r}. "
            "Either upload that JSONL object to MinIO or switch config.yaml "
            "to data.source: mock for local smoke-test training."
        ) from exc
    payload = response["Body"].read().decode("utf-8")

    pairs: list[tuple[str, str]] = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        # Each line is expected to be a JSON object with {"input": ..., "target": ...}.
        record = json.loads(line)
        pairs.append((record["input"], record["target"]))

    return RecipeTextDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        task_prefix=cfg["model"]["task_prefix"],
        max_input_length=cfg["tokenization"]["max_input_length"],
        max_target_length=cfg["tokenization"]["max_target_length"],
    )


def build_datasets(cfg: dict[str, Any], tokenizer: Any) -> tuple[Dataset, Dataset]:
    if cfg["data"]["source"] == "mock":
        return MockRecipeDataset(tokenizer, cfg, "train"), MockRecipeDataset(tokenizer, cfg, "val")
    return load_minio_dataset(cfg, tokenizer, "train"), load_minio_dataset(cfg, tokenizer, "val")


def build_dataloaders(
    cfg: dict[str, Any], tokenizer: Any, accelerator: Accelerator
) -> tuple[DataLoader, DataLoader, Dataset, Dataset]:
    train_dataset, val_dataset = build_datasets(cfg, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=accelerator.device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=accelerator.device.type == "cuda",
    )
    return train_loader, val_loader, train_dataset, val_dataset


def summarize_training_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": cfg["model"]["name"],
        "model_source": resolve_model_source(cfg),
        "data_source": cfg["data"]["source"],
        "num_epochs": cfg["training"]["num_epochs"],
        "train_batch_size": cfg["training"]["per_device_train_batch_size"],
        "eval_batch_size": cfg["training"]["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "learning_rate": cfg["training"]["learning_rate"],
        "warmup_ratio": cfg["training"]["warmup_ratio"],
        "checkpoint_dir": cfg["checkpointing"]["checkpoint_dir"],
        "tracking_uri": cfg["mlflow"]["tracking_uri"],
    }


def summarize_batch(batch: dict[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for key, value in batch.items():
        shape = tuple(value.shape)
        parts.append(f"{key}=shape{shape},dtype={value.dtype}")
    return "; ".join(parts)


def resolve_hf_cache_dir(cfg: dict[str, Any]) -> Path:
    configured = cfg.get("huggingface", {}).get("cache_dir")
    if configured:
        return Path(configured).expanduser()
    checkpoint_dir = Path(cfg["checkpointing"]["checkpoint_dir"]).expanduser()
    return checkpoint_dir.parent / ".hf-cache"


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
    model_snapshot_path = snapshot_download(
        repo_id=cfg["model"]["name"],
        cache_dir=str(cache_dir),
    )
    cfg.setdefault("model", {})["local_path"] = model_snapshot_path
    return model_snapshot_path


# =============================================================================
# Training and evaluation utilities
# =============================================================================

def get_optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = {"bias", "LayerNorm.weight"}
    named_parameters = list(model.named_parameters())
    return [
        {
            "params": [param for name, param in named_parameters if not any(term in name for term in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [param for name, param in named_parameters if any(term in name for term in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def move_generate_kwargs_to_device(generate_kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in generate_kwargs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_safely(model: torch.nn.Module, accelerator: Accelerator, **generate_kwargs: Any) -> torch.Tensor:
    unwrapped_model = accelerator.unwrap_model(model)
    try:
        return unwrapped_model.generate(**generate_kwargs)
    except RuntimeError as error:
        if accelerator.device.type != "cuda" or not is_cuda_oom(error):
            raise

        # Evaluation can still finish if generation falls back to CPU after a CUDA OOM.
        accelerator.print("CUDA OOM during generation; retrying on CPU.")
        torch.cuda.empty_cache()

        cpu_model = copy.deepcopy(unwrapped_model).to("cpu")
        cpu_model.eval()
        cpu_kwargs = move_generate_kwargs_to_device(generate_kwargs, torch.device("cpu"))
        with torch.no_grad():
            generated = cpu_model.generate(**cpu_kwargs)

        del cpu_model
        torch.cuda.empty_cache()
        return generated.to(accelerator.device)


def decode_batch(tokenizer: Any, token_ids: torch.Tensor) -> list[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def safe_metric_value(value: Any) -> float:
    if hasattr(value, "mid"):
        return float(value.mid.fmeasure)
    return float(value)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    accelerator: Accelerator,
    cfg: dict[str, Any],
) -> dict[str, float]:
    # Validation computes both loss and decoded text metrics such as ROUGE.
    model.eval()
    total_loss = 0.0
    num_batches = 0
    rouge_metric = load_metric("rouge") if accelerator.is_main_process else None
    debug_log(accelerator, f"Starting evaluation over {len(loader)} batches.")

    for batch_index, batch in enumerate(loader, start=1):
        num_batches += 1
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.float()
        total_loss += accelerator.reduce(loss.detach(), reduction="mean").item()

        if batch_index == 1:
            debug_log(
                accelerator,
                f"First eval batch summary: {summarize_batch(batch)}",
            )

        generated = generate_safely(
            model,
            accelerator,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg["tokenization"]["max_target_length"],
        )

        labels_for_decode = labels.clone()
        labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id

        # Gather predictions across all processes before computing metrics on rank 0.
        generated = accelerator.pad_across_processes(generated, dim=1, pad_index=tokenizer.pad_token_id)
        labels_for_decode = accelerator.pad_across_processes(
            labels_for_decode, dim=1, pad_index=tokenizer.pad_token_id
        )
        gathered_generated, gathered_labels = accelerator.gather_for_metrics((generated, labels_for_decode))

        if accelerator.is_main_process and rouge_metric is not None:
            rouge_metric.add_batch(
                predictions=decode_batch(tokenizer, gathered_generated),
                references=decode_batch(tokenizer, gathered_labels),
            )

        if batch_index == len(loader) or batch_index % 10 == 0:
            debug_log(
                accelerator,
                f"Evaluation progress: batch {batch_index}/{len(loader)} "
                f"(running avg loss: {total_loss / max(num_batches, 1):.4f})",
            )

        del outputs, generated, labels_for_decode

    avg_loss = total_loss / max(num_batches, 1)
    if accelerator.is_main_process and rouge_metric is not None:
        rouge_scores = rouge_metric.compute(use_stemmer=True)
    else:
        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    metrics = {
        "eval_loss": avg_loss,
        "eval_rouge1": safe_metric_value(rouge_scores["rouge1"]),
        "eval_rouge2": safe_metric_value(rouge_scores["rouge2"]),
        "eval_rougeL": safe_metric_value(rouge_scores["rougeL"]),
    }
    debug_log(accelerator, f"Finished evaluation with metrics: {json.dumps(metrics, sort_keys=True)}")
    return metrics


# =============================================================================
# Experiment tracking and checkpointing
# =============================================================================

def get_peak_gpu_metrics(accelerator: Accelerator) -> dict[str, float]:
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_gb": 0.0,
            "gpu_memory_reserved_gb": 0.0,
        }

    device_index = accelerator.device.index or 0
    return {
        "gpu_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device_index) / 1e9, 3),
        "gpu_memory_reserved_gb": round(torch.cuda.max_memory_reserved(device_index) / 1e9, 3),
    }


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
        "env.torch_version": torch.__version__,
        "env.transformers_version": transformers.__version__,
        "env.accelerate_version": accelerate_pkg.__version__,
        "env.mlflow_version": mlflow.__version__,
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


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    epoch: int,
    metrics: dict[str, float],
) -> str:
    path = checkpoint_dir / f"epoch-{epoch:02d}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
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
    total_steps: int,
    warmup_steps: int,
    trainable_params: int,
    model: torch.nn.Module,
    context: TrainingContext,
) -> str | None:
    if not accelerator.is_main_process:
        return None

    experiment_name = context.mlflow_experiment_name or get_mlflow_experiment_name(cfg, context.mode)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment_id = ensure_mlflow_experiment(cfg, experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run = mlflow.start_run(
        run_name=context.mlflow_run_name,
        nested=context.mlflow_nested,
        parent_run_id=context.mlflow_parent_run_id,
    )
    mlflow.set_tag("mlflow.runName", format_mlflow_run_name(run.info.run_id))

    flat_params = filter_mlflow_run_params(cfg, context.mode)
    flat_params.update(
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
            "data_source": cfg["data"]["source"],
            "num_train_examples": len(train_dataset),
            "num_val_examples": len(val_dataset),
        }
    )
    mlflow.log_params(sanitize_mlflow_params(flat_params))
    if context.trial_params:
        mlflow.log_params(
            sanitize_mlflow_params(
                {f"trial_param.{key}": value for key, value in context.trial_params.items()}
            )
        )
    mlflow.set_tags(log_environment_info())
    if context.mlflow_tags:
        mlflow.set_tags({key: str(value) for key, value in context.mlflow_tags.items()})
    log_model_summary(accelerator.unwrap_model(model))
    log_yaml_artifact(cfg, "resolved_config.yaml", artifact_path="config")
    return run.info.run_id


def upload_directory_to_minio(local_dir: Path, bucket: str, prefix: str) -> dict[str, str]:
    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"Cannot upload missing directory to MinIO: {local_dir}")

    client = build_s3_client()
    uploaded_keys: list[str] = []
    normalized_prefix = prefix.strip("/")

    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(local_dir).as_posix()
        object_key = f"{normalized_prefix}/{relative_path}" if normalized_prefix else relative_path
        client.upload_file(str(path), bucket, object_key)
        uploaded_keys.append(object_key)

    if not uploaded_keys:
        raise ValueError(f"No files found to upload from {local_dir}")

    return {
        "bucket": bucket,
        "prefix": normalized_prefix,
        "s3_uri": f"s3://{bucket}/{normalized_prefix}" if normalized_prefix else f"s3://{bucket}",
        "file_count": str(len(uploaded_keys)),
    }


def promote_best_checkpoint_to_model_registry(
    cfg: dict[str, Any],
    best_checkpoint: str | None,
    run_id: str | None,
    experiment_name: str,
) -> dict[str, str] | None:
    if not best_checkpoint:
        return None
    if not bool(cfg.get("model_registry", {}).get("promote_best_checkpoint", True)):
        return None

    model_registry_cfg = cfg.get("model_registry", {})
    model_bucket = str(model_registry_cfg.get("bucket") or os.getenv("MINIO_BUCKET_MODELS", "model-registry"))
    model_name = sanitize_storage_path_component(
        str(model_registry_cfg.get("model_name") or cfg.get("mlflow", {}).get("registered_model_name") or cfg["model"]["name"])
    )
    experiment_component = sanitize_storage_path_component(experiment_name)
    run_component = sanitize_storage_path_component(run_id or "manual-run")
    prefix = f"{model_name}/experiments/{experiment_component}/runs/{run_component}/best-checkpoint"
    return upload_directory_to_minio(Path(best_checkpoint), model_bucket, prefix)


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


# =============================================================================
# Main training loop
# =============================================================================

def train_worker(config_dict: dict[str, Any], context: TrainingContext | None = None) -> TrainingResult:
    context = context or TrainingContext()
    mixed_precision = os.getenv("ACCELERATE_MIXED_PRECISION", "no")
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config_dict["training"]["gradient_accumulation_steps"],
    )
    set_seed(config_dict["training"]["seed"])
    debug_log(
        accelerator,
        "Initializing training worker and loading configuration.",
        section="TRAINING START",
    )
    debug_log(accelerator, f"Execution mode: {context.mode}")
    debug_log(accelerator, f"Using mixed precision mode: {mixed_precision}")
    debug_log(accelerator, f"Training config summary: {json.dumps(summarize_training_config(config_dict), sort_keys=True)}")

    model_source = prepare_model_cache(config_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, local_files_only=True)
    debug_log(
        accelerator,
        f"Loaded tokenizer and model for {model_source}.",
        section="MODEL SETUP",
    )
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(config_dict, tokenizer, accelerator)
    debug_log(
        accelerator,
        f"Dataset summary: train_examples={len(train_dataset)}, val_examples={len(val_dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}",
        section="DATA READY",
    )

    optimizer = AdamW(
        get_optimizer_param_groups(model, config_dict["training"]["weight_decay"]),
        lr=config_dict["training"]["learning_rate"],
    )
    steps_per_epoch = math.ceil(len(train_loader) / config_dict["training"]["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * config_dict["training"]["num_epochs"]
    warmup_steps = int(total_steps * config_dict["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    debug_log(
        accelerator,
        f"Optimizer/scheduler summary: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, processes={accelerator.num_processes}",
        section="OPTIMIZER SETUP",
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    debug_log(accelerator, f"Accelerator prepared objects on device {accelerator.device}.", main_process_only=False)

    checkpoint_dir = Path(config_dict["checkpointing"]["checkpoint_dir"])
    objective_metric_name = context.objective_metric_name or config_dict["evaluation"]["metric_for_best_model"]
    objective_direction = context.objective_direction or infer_metric_direction(objective_metric_name)
    best_metric = initial_best_metric(objective_direction)
    best_checkpoint: str | None = None
    final_metrics: dict[str, float] = {}
    global_step = 0
    train_loss_accumulator = 0.0
    train_loss_window_count = 0
    run_id = maybe_start_mlflow_run(
        config_dict,
        train_dataset,
        val_dataset,
        accelerator,
        total_steps,
        warmup_steps,
        sum(parameter.numel() for parameter in accelerator.unwrap_model(model).parameters() if parameter.requires_grad),
        model,
        context,
    )
    run_checkpoint_dir = resolve_run_checkpoint_dir(checkpoint_dir, run_id)
    training_start = time.time()
    debug_log(
        accelerator,
        f"MLflow run id: {run_id or 'not-started-on-this-rank'}; "
        f"local checkpoints will be stored under {run_checkpoint_dir}",
        section="RUN CONTEXT",
    )

    try:
        optimizer.zero_grad()
        for epoch in range(1, config_dict["training"]["num_epochs"] + 1):
            pre_epoch_prune_requested = False
            if accelerator.is_main_process and prune_requested_via_file(context):
                pre_epoch_prune_requested = True
                mark_mlflow_run_pruned(
                    epoch=epoch,
                    global_step=global_step,
                    objective_metric_name=objective_metric_name,
                    objective_direction=objective_direction,
                )
            should_prune_before_epoch = accelerator.reduce(
                torch.tensor(int(pre_epoch_prune_requested), device=accelerator.device),
                reduction="sum",
            ).item() > 0
            if should_prune_before_epoch:
                raise optuna.TrialPruned(f"Trial pruned before epoch {epoch} started.")

            model.train()
            epoch_start = time.time()
            debug_log(
                accelerator,
                f"Starting epoch {epoch}/{config_dict['training']['num_epochs']}.",
                section=f"EPOCH {epoch}/{config_dict['training']['num_epochs']}",
            )
            if accelerator.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(accelerator.device.index or 0)

            for batch_index, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)

                if batch_index == 1:
                    debug_log(
                        accelerator,
                        f"First training batch summary: {summarize_batch(batch)}",
                    )

                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss.float()
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite loss detected at epoch {epoch}, batch {batch_index}: {loss.item()}"
                        )
                    train_loss_accumulator += accelerator.reduce(loss.detach(), reduction="mean").item()
                    train_loss_window_count += 1
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config_dict["training"]["max_grad_norm"])

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    # `sync_gradients` is only true on the last micro-batch of an accumulation window.
                    global_step += 1
                    if global_step == 1 or global_step % 25 == 0:
                        debug_log(
                            accelerator,
                            f"Training progress: epoch={epoch}, batch={batch_index}/{len(train_loader)}, "
                            f"global_step={global_step}, loss={loss.item():.4f}, "
                            f"lr={scheduler.get_last_lr()[0]:.8f}",
                        )
                    if accelerator.is_main_process and global_step % config_dict["evaluation"]["logging_steps"] == 0:
                        mlflow.log_metrics(
                            {
                                "train_loss": train_loss_accumulator / max(train_loss_window_count, 1),
                                "learning_rate": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
                        debug_log(
                            accelerator,
                            f"Logged MLflow training metrics at global_step={global_step}.",
                        )
                        train_loss_accumulator = 0.0
                        train_loss_window_count = 0

            epoch_time = time.time() - epoch_start
            debug_log(
                accelerator,
                f"Epoch {epoch} training phase finished in {epoch_time:.2f}s. Starting evaluation.",
                section=f"EPOCH {epoch} EVALUATION",
            )
            eval_metrics = evaluate(model, val_loader, tokenizer, accelerator, config_dict)
            accelerator.wait_for_everyone()

            epoch_metrics = {
                **eval_metrics,
                "epoch_time_sec": round(epoch_time, 2),
                "samples_per_sec": round(len(train_dataset) / max(epoch_time, 1e-6), 2),
                **get_peak_gpu_metrics(accelerator),
            }
            current_metric = epoch_metrics[objective_metric_name]
            debug_log(
                accelerator,
                f"Evaluation summary for epoch {epoch}: {json.dumps(epoch_metrics, sort_keys=True)}",
            )
            if accelerator.is_main_process:
                append_jsonl_file(
                    context.progress_file,
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "current_metric": current_metric,
                        "objective_metric_name": objective_metric_name,
                        "metrics": epoch_metrics,
                    },
                )

            prune_requested = False
            if accelerator.is_main_process and context.trial is not None:
                context.trial.report(current_metric, step=epoch)
                mlflow.log_metric("objective_value", current_metric, step=global_step)
                if context.trial.should_prune():
                    prune_requested = True
                    mark_mlflow_run_pruned(
                        epoch=epoch,
                        global_step=global_step,
                        objective_metric_name=objective_metric_name,
                        objective_direction=objective_direction,
                        current_metric=current_metric,
                    )
            if accelerator.is_main_process and prune_requested_via_file(context):
                prune_requested = True
                mark_mlflow_run_pruned(
                    epoch=epoch,
                    global_step=global_step,
                    objective_metric_name=objective_metric_name,
                    objective_direction=objective_direction,
                    current_metric=current_metric,
                )

            prune_signal = torch.tensor(int(prune_requested), device=accelerator.device)
            should_prune = accelerator.reduce(prune_signal, reduction="sum").item() > 0
            if should_prune:
                debug_log(
                    accelerator,
                    f"Pruning triggered at epoch {epoch} using {objective_metric_name}={current_metric:.4f}.",
                    section="TRIAL PRUNED",
                    main_process_only=False,
                )
                raise optuna.TrialPruned(f"Trial pruned at epoch {epoch} with {objective_metric_name}={current_metric:.4f}")

            checkpoint_path = save_checkpoint(
                accelerator,
                model,
                tokenizer,
                run_checkpoint_dir,
                epoch,
                epoch_metrics,
            )
            debug_log(
                accelerator,
                f"Epoch {epoch} complete. Checkpoint saved to {checkpoint_path}. "
                f"Metrics: {json.dumps(epoch_metrics, sort_keys=True)}",
                section=f"EPOCH {epoch} COMPLETE",
            )

            if accelerator.is_main_process:
                mlflow.log_metrics(epoch_metrics, step=global_step)
                mlflow.log_metric("objective_value", current_metric, step=global_step)

                if is_better_metric(current_metric, best_metric, objective_direction):
                    best_metric = current_metric
                    best_checkpoint = checkpoint_path
                    mlflow.log_metric(
                        f"best_{objective_metric_name}",
                        best_metric,
                        step=global_step,
                    )
                    mlflow.set_tag("best_checkpoint", best_checkpoint)
                    debug_log(
                        accelerator,
                        f"New best checkpoint at epoch {epoch}: {best_checkpoint} "
                        f"({objective_metric_name}={current_metric:.4f})",
                        section="BEST MODEL UPDATED",
                    )

            final_metrics = epoch_metrics

        if accelerator.is_main_process:
            total_training_time = round(time.time() - training_start, 2)
            mlflow.log_metric("total_training_time_sec", total_training_time, step=global_step)
            mlflow.log_metric(
                f"best_{objective_metric_name}",
                best_metric_to_log(best_metric, objective_direction),
                step=global_step,
            )
            experiment_name = context.mlflow_experiment_name or get_mlflow_experiment_name(config_dict, context.mode)
            promoted_model_info = promote_best_checkpoint_to_model_registry(
                config_dict,
                best_checkpoint,
                run_id,
                experiment_name,
            )
            mlflow_registry_info = maybe_log_best_model_to_mlflow_registry(
                config_dict,
                best_checkpoint,
            )
            run_tags = {
                "mode": context.mode,
                "model_name": config_dict["model"]["name"],
                "task": "recipe-correction",
                "data_source": config_dict["data"]["source"],
                "status": "complete",
                "best_checkpoint": best_checkpoint or "none",
                "objective_metric_name": objective_metric_name,
                "objective_direction": objective_direction,
                "best_model_storage": "minio-model-registry+mlflow-model-registry",
            }
            if promoted_model_info is not None:
                run_tags.update(
                    {
                        "best_checkpoint_s3_uri": promoted_model_info["s3_uri"],
                        "best_model_s3_uri": promoted_model_info["s3_uri"],
                        "best_model_bucket": promoted_model_info["bucket"],
                        "best_model_prefix": promoted_model_info["prefix"],
                    }
                )
                mlflow.log_params(
                    {
                        "best_checkpoint_s3_uri": promoted_model_info["s3_uri"],
                        "best_model_bucket": promoted_model_info["bucket"],
                        "best_model_prefix": promoted_model_info["prefix"],
                    }
                )
                log_json_artifact(
                    {
                        "run_id": run_id,
                        "experiment_name": experiment_name,
                        "best_checkpoint_local_path": best_checkpoint,
                        "best_checkpoint_s3_uri": promoted_model_info["s3_uri"],
                        "best_model_s3_uri": promoted_model_info["s3_uri"],
                        "best_model_bucket": promoted_model_info["bucket"],
                        "best_model_prefix": promoted_model_info["prefix"],
                        "uploaded_file_count": int(promoted_model_info["file_count"]),
                    },
                    "best_model_pointer.json",
                    artifact_path="model_registry",
                )
            if mlflow_registry_info is not None:
                run_tags.update(
                    {
                        "mlflow_registered_model_name": mlflow_registry_info["registered_model_name"],
                        "mlflow_best_model_uri": mlflow_registry_info["model_uri"],
                    }
                )
                mlflow.log_params(
                    {
                        "mlflow_registered_model_name": mlflow_registry_info["registered_model_name"],
                        "mlflow_best_model_uri": mlflow_registry_info["model_uri"],
                    }
                )
            mlflow.set_tags(run_tags)
            emit_console_summary(
                accelerator.print,
                "RUN SUMMARY",
                {
                    "mode": context.mode,
                    "run_id": run_id,
                    "experiment_name": experiment_name,
                    "model_name": config_dict["model"]["name"],
                    "data_source": config_dict["data"]["source"],
                    "total_epochs": config_dict["training"]["num_epochs"],
                    "global_steps": global_step,
                    "best_metric_name": objective_metric_name,
                    "best_metric": best_metric_to_log(best_metric, objective_direction),
                    "best_checkpoint": best_checkpoint,
                    "best_checkpoint_s3_uri": promoted_model_info["s3_uri"] if promoted_model_info is not None else "none",
                    "total_training_time_sec": total_training_time,
                    "best_model_storage": "minio-model-registry+mlflow-model-registry",
                    "mlflow_registered_model_name": (
                        mlflow_registry_info["registered_model_name"] if mlflow_registry_info is not None else "none"
                    ),
                },
            )
            if final_metrics:
                emit_console_summary(
                    accelerator.print,
                    "FINAL METRICS",
                    final_metrics,
                )

        training_result = TrainingResult(
            best_metric=best_metric_to_log(best_metric, objective_direction),
            best_metric_name=objective_metric_name,
            best_checkpoint=best_checkpoint,
            run_id=run_id,
            final_metrics=final_metrics,
            resolved_config=copy.deepcopy(config_dict),
        )
        if accelerator.is_main_process:
            write_training_result_payload(
                context,
                {"status": "complete", "result": serialize_training_result(training_result)},
            )
        return training_result
    except optuna.TrialPruned:
        debug_log(
            accelerator,
            "Training trial was pruned.",
            main_process_only=False,
            section="RUN STOPPED",
        )
        if accelerator.is_main_process:
            write_training_result_payload(
                context,
                {"status": "pruned", "message": "Training trial was pruned."},
            )
        raise
    except Exception as error:
        debug_log(
            accelerator,
            f"Training failed with {error.__class__.__name__}: {error}",
            main_process_only=False,
            section="RUN FAILED",
        )
        debug_log(
            accelerator,
            traceback.format_exc(),
            main_process_only=False,
        )
        if accelerator.is_main_process and mlflow.active_run() is not None:
            mlflow.set_tag("status", "failed")
        if accelerator.is_main_process:
            write_training_result_payload(
                context,
                {
                    "status": "failed",
                    "error_type": error.__class__.__name__,
                    "error_message": str(error),
                },
            )
        raise
    finally:
        if accelerator.is_main_process and mlflow.active_run() is not None:
            mlflow.end_run()
        accelerator.end_training()


def run_optuna_study(cfg: dict[str, Any]) -> optuna.study.Study:
    validate_optuna_config(cfg)
    prepare_model_cache(cfg)
    optuna_cfg = cfg["optuna"]
    study_output_dir = resolve_study_output_dir(cfg)
    study_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = get_mlflow_experiment_name(cfg, "tune")
    study_name = get_optuna_study_name(cfg)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment_id = ensure_mlflow_experiment(cfg, experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    study = optuna.create_study(
        study_name=study_name,
        direction=optuna_cfg["direction"],
        storage=optuna_cfg.get("storage"),
        load_if_exists=bool(optuna_cfg.get("storage")),
        sampler=build_optuna_sampler(optuna_cfg.get("sampler", {})),
        pruner=build_optuna_pruner(optuna_cfg.get("pruner", {})),
    )
    study.set_user_attr("status", "running")
    study.set_user_attr("mlflow_experiment_name", experiment_name)
    study.set_user_attr("study_output_dir", str(study_output_dir))

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = sample_trial_params(trial, optuna_cfg["search_space"])
        resolved_cfg = apply_trial_params(cfg, trial_params)
        trial.set_user_attr("trial_params", trial_params)
        context = TrainingContext(
            mode="tune",
            mlflow_experiment_name=experiment_name,
            mlflow_tags={
                "mode": "tune",
                "study_name": study_name,
                "trial_number": trial.number,
            },
            register_model=False,
            trial=trial,
            objective_direction=optuna_cfg["direction"],
            objective_metric_name=cfg["evaluation"]["metric_for_best_model"],
            trial_params=trial_params,
        )
        try:
            result = run_distributed_trial(resolved_cfg, context=context, trial=trial)
            trial.set_user_attr("run_id", result.run_id)
            trial.set_user_attr("best_checkpoint", result.best_checkpoint)
            trial.set_user_attr("best_metric_name", result.best_metric_name)
            trial.set_user_attr("final_metrics", result.final_metrics)
            trial.set_user_attr("resolved_config", result.resolved_config)
            return result.best_metric
        except optuna.TrialPruned as error:
            message = str(error) or f"Trial {trial.number} was pruned."
            trial.set_user_attr("status_message", message)
            raise
        except Exception as error:
            failure_type = error.__class__.__name__
            failure_message = str(error) or repr(error)
            trial.set_user_attr("failure_type", failure_type)
            trial.set_user_attr("failure_message", failure_message)
            trial.set_user_attr("failure_traceback", traceback.format_exc())
            trial.set_user_attr("resolved_config", resolved_cfg)
            emit_console_summary(
                print,
                f"TUNE TRIAL {trial.number:04d} FAILURE RECORDED",
                {
                    "error_type": failure_type,
                    "error_message": failure_message,
                    "study_name": study_name,
                    "trial_number": trial.number,
                },
            )
            raise

    try:
        study.optimize(
            objective,
            n_trials=int(optuna_cfg["n_trials"]),
            timeout=optuna_cfg.get("timeout_sec"),
            catch=(Exception,),
        )

        summary = build_trial_summary(study)
        summary_path = study_output_dir / "trial_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        trial_counts = summarize_trial_counts(study)
        study.set_user_attr("trial_summary_path", str(summary_path))
        study.set_user_attr("trial_counts", trial_counts)
        emit_console_summary(
            print,
            "OPTUNA STUDY SUMMARY",
            {
                "study_name": study.study_name,
                "total_trials": trial_counts["total"],
                "completed_trials": trial_counts["complete"],
                "failed_trials": trial_counts["failed"],
                "pruned_trials": trial_counts["pruned"],
                "trial_summary_path": str(summary_path),
            },
        )

        completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            failed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]
            last_failure = failed_trials[-1] if failed_trials else None
            if last_failure is not None:
                failure_type = last_failure.user_attrs.get("failure_type", "unknown")
                failure_message = last_failure.user_attrs.get("failure_message", "unknown error")
                raise RuntimeError(
                    "Optuna study finished without any completed trials. "
                    f"failed={trial_counts['failed']}, pruned={trial_counts['pruned']}. "
                    f"Last failed trial {last_failure.number}: {failure_type}: {failure_message}. "
                    f"See {summary_path} for the full trial summary."
                )
            raise RuntimeError(
                "Optuna study finished without any completed trials. "
                f"failed={trial_counts['failed']}, pruned={trial_counts['pruned']}. "
                f"See {summary_path} for the full trial summary."
            )

        best_trial = study.best_trial
        best_config = best_trial.user_attrs["resolved_config"]
        best_config_path = study_output_dir / "best_config.yaml"
        write_yaml_file(best_config_path, best_config)

        emit_console_summary(
            print,
            "OPTUNA BEST MODEL",
            {
                "study_name": study.study_name,
                "best_trial_number": best_trial.number,
                "best_value": study.best_value,
                "best_metric_name": best_trial.user_attrs.get("best_metric_name", "unknown"),
                "best_run_id": best_trial.user_attrs.get("run_id", "unknown"),
                "best_checkpoint": best_trial.user_attrs.get("best_checkpoint", "none"),
                "best_config_path": str(best_config_path),
                "trial_summary_path": str(summary_path),
                "best_params": best_trial.params,
                "final_metrics": best_trial.user_attrs.get("final_metrics", {}),
            },
        )

        study.set_user_attr("best_value", float(study.best_value))
        study.set_user_attr("best_trial_number", int(best_trial.number))
        study.set_user_attr("best_run_id", str(best_trial.user_attrs.get("run_id", "unknown")))
        study.set_user_attr("best_config_path", str(best_config_path))
        study.set_user_attr("status", "complete")
        return study
    except Exception as error:
        study.set_user_attr("status", "failed")
        study.set_user_attr("error_type", error.__class__.__name__)
        study.set_user_attr("error_message", str(error) or repr(error))
        study.set_user_attr("trial_counts", summarize_trial_counts(study))
        emit_console_summary(
            print,
            "OPTUNA STUDY FAILED",
            {
                "study_name": study.study_name,
                "error_type": error.__class__.__name__,
                "error_message": str(error) or repr(error),
                **summarize_trial_counts(study),
            },
        )
        raise


# =============================================================================
# CLI entrypoint
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a T5 recipe-correction model with Accelerate.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the training config YAML file.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "tune"),
        default="train",
        help="Whether to run one training job or an Optuna tuning study.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the MLflow experiment name for this invocation.",
    )
    parser.add_argument(
        "--context-file",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.experiment_name:
        cfg.setdefault("mlflow", {})
        cfg["mlflow"]["active_experiment_name"] = args.experiment_name
    context = load_training_context(args.context_file)

    if args.mode == "tune":
        ensure_supported_tune_runtime()
        run_optuna_study(cfg)
    else:
        try:
            train_worker(cfg, context=context)
        except optuna.TrialPruned:
            # Distributed tune workers report pruning through the result file for the
            # controller process, so they should exit cleanly instead of surfacing as
            # failed torchrun ranks.
            if context is not None and context.result_file is not None:
                return
            raise


if __name__ == "__main__":
    main()
