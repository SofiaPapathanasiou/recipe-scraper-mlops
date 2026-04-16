import json
from typing import Any

from accelerate import Accelerator

LOG_DELIMITER = "=" * 88
LOG_SUBDELIMITER = "-" * 88

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
