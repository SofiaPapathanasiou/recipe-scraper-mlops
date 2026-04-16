import copy
import json
from typing import Any

import torch
from accelerate import Accelerator
from evaluate import load as load_metric
from torch.utils.data import DataLoader

from .utils_logging import debug_log
from .utils_recipes import summarize_batch

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
