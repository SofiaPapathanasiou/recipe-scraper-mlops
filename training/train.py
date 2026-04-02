import argparse
import copy
import json
import math
import os
import platform
import socket
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import boto3
import mlflow
import mlflow.pytorch
import psutil
import torch
import yaml
from accelerate import Accelerator
import accelerate as accelerate_pkg
from accelerate.utils import set_seed
from evaluate import load as load_metric
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

CORRUPTIONS = [
    (
        "Ingredientss: 2 cup flur, 1 eg, salt tt taste.",
        "Ingredients: 2 cups flour, 1 egg, salt to taste.",
    ),
    (
        "Directons: Pre-heat ovn to 180C. Bake fr 30 min.",
        "Directions: Preheat oven to 180°C. Bake for 30 minutes.",
    ),
    (
        "Title Choclate Cake\nIngredents sugar buttr eggs flowur",
        "Title: Chocolate Cake\nIngredients: sugar, butter, eggs, flour",
    ),
    (
        "Sevings: 4\nPrep tme 15 mins\nCok time: 30minut",
        "Servings: 4\nPrep time: 15 mins\nCook time: 30 minutes",
    ),
    (
        "1/2 cupp milk or creme\n2 tsp vanil extrat",
        "1/2 cup milk or cream\n2 tsp vanilla extract",
    ),
]


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def debug_log(accelerator: Accelerator, message: str, *, main_process_only: bool = True) -> None:
    if main_process_only and not accelerator.is_main_process:
        return
    prefix = f"[rank {accelerator.process_index}]"
    accelerator.print(f"{prefix} {message}")


def load_config(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_source = config["data"]["source"]
    if data_source not in {"mock", "minio"}:
        raise ValueError(f"Unsupported data.source {data_source!r}; expected 'mock' or 'minio'.")

    return config


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
    response = client.get_object(Bucket=cfg["data"]["minio_bucket"], Key=cfg["data"][key_name])
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


def log_environment_info() -> dict[str, str]:
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
    mlflow.log_artifact(summary_path, artifact_path="model_summary")
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
) -> str | None:
    if not accelerator.is_main_process:
        return None

    # Only the main process writes run metadata to avoid duplicate MLflow entries.
    mlflow.end_run()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    run = mlflow.start_run(
        run_name=(
            f"{cfg['model']['name']}__lr{cfg['training']['learning_rate']}"
            f"__ep{cfg['training']['num_epochs']}"
        )
    )

    flat_params = flatten_dict(cfg)
    flat_params.update(
        {
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
    mlflow.log_params(flat_params)
    mlflow.set_tags(log_environment_info())
    log_model_summary(accelerator.unwrap_model(model))
    return run.info.run_id


def maybe_register_best_model(cfg: dict[str, Any], best_checkpoint: str | None) -> None:
    if not best_checkpoint or not cfg["mlflow"]["register_model"]:
        return

    best_model = AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint)
    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="best-model",
        registered_model_name=cfg["mlflow"]["registered_model_name"],
    )


# =============================================================================
# Main training loop
# =============================================================================

def train_worker(config_dict: dict[str, Any]) -> None:
    mixed_precision = os.getenv("ACCELERATE_MIXED_PRECISION", "no")
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config_dict["training"]["gradient_accumulation_steps"],
    )
    set_seed(config_dict["training"]["seed"])
    debug_log(accelerator, f"Using mixed precision mode: {mixed_precision}")
    debug_log(accelerator, f"Training config summary: {json.dumps(summarize_training_config(config_dict), sort_keys=True)}")

    tokenizer = AutoTokenizer.from_pretrained(config_dict["model"]["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config_dict["model"]["name"])
    debug_log(accelerator, f"Loaded tokenizer and model for {config_dict['model']['name']}.")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(config_dict, tokenizer, accelerator)
    debug_log(
        accelerator,
        f"Dataset summary: train_examples={len(train_dataset)}, val_examples={len(val_dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}",
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
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    debug_log(accelerator, f"Accelerator prepared objects on device {accelerator.device}.", main_process_only=False)

    checkpoint_dir = Path(config_dict["checkpointing"]["checkpoint_dir"])
    best_metric = -float("inf")
    best_checkpoint: str | None = None
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
    )
    run_checkpoint_dir = resolve_run_checkpoint_dir(checkpoint_dir, run_id)
    training_start = time.time()
    debug_log(
        accelerator,
        f"MLflow run id: {run_id or 'not-started-on-this-rank'}; "
        f"local checkpoints will be stored under {run_checkpoint_dir}",
    )

    try:
        optimizer.zero_grad()
        for epoch in range(1, config_dict["training"]["num_epochs"] + 1):
            model.train()
            epoch_start = time.time()
            debug_log(accelerator, f"Starting epoch {epoch}/{config_dict['training']['num_epochs']}.")
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
                        train_loss_accumulator = 0.0
                        train_loss_window_count = 0

            epoch_time = time.time() - epoch_start
            eval_metrics = evaluate(model, val_loader, tokenizer, accelerator, config_dict)
            accelerator.wait_for_everyone()

            epoch_metrics = {
                **eval_metrics,
                "epoch_time_sec": round(epoch_time, 2),
                "samples_per_sec": round(len(train_dataset) / max(epoch_time, 1e-6), 2),
                **get_peak_gpu_metrics(accelerator),
            }
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
            )

            if accelerator.is_main_process:
                mlflow.log_metrics(epoch_metrics, step=global_step)
                mlflow.log_artifacts(checkpoint_path, artifact_path=f"checkpoints/epoch-{epoch:02d}")

                # Track the best checkpoint using the configured validation metric.
                metric_name = config_dict["evaluation"]["metric_for_best_model"]
                current_metric = epoch_metrics[metric_name]
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_checkpoint = checkpoint_path
                    mlflow.log_metric("best_rougeL", best_metric, step=global_step)
                    mlflow.set_tag("best_checkpoint", best_checkpoint)
                    debug_log(
                        accelerator,
                        f"New best checkpoint at epoch {epoch}: {best_checkpoint} "
                        f"({metric_name}={current_metric:.4f})",
                    )

        if accelerator.is_main_process:
            total_training_time = round(time.time() - training_start, 2)
            mlflow.log_metric("total_training_time_sec", total_training_time, step=global_step)
            mlflow.log_metric("best_rougeL", best_metric if best_metric > -float("inf") else 0.0, step=global_step)
            maybe_register_best_model(config_dict, best_checkpoint)
            mlflow.set_tags(
                {
                    "model_name": config_dict["model"]["name"],
                    "task": "recipe-correction",
                    "data_source": config_dict["data"]["source"],
                    "status": "complete",
                    "best_checkpoint": best_checkpoint or "none",
                }
            )
            accelerator.print(f"Training complete. Run ID: {run_id}")
            accelerator.print(f"Best checkpoint: {best_checkpoint}")
    except Exception as error:
        debug_log(
            accelerator,
            f"Training failed with {error.__class__.__name__}: {error}",
            main_process_only=False,
        )
        debug_log(
            accelerator,
            traceback.format_exc(),
            main_process_only=False,
        )
        if accelerator.is_main_process and mlflow.active_run() is not None:
            mlflow.set_tag("status", "failed")
        raise
    finally:
        if accelerator.is_main_process and mlflow.active_run() is not None:
            mlflow.end_run()
        accelerator.end_training()


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
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_worker(cfg)


if __name__ == "__main__":
    main()
