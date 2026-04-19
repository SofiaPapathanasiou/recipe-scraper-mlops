import argparse
import copy
import json
import math
import os
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mlflow
import optuna
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from utils import (
    TrainingContext,
    TrainingResult,
    append_jsonl_file,
    apply_trial_params,
    best_metric_to_log,
    build_accelerate_launch_command,
    build_accelerate_launch_env,
    build_dataloaders,
    build_mlflow_run_tags,
    build_optuna_pruner,
    build_optuna_sampler,
    build_trial_summary,
    debug_log,
    emit_console_summary,
    ensure_mlflow_experiment,
    ensure_supported_tune_runtime,
    evaluate,
    get_mlflow_experiment_name,
    get_optuna_study_name,
    get_optimizer_param_groups,
    get_peak_gpu_metrics,
    infer_metric_direction,
    initial_best_metric,
    is_better_metric,
    load_config,
    load_training_context,
    log_best_checkpoint_artifacts,
    log_json_artifact,
    mark_mlflow_run_pruned,
    maybe_log_best_model_to_mlflow_registry,
    maybe_start_mlflow_run,
    prepare_model_cache,
    prune_requested_via_file,
    resolve_accelerate_config_path,
    resolve_best_checkpoint_dir,
    resolve_default_config_path,
    resolve_mlflow_tracking_uri,
    resolve_num_processes,
    resolve_run_checkpoint_dir,
    resolve_study_output_dir,
    run_distributed_trial,
    sample_trial_params,
    save_checkpoint,
    save_checkpoint_to_path,
    serialize_training_result,
    summarize_training_config,
    summarize_batch,
    summarize_trial_counts,
    validate_optuna_config,
    write_optuna_search_space_file,
    write_training_result_payload,
    write_yaml_file,
)


def train_worker(config_dict: dict[str, Any], context: TrainingContext | None = None) -> TrainingResult:
    context = context or TrainingContext()
    mixed_precision = os.getenv("ACCELERATE_MIXED_PRECISION", "no")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
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
    save_intermediate_checkpoints = bool(
        config_dict["checkpointing"].get("save_intermediate_checkpoints", True)
    )
    objective_metric_name = context.objective_metric_name or config_dict["evaluation"]["metric_for_best_model"]
    objective_direction = context.objective_direction or infer_metric_direction(objective_metric_name)
    evaluation_every_n_epochs = max(1, int(config_dict["evaluation"].get("every_n_epochs", 1)))
    evaluate_on_last_epoch = bool(config_dict["evaluation"].get("run_on_last_epoch", True))
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
        mixed_precision,
        total_steps,
        warmup_steps,
        sum(parameter.numel() for parameter in accelerator.unwrap_model(model).parameters() if parameter.requires_grad),
        model,
        context,
    )
    run_checkpoint_dir = resolve_run_checkpoint_dir(checkpoint_dir, run_id)
    persistent_best_checkpoint_dir = resolve_best_checkpoint_dir(checkpoint_dir, run_id)
    training_start = time.time()
    debug_log(
        accelerator,
        (
            f"MLflow run id: {run_id or 'not-started-on-this-rank'}; "
            f"local checkpoints will be stored under {run_checkpoint_dir}"
        )
        if save_intermediate_checkpoints
        else (
            f"MLflow run id: {run_id or 'not-started-on-this-rank'}; "
            f"intermediate local checkpoint saving is disabled and only the best checkpoint "
            f"will be persisted under {persistent_best_checkpoint_dir}"
        ),
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
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)
                labels = batch["labels"].to(accelerator.device, non_blocking=True)

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
            epoch_metrics = {
                "epoch_time_sec": round(epoch_time, 2),
                "samples_per_sec": round(len(train_dataset) / max(epoch_time, 1e-6), 2),
                **get_peak_gpu_metrics(accelerator),
            }
            should_run_evaluation = (
                epoch % evaluation_every_n_epochs == 0
                or (evaluate_on_last_epoch and epoch == config_dict["training"]["num_epochs"])
            )
            if not should_run_evaluation:
                epoch_metrics["evaluation_skipped"] = 1.0
                debug_log(
                    accelerator,
                    (
                        f"Epoch {epoch} training phase finished in {epoch_time:.2f}s. "
                        f"Skipping evaluation because evaluation.every_n_epochs={evaluation_every_n_epochs}."
                    ),
                    section=f"EPOCH {epoch} EVALUATION SKIPPED",
                )
                if accelerator.is_main_process:
                    mlflow.log_metrics(epoch_metrics, step=global_step)
                final_metrics = epoch_metrics
                continue

            debug_log(
                accelerator,
                f"Epoch {epoch} training phase finished in {epoch_time:.2f}s. Starting evaluation.",
                section=f"EPOCH {epoch} EVALUATION",
            )
            eval_metrics = evaluate(model, val_loader, tokenizer, accelerator, config_dict)
            accelerator.wait_for_everyone()

            epoch_metrics.update(eval_metrics)
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

            checkpoint_path: str | None = None
            if save_intermediate_checkpoints:
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
            else:
                debug_log(
                    accelerator,
                    f"Epoch {epoch} complete. Intermediate checkpoint saving is disabled. "
                    f"Metrics: {json.dumps(epoch_metrics, sort_keys=True)}",
                    section=f"EPOCH {epoch} COMPLETE",
                )

            is_new_best = is_better_metric(current_metric, best_metric, objective_direction)
            if is_new_best:
                best_metric = current_metric
                if save_intermediate_checkpoints:
                    best_checkpoint = checkpoint_path
                else:
                    best_checkpoint = save_checkpoint_to_path(
                        accelerator,
                        model,
                        tokenizer,
                        persistent_best_checkpoint_dir,
                        epoch_metrics,
                    )

            if accelerator.is_main_process:
                mlflow.log_metrics(epoch_metrics, step=global_step)
                mlflow.log_metric("objective_value", current_metric, step=global_step)

                if is_new_best:
                    mlflow.log_metric(
                        f"best_{objective_metric_name}",
                        best_metric,
                        step=global_step,
                    )
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
            logged_checkpoint_info = log_best_checkpoint_artifacts(best_checkpoint)
            mlflow_registry_info = maybe_log_best_model_to_mlflow_registry(
                config_dict,
                best_checkpoint,
            )
            if logged_checkpoint_info is not None:
                log_json_artifact(
                    {
                        "run_id": run_id,
                        "experiment_name": experiment_name,
                        "best_checkpoint_local_path": best_checkpoint,
                        "best_checkpoint_artifact_path": logged_checkpoint_info["artifact_path"],
                        "mlflow_registered_model_name": (
                            mlflow_registry_info["registered_model_name"] if mlflow_registry_info is not None else None
                        ),
                        "mlflow_best_model_uri": (
                            mlflow_registry_info["model_uri"] if mlflow_registry_info is not None else None
                        ),
                    },
                    "best_model_pointer.json",
                    artifact_path="model_registry",
                )
            mlflow.set_tags(build_mlflow_run_tags(config_dict, context, status="complete"))
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
                    "best_checkpoint_local_path": best_checkpoint,
                    "best_checkpoint_artifact_path": (
                        logged_checkpoint_info["artifact_path"] if logged_checkpoint_info is not None else "none"
                    ),
                    "total_training_time_sec": total_training_time,
                    "best_model_storage": "mlflow-artifacts+mlflow-model-registry",
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
        if best_checkpoint is not None:
            debug_log(
                accelerator,
                f"Best checkpoint remains on disk at {best_checkpoint}.",
                main_process_only=False,
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
    search_space_path = write_optuna_search_space_file(cfg, experiment_name)
    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(cfg))
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
    if search_space_path is not None:
        study.set_user_attr("optuna_search_space_path", str(search_space_path))

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
                "optuna_search_space_path": str(search_space_path) if search_space_path is not None else "none",
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
                "optuna_search_space_path": str(search_space_path) if search_space_path is not None else "none",
                "trial_summary_path": str(summary_path),
                "best_params": best_trial.params,
                "final_metrics": best_trial.user_attrs.get("final_metrics", {}),
            },
        )

        study.set_user_attr("best_value", float(study.best_value))
        study.set_user_attr("best_trial_number", int(best_trial.number))
        study.set_user_attr("best_run_id", str(best_trial.user_attrs.get("run_id", "unknown")))
        study.set_user_attr("best_config_path", str(best_config_path))
        if search_space_path is not None:
            study.set_user_attr("optuna_search_space_path", str(search_space_path))
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

def _resolve_cli_argv() -> list[str]:
    if os.getenv("TRAIN_EXTRA_ARGS_FORWARDED") == "1":
        return sys.argv[1:]
    env_extra_args = os.getenv("TRAIN_EXTRA_ARGS", "").strip()
    if not env_extra_args:
        return sys.argv[1:]
    return [*shlex.split(env_extra_args), *sys.argv[1:]]


def _should_bootstrap_accelerate(mode: str) -> bool:
    return (
        mode == "train"
        and os.getenv("TRAIN_ACCELERATE_BOOTSTRAPPED") != "1"
        and os.getenv("LOCAL_RANK") is None
        and os.getenv("WORLD_SIZE") is None
    )


def _bootstrap_train_via_accelerate(cfg: dict[str, Any], argv: list[str]) -> None:
    command = build_accelerate_launch_command(
        cfg,
        script_path=Path(__file__).resolve(),
        script_args=argv,
    )
    child_env = build_accelerate_launch_env()
    os.execvpe(sys.executable, command, child_env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a T5 recipe-correction model with Accelerate.")
    parser.add_argument(
        "--config",
        default=os.getenv("TRAIN_CONFIG") or resolve_default_config_path(),
        help="Path to the training config YAML file.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "tune"),
        default=os.getenv("TRAIN_MODE", "train"),
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
    argv = _resolve_cli_argv()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.experiment_name:
        cfg.setdefault("mlflow", {})
        cfg["mlflow"]["active_experiment_name"] = args.experiment_name
    if _should_bootstrap_accelerate(args.mode):
        _bootstrap_train_via_accelerate(cfg, argv)
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
