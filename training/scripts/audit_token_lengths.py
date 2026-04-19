#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit token lengths for recipe train/eval JSONL files.",
    )
    parser.add_argument(
        "--config",
        default="config/config.t5-base.yaml",
        help="Training config used to resolve the tokenizer and dataset paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "eval"],
        choices=["train", "eval"],
        help="Dataset splits to audit.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[256, 384, 512],
        help="Token limits to evaluate for truncation rates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to scan per split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="How many rows to tokenize per batch.",
    )
    return parser.parse_args()


def percentile(sorted_values: list[int], pct: int) -> int:
    if not sorted_values:
        return 0
    index = math.ceil((pct / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def format_rate(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def resolve_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return cfg


def remap_container_path(path_value: str, repo_root: Path, training_root: Path) -> Path:
    raw_path = Path(path_value)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
        raw_parts = raw_path.parts
        if len(raw_parts) > 1 and raw_parts[1] == "app":
            relative_parts = raw_parts[2:]
            candidates.append(training_root.joinpath(*relative_parts))
            candidates.append(repo_root.joinpath(*relative_parts))
    else:
        candidates.append((training_root / raw_path).resolve())
        candidates.append((repo_root / raw_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_dataset_path(repo_root: Path, training_root: Path, config_path: Path, cfg: dict[str, Any], split: str) -> Path:
    data_cfg = cfg.get("data", {})
    explicit = data_cfg.get("train_path") if split == "train" else data_cfg.get("eval_path")
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        return candidate

    file_name = data_cfg.get("train_file", "train.jsonl") if split == "train" else data_cfg.get("eval_file", "eval.jsonl")
    data_dir = data_cfg.get("data_dir")
    candidates: list[Path] = []

    if data_dir:
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            data_dir_path = (config_path.parent / data_dir_path).resolve()
        candidates.append(data_dir_path / file_name)

    candidates.extend(
        [
            repo_root / "data" / file_name,
            training_root / "data" / file_name,
            repo_root / file_name,
            training_root / file_name,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_model_source(repo_root: Path, training_root: Path, cfg: dict[str, Any]) -> str:
    model_name = str(cfg.get("model", {}).get("name", "")).strip()
    cache_dir = cfg.get("huggingface", {}).get("cache_dir")
    if cache_dir:
        cache_path = remap_container_path(str(cache_dir), repo_root, training_root)
        model_cache_dir = cache_path / f"models--{model_name.replace('/', '--')}" / "snapshots"
        if model_cache_dir.exists():
            snapshots = sorted(path for path in model_cache_dir.iterdir() if path.is_dir())
            if snapshots:
                return str(snapshots[0])
    return model_name


def load_lengths(
    dataset_path: Path,
    tokenizer: Any,
    task_prefix: str,
    limit: int | None,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    pending_inputs: list[str] = []
    pending_targets: list[str] = []

    def flush_batch() -> None:
        if not pending_inputs:
            return
        encoded_inputs = tokenizer(pending_inputs, truncation=False)
        encoded_targets = tokenizer(pending_targets, truncation=False)
        input_lengths.extend(len(ids) for ids in encoded_inputs.input_ids)
        target_lengths.extend(len(ids) for ids in encoded_targets.input_ids)
        pending_inputs.clear()
        pending_targets.clear()

    with open(dataset_path, "r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            input_text = str(record["input"])
            target_text = str(record["target"])
            prefixed_input = f"{task_prefix}{input_text}" if task_prefix and not input_text.startswith(task_prefix) else input_text
            pending_inputs.append(prefixed_input)
            pending_targets.append(target_text)
            if len(pending_inputs) >= batch_size:
                flush_batch()

    flush_batch()

    return input_lengths, target_lengths


def print_length_block(name: str, values: list[int], thresholds: list[int]) -> None:
    ordered = sorted(values)
    print(name)
    print(f"  rows: {len(ordered)}")
    print(
        "  tokens:"
        f" min={ordered[0]}"
        f" p50={percentile(ordered, 50)}"
        f" p90={percentile(ordered, 90)}"
        f" p95={percentile(ordered, 95)}"
        f" p99={percentile(ordered, 99)}"
        f" max={ordered[-1]}"
    )
    print("  truncation rates:")
    for threshold in thresholds:
        truncated = sum(1 for value in ordered if value > threshold)
        print(f"    > {threshold}: {truncated} / {len(ordered)} ({format_rate(truncated, len(ordered))})")


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    repo_root = Path(__file__).resolve().parents[2]
    training_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
        if not config_path.exists():
            config_path = (training_root / args.config).resolve()
        if not config_path.exists():
            config_path = (repo_root / args.config).resolve()

    cfg = resolve_config(config_path)
    task_prefix = str(cfg.get("model", {}).get("task_prefix", ""))
    model_source = resolve_model_source(repo_root, training_root, cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)

    print("TOKEN LENGTH AUDIT")
    print(f"config: {config_path}")
    print(f"model_source: {model_source}")
    print(f"task_prefix: {task_prefix!r}")
    print(f"thresholds: {thresholds}")
    if args.limit is not None:
        print(f"row_limit_per_split: {args.limit}")
    print("")

    for split in args.splits:
        dataset_path = resolve_dataset_path(repo_root, training_root, config_path, cfg, split)
        input_lengths, target_lengths = load_lengths(
            dataset_path,
            tokenizer,
            task_prefix,
            args.limit,
            args.batch_size,
        )
        print(f"[{split}] {dataset_path}")
        print_length_block("input", input_lengths, thresholds)
        print_length_block("target", target_lengths, thresholds)
        print("")


if __name__ == "__main__":
    main()
