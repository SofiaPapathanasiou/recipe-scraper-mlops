import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


if sys.version_info < (3, 10):
    pytest.skip("training tests require Python 3.10+.", allow_module_level=True)

pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("boto3")
pytest.importorskip("mlflow")
pytest.importorskip("transformers")
pytest.importorskip("yaml")

import torch


TRAIN_MODULE_PATH = Path(__file__).resolve().parents[1] / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("training_train", TRAIN_MODULE_PATH)
train = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC.loader is not None
TRAIN_SPEC.loader.exec_module(train)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        del padding, truncation, return_tensors
        encoded = [((ord(char) % 11) + 1) for char in text[: max_length - 2]]
        encoded = encoded + [0] * (max_length - len(encoded))
        attention = [1 if token else 0 for token in encoded]
        return SimpleNamespace(
            input_ids=torch.tensor([encoded], dtype=torch.long),
            attention_mask=torch.tensor([attention], dtype=torch.long),
        )

    def batch_decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return ["decoded" for _ in token_ids]


class FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")


def make_cfg() -> dict:
    return {
        "model": {"name": "t5-small", "task_prefix": "fix recipe: "},
        "tokenization": {"max_input_length": 12, "max_target_length": 10},
        "training": {
            "learning_rate": 3.0e-4,
            "weight_decay": 0.01,
            "num_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.06,
            "max_grad_norm": 1.0,
            "seed": 42,
        },
        "evaluation": {
            "logging_steps": 50,
            "eval_steps": 500,
            "metric_for_best_model": "eval_rougeL",
        },
        "checkpointing": {
            "checkpoint_dir": "/tmp/checkpoints",
            "load_best_model_at_end": True,
        },
        "mlflow": {
            "experiment_name": "recipe-correction-t5",
            "tracking_uri": "http://mlflow:5000",
            "register_model": True,
            "registered_model_name": "recipe-correction-t5",
        },
        "data": {
            "source": "mock",
            "mock_train_size": 4,
            "mock_val_size": 2,
            "minio_bucket": "recipe-datasets",
            "minio_train_key": "train.jsonl",
            "minio_val_key": "val.jsonl",
        },
    }


def test_load_config_applies_env_overrides_and_types(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: t5-small
  task_prefix: "fix recipe: "
tokenization:
  max_input_length: 512
  max_target_length: 512
training:
  learning_rate: 3.0e-4
  weight_decay: 0.01
  num_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  warmup_ratio: 0.06
  max_grad_norm: 1.0
  seed: 42
evaluation:
  logging_steps: 50
  eval_steps: 500
  metric_for_best_model: eval_rougeL
checkpointing:
  checkpoint_dir: /tmp/checkpoints
  load_best_model_at_end: true
mlflow:
  experiment_name: recipe-correction-t5
  tracking_uri: http://mlflow:5000
  register_model: true
  registered_model_name: recipe-correction-t5
data:
  source: mock
  mock_train_size: 200
  mock_val_size: 40
  minio_bucket: recipe-datasets
  minio_train_key: train.jsonl
  minio_val_key: val.jsonl
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_NAME", "t5-base")
    monkeypatch.setenv("NUM_EPOCHS", "7")
    monkeypatch.setenv("LEARNING_RATE", "0.001")
    monkeypatch.setenv("DATA_SOURCE", "minio")

    cfg = train.load_config(str(config_path))

    assert cfg["model"]["name"] == "t5-base"
    assert cfg["training"]["num_epochs"] == 7
    assert cfg["training"]["learning_rate"] == 0.001
    assert cfg["data"]["source"] == "minio"


def test_mock_recipe_dataset_uses_requested_size_and_masks_padding():
    cfg = make_cfg()
    cfg["data"]["mock_train_size"] = 3
    dataset = train.MockRecipeDataset(FakeTokenizer(), cfg, "train")

    assert len(dataset) == 3
    sample = dataset[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
    assert (sample["labels"] == -100).any()


def test_get_optimizer_param_groups_separates_decay_and_no_decay():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    named_params = dict(model.named_parameters())

    param_groups = train.get_optimizer_param_groups(model, weight_decay=0.01)
    decay_group = {id(param) for param in param_groups[0]["params"]}
    no_decay_group = {id(param) for param in param_groups[1]["params"]}

    assert id(named_params["0.weight"]) in decay_group
    assert id(named_params["0.bias"]) in no_decay_group
    assert id(named_params["1.weight"]) in no_decay_group
    assert id(named_params["1.bias"]) in no_decay_group


def test_load_minio_dataset_reads_jsonl_with_mocked_boto3(monkeypatch):
    payload = b'{"input":"bad recipe","target":"good recipe"}\n'

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            assert Bucket == "recipe-datasets"
            assert Key == "train.jsonl"
            return {"Body": io.BytesIO(payload)}

    monkeypatch.setattr(train.boto3, "client", lambda *args, **kwargs: FakeS3Client())

    cfg = make_cfg()
    cfg["data"]["source"] = "minio"
    tokenizer = FakeTokenizer()

    dataset = train.load_minio_dataset(cfg, tokenizer, "train")

    assert len(dataset) == 1
    assert dataset.inputs[0].startswith(cfg["model"]["task_prefix"])
    assert dataset.targets[0] == "good recipe"


def test_build_dataloaders_smoke_for_mock_source():
    cfg = make_cfg()
    train_loader, val_loader, train_dataset, val_dataset = train.build_dataloaders(
        cfg,
        FakeTokenizer(),
        FakeAccelerator(),
    )

    assert len(train_dataset) == cfg["data"]["mock_train_size"]
    assert len(val_dataset) == cfg["data"]["mock_val_size"]
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}


def test_resolve_run_checkpoint_dir_uses_mlflow_run_id(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"

    resolved = train.resolve_run_checkpoint_dir(checkpoint_dir, "abc123")

    assert resolved == checkpoint_dir / "abc123"


def test_resolve_run_checkpoint_dir_falls_back_to_manual_run_name(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(train.time, "time", lambda: 1234.56)

    resolved = train.resolve_run_checkpoint_dir(checkpoint_dir, None)

    assert resolved == checkpoint_dir / "manual-run-1234"
