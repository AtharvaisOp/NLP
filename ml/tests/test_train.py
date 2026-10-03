"""Orchestration checks with fake runtime: no model download or real training."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ml import train
from ml.config import TrainingConfig
from ml.dataset import DatasetRecord, LABEL_MAPPING


def fake_training_runtime(monkeypatch, tmp_path, *, fail_at=None, changed_data=False):
    events = []
    processed = tmp_path / "processed"
    processed.mkdir()
    train._write_json(processed / "dataset_report.json", {
        "dataset_name": "fixture", "split_strategy": "official_upstream",
        "source_metadata": {"source_url": "https://example.test/fixture", "source_revision": "fixture-revision"},
    })
    records = {
        split: [DatasetRecord(
            record_id=f"{split}-{label}", text=f"मराठी {split} {label}",
            model_text=f"मराठी {split} {label}", label=label, label_name=name,
            domain=None, source_file="fixture.csv", source_row=label + 2, split=split,
        ) for name, label in LABEL_MAPPING.items()]
        for split in ("train", "validation", "test")
    }

    def load_split(path, split):
        events.append(f"load:{split}")
        if split == "test":
            assert "trained" in events and "evaluate:validation" in events
        return records[split]

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            events.append("load:tokenizer")
            return cls()

        def save_pretrained(self, path):
            train._write_json(path / "tokenizer_config.json", {"fixture": True})

    class Model:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            events.append("load:model")
            assert kwargs["label2id"] == LABEL_MAPPING
            return cls()

    class Arguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return self.kwargs

    class Dataset:
        def __init__(self, split_records, *args):
            self.records = split_records
            self.split = split_records[0].split
            events.append(f"tokenize:{self.split}")

    class Trainer:
        def __init__(self, **kwargs):
            selected = Path(kwargs["args"].kwargs["output_dir"]) / "checkpoint-3"
            selected.mkdir()
            self.state = SimpleNamespace(
                best_model_checkpoint=str(selected), best_metric=1.0,
                global_step=3, epoch=3.0,
            )
            if fail_at == "selection":
                self.state.best_model_checkpoint = None
            assert kwargs["train_dataset"].split == "train"
            assert kwargs["eval_dataset"].split == "validation"

        def add_callback(self, callback):
            pass

        def train(self, **kwargs):
            assert "load:test" not in events
            events.append("train")
            if fail_at == "train":
                raise RuntimeError("fixture training failure")
            events.append("trained")

        def evaluate(self, *, eval_dataset):
            assert eval_dataset.split == "validation" and "trained" in events
            assert "load:test" not in events
            events.append("evaluate:validation")
            if fail_at == "validation":
                raise RuntimeError("fixture validation failure")
            return {"eval_loss": 0.1, "eval_accuracy": 1.0, "eval_macro_f1": 1.0}

        def predict(self, dataset):
            assert dataset.split == "test" and "evaluate:validation" in events
            assert "predict:test" not in events
            events.append("predict:test")
            if fail_at == "prediction":
                raise RuntimeError("fixture prediction failure")
            return SimpleNamespace(predictions=np.eye(3) * 8, label_ids=np.asarray([0, 1, 2]))

        def save_model(self, path):
            if fail_at == "save":
                raise RuntimeError("fixture save failure")
            (path / "model.safetensors").write_bytes(b"fixture; not actual weights")
            train._write_json(path / "config.json", {"label2id": LABEL_MAPPING})

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(train, "_require_training_dependencies", lambda: (
        np, fake_torch, Model, Tokenizer, lambda **kwargs: object(), lambda **kwargs: object(),
        object, Trainer, Arguments,
    ))
    monkeypatch.setattr(train, "load_split_records", load_split)
    monkeypatch.setattr(train, "TextClassificationDataset", Dataset)
    monkeypatch.setattr(train, "seed_everything", lambda seed: None)
    monkeypatch.setattr(train, "runtime_info", lambda: {"device": "fixture"})
    monkeypatch.setattr(train, "_trainer_processing_argument", lambda tokenizer: {})
    monkeypatch.setattr(train, "audit_prepared_dataset", lambda path: {"sha256": {"fixture": "original"}})
    monkeypatch.setattr(train, "prepared_data_integrity", lambda path: {"fixture": "changed" if changed_data else "original"})
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    return processed, events


def test_test_split_is_deferred_until_training_and_validation_selection_finish(tmp_path, monkeypatch):
    processed, events = fake_training_runtime(monkeypatch, tmp_path)
    artifact = train.train_model(processed, tmp_path / "artifacts", TrainingConfig(), "fixture-full")
    assert events.index("trained") < events.index("evaluate:validation") < events.index("load:test")
    assert events.count("load:test") == events.count("predict:test") == 1
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_summary"]["selection_split"] == "validation"
    assert Path(manifest["final_test_evaluation"]["checkpoint"]).name == "checkpoint-3"
    assert manifest["final_test_evaluation"]["count"] == 3
    assert manifest["smoke_test"] is False
    assert manifest["production_ready"] is False
    assert manifest["lifecycle_validation"]["held_out_test_evaluated"] is True
    assert not manifest["lifecycle_validation"]["api_integration_validated"]
    assert not (artifact / "_trainer" / "final-test.lock").exists()


@pytest.mark.parametrize("stage", ["train", "validation"])
def test_failed_selection_never_loads_or_predicts_test(tmp_path, monkeypatch, stage):
    processed, events = fake_training_runtime(monkeypatch, tmp_path, fail_at=stage)
    with pytest.raises(RuntimeError, match="fixture"):
        train.train_model(processed, tmp_path / "artifacts", TrainingConfig(), "fixture-full")
    assert "load:test" not in events and "predict:test" not in events


def test_dataset_mutation_refuses_final_evaluation(tmp_path, monkeypatch):
    processed, events = fake_training_runtime(monkeypatch, tmp_path, changed_data=True)
    with pytest.raises(train.TrainingError, match="dataset changed"):
        train.train_model(processed, tmp_path / "artifacts", TrainingConfig(), "fixture-full")
    assert "load:test" not in events and "predict:test" not in events


def test_missing_selected_checkpoint_refuses_test_access(tmp_path, monkeypatch):
    processed, events = fake_training_runtime(monkeypatch, tmp_path, fail_at="selection")
    with pytest.raises(train.TrainingError, match="checkpoint selection did not complete"):
        train.train_model(processed, tmp_path / "artifacts", TrainingConfig(), "fixture-full")
    assert "load:test" not in events and "predict:test" not in events


@pytest.mark.parametrize("stage", ["prediction", "save"])
def test_interrupted_final_prediction_preserves_once_only_lock(tmp_path, monkeypatch, stage):
    processed, events = fake_training_runtime(monkeypatch, tmp_path, fail_at=stage)
    with pytest.raises(RuntimeError, match="fixture"):
        train.train_model(processed, tmp_path / "artifacts", TrainingConfig(), "fixture-full")
    artifact = tmp_path / "artifacts" / "sentiment" / "fixture-full"
    assert events.count("predict:test") == 1
    assert (artifact / "_trainer" / "final-test.lock").exists()
    assert not (artifact / "model_manifest.json").exists()
