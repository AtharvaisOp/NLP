"""MuRIL fine-tuning entry point with lazy heavy dependencies."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .dataset import LABEL_MAPPING, DatasetRecord, load_split_records
from .metrics import classification_metrics, prediction_records
from .preprocessing import PREPROCESSING_VERSION
from .reproducibility import runtime_info, seed_everything


class TrainingError(RuntimeError):
    """Raised when training cannot start or complete safely."""


def _require_training_dependencies():
    try:
        import numpy as np
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            EvalPrediction,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise TrainingError(
            "MuRIL training requires ml/requirements.txt; no model download was attempted"
        ) from exc
    return (
        np,
        torch,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        EvalPrediction,
        Trainer,
        TrainingArguments,
    )


class TextClassificationDataset:
    def __init__(self, records: list[DatasetRecord], tokenizer: Any, max_length: int, torch: Any):
        self.records = records
        self.encodings = tokenizer(
            [record.model_text for record in records],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            key: self.torch.tensor(value[index], dtype=self.torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = self.torch.tensor(self.records[index].label, dtype=self.torch.long)
        return item


def _limit_smoke(records: list[DatasetRecord], limit: int) -> list[DatasetRecord]:
    by_label: dict[int, list[DatasetRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    selected: list[DatasetRecord] = []
    while len(selected) < limit and any(by_label.values()):
        for label in sorted(by_label):
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].pop(0))
    return selected


def _metrics_for_prediction(eval_prediction: Any, np: Any) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1).tolist()
    metrics = classification_metrics(eval_prediction.label_ids.tolist(), predicted, probabilities)
    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def _training_arguments(TrainingArguments: Any, config: TrainingConfig, output_dir: Path) -> Any:
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "seed": config.random_seed,
        "data_seed": config.random_seed,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "report_to": [],
    }
    parameters = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if config.smoke_test:
        kwargs["max_steps"] = config.smoke_max_steps
    return TrainingArguments(**kwargs)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_model(
    processed_dir: Path,
    artifact_root: Path,
    config: TrainingConfig,
    model_version: str,
) -> Path:
    """Fine-tune MuRIL and write a complete versioned artifact."""

    dependencies = _require_training_dependencies()
    (
        np,
        torch,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        EvalPrediction,
        Trainer,
        TrainingArguments,
    ) = dependencies
    seed_everything(config.random_seed)
    train_records = load_split_records(processed_dir, "train")
    validation_records = load_split_records(processed_dir, "validation")
    test_records = load_split_records(processed_dir, "test")
    if config.smoke_test:
        train_records = _limit_smoke(train_records, config.smoke_max_train_samples)
        validation_records = _limit_smoke(validation_records, config.smoke_max_eval_samples)
        test_records = _limit_smoke(test_records, config.smoke_max_eval_samples)
    if not train_records or not validation_records or not test_records:
        raise TrainingError("Training, validation, and test manifests must all contain records")
    artifact_dir = artifact_root / "sentiment" / model_version
    if artifact_dir.exists():
        raise TrainingError(f"Artifact directory already exists: {artifact_dir}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    train_dataset = TextClassificationDataset(train_records, tokenizer, config.max_length, torch)
    validation_dataset = TextClassificationDataset(validation_records, tokenizer, config.max_length, torch)
    test_dataset = TextClassificationDataset(test_records, tokenizer, config.max_length, torch)
    training_dir = artifact_dir / "_trainer"
    args = _training_arguments(TrainingArguments, config, training_dir)

    def compute_metrics(prediction: Any) -> dict[str, float]:
        return _metrics_for_prediction(prediction, np)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()
    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset)
    test_prediction = trainer.predict(test_dataset)
    test_metrics = _metrics_for_prediction(test_prediction, np)
    logits = test_prediction.predictions[0] if isinstance(test_prediction.predictions, tuple) else test_prediction.predictions
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    detailed_test_metrics = classification_metrics(
        test_prediction.label_ids.tolist(), probabilities.argmax(axis=1).tolist(), probabilities
    )
    tokenizer.save_pretrained(artifact_dir)
    trainer.save_model(artifact_dir)
    dataset_report = json.loads((processed_dir / "dataset_report.json").read_text(encoding="utf-8"))
    metrics = {
        "smoke_test": config.smoke_test,
        "validation": {key.removeprefix("eval_"): value for key, value in validation_metrics.items() if key.startswith("eval_")},
        "test": detailed_test_metrics,
        "test_summary": test_metrics,
    }
    _write_json(artifact_dir / "training_config.json", config.as_dict())
    _write_json(
        artifact_dir / "label_mapping.json",
        {
            "canonical_to_id": LABEL_MAPPING,
            "id_to_canonical": {str(value): key for key, value in LABEL_MAPPING.items()},
        },
    )
    _write_json(artifact_dir / "dataset_report.json", dataset_report)
    _write_json(artifact_dir / "metrics.json", metrics)
    with (artifact_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in prediction_records(test_records, probabilities.argmax(axis=1), probabilities):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_metadata = dataset_report["source_metadata"]
    manifest = {
        "project": "MahaPulse",
        "task": "3-class Marathi sentiment classification",
        "architecture": "AutoModelForSequenceClassification",
        "model_name": config.model_name,
        "huggingface_source": config.model_name,
        "model_version": model_version,
        "label_mapping": LABEL_MAPPING,
        "dataset_name": dataset_report["dataset_name"],
        "dataset_source": source_metadata["source_url"],
        "dataset_revision": source_metadata["source_revision"],
        "split_strategy": dataset_report["split_strategy"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "max_length": config.max_length,
        "training_seed": config.random_seed,
        "smoke_test": config.smoke_test,
        "created_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime": runtime_info(),
        "evaluation_summary": None if config.smoke_test else detailed_test_metrics,
    }
    _write_json(artifact_dir / "model_manifest.json", manifest)
    return artifact_dir
