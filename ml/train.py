"""MuRIL fine-tuning entry point with lazy heavy dependencies."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import TrainingConfig
from .checkpoints import CheckpointError, validate_resume_checkpoint
from .audit import audit_prepared_dataset, prepared_data_integrity
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


def _training_arguments(
    TrainingArguments: Any, config: TrainingConfig, output_dir: Path, train_size: int
) -> Any:
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "weight_decay": config.weight_decay,
        "seed": config.random_seed,
        "data_seed": config.random_seed,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 100 if not config.smoke_test else 1,
        "disable_tqdm": True,
        "report_to": [],
    }
    parameters = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = config.warmup_ratio
    elif "warmup_steps" in parameters:
        # Transformers versions that removed warmup_ratio still accept an
        # equivalent absolute-step setting. Smoke mode uses its explicit step
        # limit; full mode uses the standard batch/epoch estimate.
        estimated_steps = (
            config.smoke_max_steps
            if config.smoke_test
            else max(1, math.ceil(train_size / config.train_batch_size) * math.ceil(config.num_epochs))
        )
        kwargs["warmup_steps"] = int(round(config.warmup_ratio * estimated_steps))
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if config.smoke_test:
        kwargs["max_steps"] = config.smoke_max_steps
    return TrainingArguments(**kwargs)


def _trainer_processing_argument(tokenizer: Any) -> dict[str, Any]:
    """Bridge Trainer's tokenizer/processing_class rename across versions."""

    try:
        from transformers import Trainer

        parameters = inspect.signature(Trainer.__init__).parameters
    except (ImportError, ValueError):
        parameters = {}
    return {"processing_class" if "processing_class" in parameters else "tokenizer": tokenizer}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_integrity(artifact_dir: Path) -> dict[str, dict[str, int | str]]:
    integrity = {}
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file() or path.name == "model_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        integrity[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return integrity


def train_model(
    processed_dir: Path,
    artifact_root: Path,
    config: TrainingConfig,
    model_version: str,
    *,
    resume_from_checkpoint: Path | None = None,
) -> Path:
    """Fine-tune MuRIL and write a complete versioned artifact."""

    dataset_audit = audit_prepared_dataset(processed_dir)
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
    if config.smoke_test:
        train_records = _limit_smoke(train_records, config.smoke_max_train_samples)
        validation_records = _limit_smoke(validation_records, config.smoke_max_eval_samples)
    if not train_records or not validation_records:
        raise TrainingError("Training and validation manifests must contain records")
    artifact_dir = artifact_root / "sentiment" / model_version
    if artifact_dir.exists() and resume_from_checkpoint is None:
        raise TrainingError(f"Artifact directory already exists: {artifact_dir}")
    if resume_from_checkpoint is not None:
        try:
            resume_from_checkpoint = validate_resume_checkpoint(
                artifact_dir, resume_from_checkpoint, config
            )
        except CheckpointError as exc:
            raise TrainingError(str(exc)) from exc
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )
    artifact_dir.mkdir(parents=True, exist_ok=resume_from_checkpoint is not None)
    train_dataset = TextClassificationDataset(train_records, tokenizer, config.max_length, torch)
    validation_dataset = TextClassificationDataset(validation_records, tokenizer, config.max_length, torch)
    training_dir = artifact_dir / "_trainer"
    training_dir.mkdir(exist_ok=True)
    started_timestamp = datetime.now(timezone.utc)
    run_path = training_dir / f"run-{started_timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    run_record = {
        "started_timestamp": started_timestamp.isoformat(),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "training_config": config.as_dict(),
        "runtime": runtime_info(),
        "completed": False,
    }
    _write_json(run_path, run_record)
    args = _training_arguments(TrainingArguments, config, training_dir, len(train_records))
    _write_json(training_dir / "effective_training_arguments.json", args.to_dict())

    def compute_metrics(prediction: Any) -> dict[str, float]:
        return _metrics_for_prediction(prediction, np)

    trainer_kwargs = _trainer_processing_argument(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        **trainer_kwargs,
    )
    from transformers import TrainerCallback

    class MemoryTelemetry(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if torch.cuda.is_available() and logs is not None:
                logs["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
                logs["gpu_reserved_bytes"] = torch.cuda.memory_reserved()

    trainer.add_callback(MemoryTelemetry())
    training_started = perf_counter()
    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else None
    )
    training_seconds = perf_counter() - training_started
    selected_checkpoint = trainer.state.best_model_checkpoint
    if not config.smoke_test and (
        not selected_checkpoint
        or Path(selected_checkpoint).resolve().parent != training_dir.resolve()
        or not Path(selected_checkpoint).is_dir()
        or type(trainer.state.best_metric) not in (int, float)
        or not math.isfinite(trainer.state.best_metric)
        or trainer.state.global_step <= 0
    ):
        raise TrainingError("Validation-based checkpoint selection did not complete; test evaluation refused")
    training_summary = {
        "completed": True,
        "selected_checkpoint": selected_checkpoint,
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "best_validation_macro_f1": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "training_duration_seconds": training_seconds,
        "resumed_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
    }
    run_record.update(training_summary)
    run_record["completed_timestamp"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_path, run_record)
    print(json.dumps({"training_summary": training_summary}), flush=True)
    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset)
    if prepared_data_integrity(processed_dir) != dataset_audit["sha256"]:
        raise TrainingError("Prepared dataset changed during training; test evaluation refused")
    # Only create and predict the held-out test dataset after training and
    # validation-based checkpoint selection have finished. Never select on it.
    test_records = load_split_records(processed_dir, "test")
    if config.smoke_test:
        test_records = _limit_smoke(test_records, config.smoke_max_eval_samples)
    if not test_records:
        raise TrainingError("Held-out test manifest is empty")
    test_dataset = TextClassificationDataset(test_records, tokenizer, config.max_length, torch)
    test_lock = training_dir / "final-test.lock"
    if not config.smoke_test:
        try:
            with test_lock.open("x", encoding="utf-8") as handle:
                handle.write("Final test prediction claimed; inspect an interrupted evaluation before retrying.\n")
        except FileExistsError as exc:
            raise TrainingError("Final held-out evaluation requires interruption review") from exc
    print("FINAL HELD-OUT TEST EVALUATION: one prediction pass", flush=True)
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
    final_test_evaluation = {
        "split": "test",
        "completed": True,
        "count": len(test_records),
        "checkpoint": selected_checkpoint,
        "dataset_revision": source_metadata["source_revision"],
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    manifest = {
        "project": "MahaPulse",
        "task": "sentiment classification",
        "architecture": "AutoModelForSequenceClassification",
        "base_model": config.model_name,
        "model_name": config.model_name,
        "huggingface_source": config.model_name,
        "num_labels": 3,
        "model_version": model_version,
        "label_mapping": LABEL_MAPPING,
        "dataset_name": dataset_report["dataset_name"],
        "dataset_source": source_metadata["source_url"],
        "dataset_revision": source_metadata["source_revision"],
        "upstream_dataset_revision": source_metadata["source_revision"],
        "split_strategy": dataset_report["split_strategy"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "max_length": config.max_length,
        "training_config": config.as_dict(),
        "training_seed": config.random_seed,
        "smoke_test": config.smoke_test,
        "production_ready": False,
        "lifecycle_validation": {
            "training_completed": True,
            "held_out_test_evaluated": not config.smoke_test,
            "artifact_reload_validated": False,
            "integrity_verified": False,
            "api_integration_validated": False,
        },
        "created_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime": runtime_info(),
        "training_summary": training_summary,
        "final_test_evaluation": final_test_evaluation,
        "dataset_verification": dataset_audit,
        "evaluation_summary": None if config.smoke_test else detailed_test_metrics,
    }
    manifest["artifact_integrity"] = _artifact_integrity(artifact_dir)
    _write_json(artifact_dir / "model_manifest.json", manifest)
    if not config.smoke_test:
        test_lock.unlink()
    return artifact_dir
