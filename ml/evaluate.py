"""Held-out test evaluation for a saved MahaPulse sentiment artifact."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import load_split_records
from .metrics import classification_metrics, prediction_records
from .train import (
    TextClassificationDataset,
    TrainingError,
    _artifact_integrity,
    _require_training_dependencies,
    _trainer_processing_argument,
    _write_json,
)


def evaluate_artifact(artifact_dir: Path, processed_dir: Path) -> dict[str, object]:
    dependencies = _require_training_dependencies()
    np, torch, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, _, _, Trainer, _ = dependencies
    manifest = json.loads((artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
    records = load_split_records(processed_dir, "test")
    if not records:
        raise TrainingError("Held-out test manifest is empty")
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    dataset = TextClassificationDataset(records, tokenizer, int(manifest["max_length"]), torch)
    from transformers import TrainingArguments

    trainer_kwargs = _trainer_processing_argument(tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(artifact_dir / "_evaluation"),
            per_device_eval_batch_size=32,
            report_to=[],
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        **trainer_kwargs,
    )
    prediction = trainer.predict(dataset)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    metrics = classification_metrics(prediction.label_ids.tolist(), probabilities.argmax(axis=1).tolist(), probabilities)
    metrics["smoke_test"] = bool(manifest.get("smoke_test", False))
    metrics_path = artifact_dir / "metrics.json"
    existing_metrics = {}
    if metrics_path.is_file():
        existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing_metrics.update({"test": metrics, "smoke_test": metrics["smoke_test"]})
    _write_json(metrics_path, existing_metrics)
    with (artifact_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in prediction_records(records, probabilities.argmax(axis=1), probabilities):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["artifact_integrity"] = _artifact_integrity(artifact_dir)
    _write_json(artifact_dir / "model_manifest.json", manifest)
    return metrics
