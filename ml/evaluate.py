"""Once-only held-out test evaluation for a selected sentiment artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .artifact_validation import ArtifactEvidenceError, read_evidence_json, verify_integrity
from .dataset import LABEL_MAPPING, load_split_records
from .metrics import classification_metrics, prediction_records
from .preprocessing import PREPROCESSING_VERSION
from .train import (
    TextClassificationDataset,
    TrainingError,
    _artifact_integrity,
    _require_training_dependencies,
    _trainer_processing_argument,
    _write_json,
)


def evaluate_artifact(artifact_dir: Path, processed_dir: Path) -> dict[str, object]:
    """Evaluate only after selection, refusing to overwrite final test evidence."""
    artifact_dir = artifact_dir.resolve()
    try:
        manifest = read_evidence_json(artifact_dir / "model_manifest.json")
        metrics_path = artifact_dir / "metrics.json"
        existing_metrics = read_evidence_json(metrics_path) if metrics_path.is_file() else {}
        # This guard precedes dataset loading, heavy imports, and model reload.
        lifecycle = manifest.get("lifecycle_validation", {})
        if (
            "test" in existing_metrics
            or manifest.get("evaluation_summary") is not None
            or manifest.get("final_test_evaluation")
            or isinstance(lifecycle, dict) and lifecycle.get("held_out_test_evaluated") is True
            or (artifact_dir / "predictions.jsonl").exists()
        ):
            raise TrainingError("Final held-out test evaluation already exists; it cannot be repeated")
        training = manifest.get("training_summary", {})
        if (
            manifest.get("smoke_test") is not False
            or not isinstance(lifecycle, dict) or lifecycle.get("training_completed") is not True
            or not isinstance(training, dict) or training.get("completed") is not True
            or training.get("selection_split") != "validation"
            or not training.get("selected_checkpoint")
        ):
            raise TrainingError("Full training and validation checkpoint selection must finish first")
        report = read_evidence_json(artifact_dir / "dataset_report.json")
        processed_report = read_evidence_json(processed_dir / "dataset_report.json")
        if report != processed_report:
            raise TrainingError("Evaluation dataset differs from the saved training dataset provenance")
        source = report.get("source_metadata")
        if (
            not isinstance(source, dict)
            or manifest.get("dataset_revision") != source.get("source_revision")
            or not manifest.get("dataset_revision")
            or manifest.get("label_mapping") != LABEL_MAPPING
            or manifest.get("preprocessing_version") != PREPROCESSING_VERSION
        ):
            raise TrainingError("Evaluation dataset labels, revision or preprocessing are incompatible")
        verify_integrity(artifact_dir, manifest, production=False)
    except ArtifactEvidenceError as exc:
        raise TrainingError(str(exc)) from exc

    evaluation_dir = artifact_dir / "_evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    lock_path = evaluation_dir / "final-test.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write("Final evaluation claimed; inspect this run before retrying.\n")
    except FileExistsError as exc:
        raise TrainingError("Final held-out evaluation is active or requires interruption review") from exc

    prediction_started = False
    completed = False
    try:
        dependencies = _require_training_dependencies()
        np, torch, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, _, _, Trainer, TrainingArguments = dependencies
        records = load_split_records(processed_dir, "test")
        expected_count = report.get("split_distribution", {}).get("test")
        if type(expected_count) is not int or expected_count <= 0 or len(records) != expected_count:
            raise TrainingError("Held-out test count differs from dataset provenance")
        if any(record.split != "test" or record.label != LABEL_MAPPING.get(record.label_name) for record in records):
            raise TrainingError("Held-out test records contain incompatible split or label evidence")
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(artifact_dir, local_files_only=True)
        dataset = TextClassificationDataset(records, tokenizer, int(manifest["max_length"]), torch)
        config = read_evidence_json(artifact_dir / "training_config.json")
        eval_batch_size = config.get("eval_batch_size", 8)
        if type(eval_batch_size) is not int or eval_batch_size <= 0:
            raise TrainingError("Saved evaluation batch size is invalid")
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(evaluation_dir), per_device_eval_batch_size=eval_batch_size,
                report_to=[],
            ),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            **_trainer_processing_argument(tokenizer),
        )
        prediction_started = True
        prediction = trainer.predict(dataset)
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        logits = np.asarray(logits)
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        metrics = classification_metrics(prediction.label_ids.tolist(), probabilities.argmax(axis=1).tolist(), probabilities)
        existing_metrics.update({"test": metrics, "smoke_test": False})
        _write_json(metrics_path, existing_metrics)
        with (artifact_dir / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in prediction_records(records, probabilities.argmax(axis=1), probabilities):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest["evaluation_summary"] = metrics
        manifest["final_test_evaluation"] = {
            "split": "test", "completed": True, "count": len(records),
            "checkpoint": training["selected_checkpoint"],
            "dataset_revision": manifest["dataset_revision"],
            "preprocessing_version": PREPROCESSING_VERSION,
            "completed_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        manifest["production_ready"] = False
        lifecycle.update({
            "held_out_test_evaluated": True, "artifact_reload_validated": False,
            "integrity_verified": False, "api_integration_validated": False,
        })
        manifest["lifecycle_validation"] = lifecycle
        manifest["artifact_integrity"] = _artifact_integrity(artifact_dir)
        _write_json(artifact_dir / "model_manifest.json", manifest)
        completed = True
        return metrics
    finally:
        # Before prediction, dependency/device errors are safe to retry. After
        # prediction starts, retain an interruption marker unless all evidence
        # was saved; never silently evaluate a held-out set twice.
        if completed or not prediction_started:
            lock_path.unlink(missing_ok=True)
