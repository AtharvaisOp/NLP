"""Small deterministic production metadata fixtures without model downloads."""

import hashlib
import json
from pathlib import Path

from ml.config import TrainingConfig
from ml.dataset import LABEL_MAPPING
from ml.metrics import classification_metrics
from ml.train import _artifact_integrity, _write_json


def write_full_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "full-artifact"
    artifact.mkdir()
    config = TrainingConfig(train_batch_size=4, eval_batch_size=8).as_dict()
    test_metrics = classification_metrics([0, 1, 2], [0, 1, 2])
    revision = "8ee29fa1329d6a841030eb46659d3c10614b5e59"
    manifest = {
        "project": "MahaPulse", "task": "sentiment classification",
        "architecture": "AutoModelForSequenceClassification",
        "base_model": "google/muril-base-cased", "model_version": "fixture-full-v1",
        "max_length": 256, "smoke_test": False, "production_ready": False,
        "label_mapping": LABEL_MAPPING, "preprocessing_version": "model-text-v1",
        "dataset_revision": revision, "training_config": config,
        "evaluation_summary": test_metrics,
        "training_summary": {
            "completed": True, "selection_split": "validation", "selection_metric": "macro_f1",
            "selected_checkpoint": "checkpoint-12", "training_duration_seconds": 10.0,
            "global_step": 12, "epoch": 3.0,
        },
        "final_test_evaluation": {
            "completed": True, "split": "test", "count": 3,
            "checkpoint": "checkpoint-12", "dataset_revision": revision,
            "preprocessing_version": "model-text-v1",
        },
        "lifecycle_validation": {
            "training_completed": True, "held_out_test_evaluated": True,
            "artifact_reload_validated": False, "integrity_verified": False,
            "api_integration_validated": False,
        },
    }
    report = {
        "label_mapping": LABEL_MAPPING, "preprocessing_version": "model-text-v1",
        "source_metadata": {"source_revision": revision}, "valid_rows": 9,
        "split_distribution": {"train": 3, "validation": 3, "test": 3},
        "class_distribution": {"negative": 3, "neutral": 3, "positive": 3},
    }
    for name, value in {
        "config.json": {"label2id": LABEL_MAPPING, "id2label": {str(value): key for key, value in LABEL_MAPPING.items()}},
        "label_mapping.json": {"canonical_to_id": LABEL_MAPPING},
        "tokenizer_config.json": {}, "tokenizer.json": {},
        "training_config.json": config, "dataset_report.json": report,
        "metrics.json": {"smoke_test": False, "validation": test_metrics, "test": test_metrics},
    }.items():
        _write_json(artifact / name, value)
    (artifact / "model.safetensors").write_bytes(b"fixture weights, never loaded")
    with (artifact / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for label in LABEL_MAPPING:
            row = {"id": f"test-{label}", "text": f"मराठी {label}", "gold_label": label,
                   "predicted_label": label, "confidence": 0.8,
                   "class_probabilities": {name: 0.8 if name == label else 0.1 for name in LABEL_MAPPING}}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["artifact_integrity"] = _artifact_integrity(artifact)
    _write_json(artifact / "model_manifest.json", manifest)
    return artifact


def write_validation_report(artifact: Path, tmp_path: Path) -> Path:
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1, "status": "passed", "phase": "prepromotion", "artifact_dir": str(artifact),
        "artifact": {
            "model_version": manifest["model_version"], "smoke_test": False, "production_ready": False,
            "integrity_verified": True, "unchanged_by_validation": True,
            "artifact_integrity": manifest["artifact_integrity"],
            "manifest_sha256": hashlib.sha256((artifact / "model_manifest.json").read_bytes()).hexdigest(),
            "held_out_prediction_records_verified": 3,
        },
        "independent_reload": {"passed": True, "local_files_only": True},
        "application": {
            "passed": True,
            "operations": {
                "/health": {"status": "ok"},
                "/ready": {"services": {"sentiment": {"state": "ready", "smoke_test": False, "production_ready": False}}},
                "/v1/model-info": {"sentiment_model": {
                    "backend": "muril", "state": "ready", "version": manifest["model_version"],
                    "smoke_test": False, "production_ready": False,
                }},
            },
            "batch": {"total_documents": 5, "successful_documents": 4, "failed_documents": 1,
                      "status": "partial", "model_version": manifest["model_version"]},
            "exports": {"csv_rows": 5, "json_documents": 5},
        },
    }
    report_path = tmp_path / "api-validation.json"
    _write_json(report_path, report)
    return report_path
