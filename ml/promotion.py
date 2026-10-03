"""Explicit production promotion for a completed sentiment artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifact_validation import (
    ArtifactEvidenceError, validate_production_evidence, verify_integrity, weight_files,
)
from .dataset import LABEL_MAPPING
from .preprocessing import PREPROCESSING_VERSION
from .train import _artifact_integrity, _write_json


class PromotionError(RuntimeError):
    """Raised when an artifact has not satisfied every production gate."""


REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "label_mapping.json",
    "model_manifest.json",
    "dataset_report.json",
    "training_config.json",
    "metrics.json",
    "predictions.jsonl",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"artifact metadata is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"artifact metadata must be an object: {path.name}")
    return value


def promote_artifact(
    artifact_dir: Path, *, api_integration_validated: bool,
    validation_report: Path | None = None,
) -> dict[str, object]:
    """Reload, verify, and mark a completed full artifact production-ready."""

    artifact_dir = artifact_dir.resolve()
    if not api_integration_validated:
        raise PromotionError("real API integration must pass before production promotion")
    if validation_report is None:
        raise PromotionError("real API integration validation report is required")
    if not artifact_dir.is_dir():
        raise PromotionError("artifact directory is unavailable")
    if any(not (artifact_dir / filename).is_file() for filename in REQUIRED_FILES):
        raise PromotionError("artifact is missing required production files")
    manifest = _read_json(artifact_dir / "model_manifest.json")
    mapping = _read_json(artifact_dir / "label_mapping.json")
    model_config = _read_json(artifact_dir / "config.json")
    if (
        manifest.get("project") != "MahaPulse"
        or manifest.get("base_model") != "google/muril-base-cased"
        or manifest.get("task") != "sentiment classification"
        or manifest.get("architecture") != "AutoModelForSequenceClassification"
        or manifest.get("preprocessing_version") != PREPROCESSING_VERSION
    ):
        raise PromotionError("artifact project, model or preprocessing is incompatible")
    if manifest.get("label_mapping") != LABEL_MAPPING:
        raise PromotionError("artifact label mapping is incompatible")
    expected_ids = {str(identifier): label for label, identifier in LABEL_MAPPING.items()}
    if (
        mapping.get("canonical_to_id") != LABEL_MAPPING
        or model_config.get("label2id") != LABEL_MAPPING
        or model_config.get("id2label") != expected_ids
    ):
        raise PromotionError("artifact classifier or tokenizer labels are incompatible")
    try:
        weight_files(artifact_dir)
        verify_integrity(artifact_dir, manifest, production=True)
        expected_test_count = validate_production_evidence(artifact_dir, manifest)
    except ArtifactEvidenceError as exc:
        raise PromotionError(str(exc)) from exc
    actual_integrity = _artifact_integrity(artifact_dir)
    report = _read_json(validation_report)
    artifact_evidence = report.get("artifact", {})
    independent_reload = report.get("independent_reload", {})
    application = report.get("application", {})
    if not all(isinstance(value, dict) for value in (artifact_evidence, independent_reload, application)):
        raise PromotionError("real API integration validation report is invalid")
    tested_manifest_hash = hashlib.sha256((artifact_dir / "model_manifest.json").read_bytes()).hexdigest()
    if (
        report.get("schema_version") != 1 or report.get("status") != "passed"
        or report.get("phase") != "prepromotion"
        or not isinstance(report.get("artifact_dir"), str)
        or Path(report["artifact_dir"]).resolve() != artifact_dir
        or artifact_evidence.get("model_version") != manifest.get("model_version")
        or artifact_evidence.get("smoke_test") is not False
        or artifact_evidence.get("production_ready") is not False
        or artifact_evidence.get("integrity_verified") is not True
        or artifact_evidence.get("unchanged_by_validation") is not True
        or artifact_evidence.get("artifact_integrity") != actual_integrity
        or artifact_evidence.get("manifest_sha256") != tested_manifest_hash
        or artifact_evidence.get("held_out_prediction_records_verified") != expected_test_count
        or independent_reload.get("passed") is not True
        or independent_reload.get("local_files_only") is not True
        or application.get("passed") is not True
    ):
        raise PromotionError("real API validation report does not prove this unchanged full artifact")
    operations, batch, exports = application.get("operations"), application.get("batch"), application.get("exports")
    if not all(isinstance(value, dict) for value in (operations, batch, exports)):
        raise PromotionError("real API integration report is missing operations or batch/export evidence")
    health, ready, info = operations.get("/health"), operations.get("/ready"), operations.get("/v1/model-info")
    if not all(isinstance(value, dict) for value in (health, ready, info)):
        raise PromotionError("real API integration report is missing endpoint evidence")
    services = ready.get("services")
    if not isinstance(services, dict):
        raise PromotionError("real API readiness evidence is invalid")
    sentiment_ready = services.get("sentiment", {})
    sentiment_info = info.get("sentiment_model", {})
    if (
        not isinstance(sentiment_ready, dict) or not isinstance(sentiment_info, dict)
        or health.get("status") != "ok" or sentiment_ready.get("state") != "ready"
        or sentiment_ready.get("smoke_test") is not False
        or sentiment_ready.get("production_ready") is not False
        or sentiment_info.get("backend") != "muril" or sentiment_info.get("state") != "ready"
        or sentiment_info.get("version") != manifest.get("model_version")
        or sentiment_info.get("smoke_test") is not False
        or sentiment_info.get("production_ready") is not False
        or batch.get("total_documents") != 5 or batch.get("successful_documents") != 4
        or batch.get("failed_documents") != 1 or batch.get("status") != "partial"
        or batch.get("model_version") != manifest.get("model_version")
        or exports.get("csv_rows") != 5 or exports.get("json_documents") != 5
    ):
        raise PromotionError("real API sentiment or batch/export integration evidence is incomplete")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            artifact_dir, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - converted at the CLI boundary
        raise PromotionError("independent local artifact reload failed") from exc
    del model, tokenizer

    manifest["production_ready"] = True
    manifest["lifecycle_validation"] = {
        "training_completed": True,
        "held_out_test_evaluated": True,
        "artifact_reload_validated": True,
        "integrity_verified": True,
        "api_integration_validated": True,
    }
    manifest["promoted_timestamp"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    # The manifest itself is intentionally excluded, so promotion metadata does
    # not create a self-referential checksum.
    manifest["artifact_integrity"] = actual_integrity
    manifest["api_validation_evidence"] = {
        "schema_version": 1,
        "report_sha256": hashlib.sha256(validation_report.read_bytes()).hexdigest(),
        "tested_manifest_sha256": tested_manifest_hash,
        "model_version": manifest["model_version"],
        "artifact_integrity": actual_integrity,
        "independent_local_reload_passed": True,
        "application_passed": True,
        "batch_successes": 4, "batch_expected_failures": 1,
        "csv_rows": 5, "json_documents": 5,
    }
    _write_json(artifact_dir / "model_manifest.json", manifest)
    return {
        "artifact_dir": str(artifact_dir),
        "model_version": manifest.get("model_version"),
        "production_ready": True,
        "integrity_files": len(actual_integrity),
        "held_out_predictions": expected_test_count,
    }
