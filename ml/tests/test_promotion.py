import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ml.promotion import PromotionError, promote_artifact
from ml.tests.artifact_fixtures import write_full_artifact, write_validation_report
from ml.train import _artifact_integrity, _write_json


def test_promotion_requires_real_api_attestation(tmp_path: Path) -> None:
    with pytest.raises(PromotionError, match="API integration"):
        promote_artifact(tmp_path / "missing", api_integration_validated=False)


def test_promotion_requires_report_in_addition_to_attestation(tmp_path: Path) -> None:
    with pytest.raises(PromotionError, match="validation report"):
        promote_artifact(tmp_path / "missing", api_integration_validated=True)


def stub_local_reload(monkeypatch):
    calls = []
    def reload(path, **kwargs):
        calls.append((path, kwargs))
        return object()
    loader = SimpleNamespace(from_pretrained=reload)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=loader, AutoModelForSequenceClassification=loader,
    ))
    return calls


def refresh_integrity(artifact):
    path = artifact / "model_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifact_integrity"] = _artifact_integrity(artifact)
    _write_json(path, manifest)


def test_production_promotion_requires_all_evidence_and_local_reload(tmp_path, monkeypatch):
    artifact = write_full_artifact(tmp_path)
    report = write_validation_report(artifact, tmp_path)
    calls = stub_local_reload(monkeypatch)

    result = promote_artifact(artifact, api_integration_validated=True, validation_report=report)

    assert result["production_ready"] is True
    assert len(calls) == 2
    assert all(kwargs == {"local_files_only": True} for _, kwargs in calls)
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    assert all(manifest["lifecycle_validation"].values())
    assert manifest["api_validation_evidence"]["artifact_integrity"] == _artifact_integrity(artifact)


@pytest.mark.parametrize("change", ["training_incomplete", "revision_mismatch", "test_checkpoint", "missing_weight_hash", "swapped_classifier_labels"])
def test_promotion_rejects_invalid_lifecycle_before_loading(tmp_path, monkeypatch, change):
    artifact = write_full_artifact(tmp_path)
    manifest_path = artifact / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if change == "training_incomplete":
        manifest["lifecycle_validation"]["training_completed"] = False
    elif change == "revision_mismatch":
        manifest["dataset_revision"] = "f" * 40
    elif change == "test_checkpoint":
        manifest["training_summary"]["selection_split"] = "test"
    elif change == "missing_weight_hash":
        del manifest["artifact_integrity"]["model.safetensors"]
    else:
        config_path = artifact / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["id2label"]["0"] = "positive"
        _write_json(config_path, config)
    _write_json(manifest_path, manifest)
    report = write_validation_report(artifact, tmp_path)
    calls = stub_local_reload(monkeypatch)

    with pytest.raises(PromotionError):
        promote_artifact(artifact, api_integration_validated=True, validation_report=report)
    assert calls == []


def test_metrics_are_recomputed_even_if_hashes_and_summary_match(tmp_path, monkeypatch):
    artifact = write_full_artifact(tmp_path)
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    metrics["test"]["accuracy"] = 0.5
    _write_json(artifact / "metrics.json", metrics)
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    manifest["evaluation_summary"] = metrics["test"]
    _write_json(artifact / "model_manifest.json", manifest)
    refresh_integrity(artifact)
    report = write_validation_report(artifact, tmp_path)
    calls = stub_local_reload(monkeypatch)

    with pytest.raises(PromotionError, match="test accuracy"):
        promote_artifact(artifact, api_integration_validated=True, validation_report=report)
    assert calls == []


@pytest.mark.parametrize("change", ["duplicate_id", "invalid_label", "unhashable_label", "nan_probability", "confidence", "missing_prediction"])
def test_prediction_evidence_is_validated(tmp_path, change):
    artifact = write_full_artifact(tmp_path)
    path = artifact / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if change == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif change == "invalid_label":
        rows[0]["gold_label"] = "unknown"
    elif change == "unhashable_label":
        rows[0]["gold_label"] = []
    elif change == "nan_probability":
        rows[0]["class_probabilities"]["negative"] = float("nan")
    elif change == "confidence":
        rows[0]["confidence"] = 0.1
    else:
        rows.pop()
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    refresh_integrity(artifact)
    report = write_validation_report(artifact, tmp_path)

    with pytest.raises(PromotionError):
        promote_artifact(artifact, api_integration_validated=True, validation_report=report)


@pytest.mark.parametrize("change", ["stale_manifest", "different_artifact", "smoke_report", "failed_batch", "missing_exports"])
def test_promotion_rejects_unbound_or_failed_api_report(tmp_path, monkeypatch, change):
    artifact = write_full_artifact(tmp_path)
    report_path = write_validation_report(artifact, tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if change == "stale_manifest":
        report["artifact"]["manifest_sha256"] = "0" * 64
    elif change == "different_artifact":
        report["artifact_dir"] = str(tmp_path / "other-artifact")
    elif change == "smoke_report":
        report["artifact"]["smoke_test"] = True
    elif change == "failed_batch":
        report["application"]["batch"]["successful_documents"] = 3
    else:
        del report["application"]["exports"]
    _write_json(report_path, report)
    calls = stub_local_reload(monkeypatch)

    with pytest.raises(PromotionError):
        promote_artifact(artifact, api_integration_validated=True, validation_report=report_path)
    assert calls == []


def test_local_reload_failure_does_not_promote(tmp_path, monkeypatch):
    artifact = write_full_artifact(tmp_path)
    report = write_validation_report(artifact, tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("fixture load failed")
    loader = SimpleNamespace(from_pretrained=fail)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=loader, AutoModelForSequenceClassification=loader,
    ))
    with pytest.raises(PromotionError, match="reload failed"):
        promote_artifact(artifact, api_integration_validated=True, validation_report=report)
    assert json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))["production_ready"] is False
