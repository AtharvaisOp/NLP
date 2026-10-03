import json
from pathlib import Path

import pytest

from ml import evaluate
from ml.tests.artifact_fixtures import write_full_artifact
from ml.train import TrainingError, _write_json


def forbid_heavy_dependencies(monkeypatch):
    def forbidden():
        pytest.fail("rejected evaluation must not import a model runtime")
    monkeypatch.setattr(evaluate, "_require_training_dependencies", forbidden)


@pytest.mark.parametrize("evidence", ["metrics", "summary", "lifecycle", "provenance", "predictions"])
def test_final_test_evaluation_cannot_be_repeated(tmp_path: Path, monkeypatch, evidence):
    artifact = write_full_artifact(tmp_path)
    manifest_path = artifact / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    if evidence != "metrics":
        del metrics["test"]
    if evidence != "summary":
        manifest["evaluation_summary"] = None
    if evidence != "lifecycle":
        manifest["lifecycle_validation"]["held_out_test_evaluated"] = False
    if evidence != "provenance":
        del manifest["final_test_evaluation"]
    if evidence != "predictions":
        (artifact / "predictions.jsonl").unlink()
    _write_json(manifest_path, manifest)
    _write_json(artifact / "metrics.json", metrics)
    forbid_heavy_dependencies(monkeypatch)

    with pytest.raises(TrainingError, match="cannot be repeated"):
        evaluate.evaluate_artifact(artifact, tmp_path / "unread-test-data")


def test_training_selection_must_finish_before_evaluation(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_json(artifact / "model_manifest.json", {
        "smoke_test": False, "evaluation_summary": None,
        "lifecycle_validation": {"training_completed": False},
    })
    forbid_heavy_dependencies(monkeypatch)
    with pytest.raises(TrainingError, match="must finish first"):
        evaluate.evaluate_artifact(artifact, tmp_path / "unread-test-data")
