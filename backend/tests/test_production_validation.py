"""Deterministic release-validator regressions; no real model loads or downloads."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.services.sentiment.muril import MurilSentimentService
from backend.tests.test_sentiment_service import FakeModel, FakeTensor, FakeTokenizer, FakeTorch, make_artifact


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate-production-model.py"
SPEC = importlib.util.spec_from_file_location("production_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def sentiment(probabilities=None, label="neutral", confidence=0.5):
    return {"label": label, "confidence": confidence,
            "probabilities": probabilities or {"negative": 0.2, "neutral": 0.5, "positive": 0.3}}


def test_validator_checks_finite_canonical_probability_and_confidence_contract() -> None:
    validator.validate_probabilities(sentiment())
    with pytest.raises(RuntimeError, match="noncanonical"):
        validator.validate_probabilities(sentiment({"NEG": 0.2, "neutral": 0.5, "positive": 0.3}))
    for probabilities in (
        {"negative": float("nan"), "neutral": 0.5, "positive": 0.3},
        {"negative": 1.2, "neutral": -0.5, "positive": 0.3},
        {"negative": True, "neutral": 0, "positive": 0},
    ):
        with pytest.raises(RuntimeError, match="finite"):
            validator.validate_probabilities(sentiment(probabilities))
    with pytest.raises(RuntimeError, match="sum"):
        validator.validate_probabilities(sentiment({"negative": 0.1, "neutral": 0.5, "positive": 0.3}))
    with pytest.raises(RuntimeError, match="label disagrees"):
        validator.validate_probabilities(sentiment(label="positive", confidence=0.3))
    with pytest.raises(RuntimeError, match="confidence disagrees"):
        validator.validate_probabilities(sentiment(confidence=0.9))


def test_validator_csv_retains_all_four_unicode_examples_and_one_empty_row() -> None:
    rows = list(csv.DictReader(io.StringIO(validator.batch_csv().decode("utf-8"))))
    assert [row["text"] for row in rows] == [text for _, text in validator.EXAMPLES] + [""]


def test_independent_reload_explicitly_uses_local_files_and_never_asserts_example_accuracy(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = make_artifact(tmp_path, smoke_test=False)
    loads = []

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loads.append((path, kwargs))
            model = FakeModel()
            model.config = SimpleNamespace(label2id=validator.LABEL_MAPPING, num_labels=3)
            return model

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loads.append((path, kwargs))
            return FakeTokenizer()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        inference_mode=nullcontext,
        softmax=lambda logits, dim: FakeTensor([[0.2, 0.5, 0.3]]),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForSequenceClassification=Loader, AutoTokenizer=TokenizerLoader))
    report = validator.independently_reload(artifact, "cpu")
    assert report["passed"] is True and report["local_files_only"] is True
    assert len(loads) == 2 and all(kwargs == {"local_files_only": True} for _, kwargs in loads)
    assert len(report["examples"]) == 4
    # Fixture predicts neutral on every example: the integration gate must not
    # turn a human's example heading into an accuracy claim or selection rule.
    assert {example["sentiment"]["label"] for example in report["examples"]} == {"neutral"}


def test_validator_migrates_sqlite_and_checks_entire_api_batch_export_flow(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = make_artifact(tmp_path, smoke_test=False)
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(MurilSentimentService, "_load_components",
                        lambda self: (tokenizer, FakeModel([[0, 0, 0]]), FakeTorch(), "cpu"))
    database = tmp_path / "validation.db"
    revision = validator.migrate_database(database)
    assert revision == "0001_initial_analysis_schema"
    settings = Settings(_env_file=None, sentiment_backend="muril", sentiment_model_path=str(artifact),
                        allow_smoke_model=False, model_device="cpu", keyword_backend="disabled",
                        topic_backend="disabled", summary_backend="extractive", store_raw_text=True,
                        persist_single_analysis=False, database_url="sqlite:///" + database.as_posix(),
                        allowed_origins=["http://localhost:5500"], max_text_length=1024)
    probabilities = {label: 1 / 3 for label in validator.LABEL_MAPPING}
    independent = {"examples": [{"model_text": text,
                                 "sentiment": {"probabilities": probabilities}}
                                for _, text in validator.EXAMPLES]}
    report = validator.validate_application(settings, tmp_path, independent, False, "disabled")
    assert report["passed"] is True
    assert report["operations"]["/v1/model-info"]["sentiment_model"]["production_ready"] is False
    assert report["batch"]["successful_documents"] == 4 and report["batch"]["failed_documents"] == 1
    assert report["exports"]["csv_rows"] == report["exports"]["json_documents"] == 5
    assert report["analytics"]["language"]["code_mixed_count"] == 1
    assert report["security"]["https://untrusted.invalid"] == 400
    assert report["security"]["request_body_limit"] == 413


def test_failed_validation_retains_readable_evidence_outside_artifact(tmp_path: Path, monkeypatch) -> None:
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(variable, "0")
    output = tmp_path / "evidence"
    assert validator.main(["--artifact-dir", str(tmp_path / "missing"), "--output-dir", str(output)]) == 1
    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["failure"]["stage"] == "artifact_integrity"
    assert report["offline"] is True and report["integration_examples_are_metrics"] is False


def test_validator_rejects_evidence_output_inside_immutable_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="outside the immutable"):
        validator.main(["--artifact-dir", str(artifact), "--output-dir", str(artifact / "evidence")])
    assert not artifact.exists()
