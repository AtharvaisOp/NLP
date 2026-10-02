import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.mocks import (
    MockKeywordService,
    MockSummaryService,
    MockTopicService,
)
from backend.app.services.orchestrator import AnalysisOrchestrator
from backend.app.services.sentiment.muril import (
    MurilSentimentService,
    MurilServiceError,
)
from ml.preprocessing import preprocess_text


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": FakeTensor([[1, 2]]), "attention_mask": FakeTensor([[1, 1]])}


class FakeModel:
    def __init__(self, logits=None):
        self.logits = logits or [[0.1, 2.0, 0.2]]
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True

    def __call__(self, **encoded):
        del encoded
        return SimpleNamespace(logits=FakeTensor(self.logits))


class FailingModel(FakeModel):
    def __call__(self, **encoded):
        del encoded
        raise RuntimeError("private model internals")


def make_artifact(tmp_path: Path, *, smoke_test: bool = True) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {
        "project": "MahaPulse",
        "task": "sentiment classification",
        "architecture": "AutoModelForSequenceClassification",
        "base_model": "google/muril-base-cased",
        "model_name": "google/muril-base-cased",
        "model_version": "test-artifact-v1",
        "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
        "preprocessing_version": "model-text-v1",
        "max_length": 256,
        "smoke_test": smoke_test,
        "artifact_integrity": {},
    }
    (artifact / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifact / "label_mapping.json").write_text(
        json.dumps({"canonical_to_id": manifest["label_mapping"]}), encoding="utf-8"
    )
    (artifact / "config.json").write_text(json.dumps({
        "label2id": manifest["label_mapping"],
        "id2label": {str(value): key for key, value in manifest["label_mapping"].items()},
    }), encoding="utf-8")
    for filename in ("tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (artifact / filename).write_text("{}", encoding="utf-8")
    return artifact


def make_service(artifact: Path, *, allow_smoke: bool = True) -> MurilSentimentService:
    settings = Settings(
        sentiment_backend="muril",
        sentiment_model_path=str(artifact),
        allow_smoke_model=allow_smoke,
        model_device="auto",
        max_text_length=100_000,
        allowed_origins=[],
    )
    return MurilSentimentService(settings)


def fake_components(service: MurilSentimentService, model=None):
    tokenizer = FakeTokenizer()
    service._load_components = lambda: (tokenizer, model or FakeModel(), FakeTorch(), "cpu")
    return tokenizer


def test_smoke_artifact_is_rejected_without_explicit_opt_in(tmp_path: Path) -> None:
    service = make_service(make_artifact(tmp_path), allow_smoke=False)

    metadata = service.metadata()

    assert metadata.state == "not_ready"
    assert metadata.smoke_test is True
    assert metadata.production_ready is False


def test_smoke_artifact_is_accepted_with_explicit_opt_in(tmp_path: Path) -> None:
    service = make_service(make_artifact(tmp_path), allow_smoke=True)
    tokenizer = fake_components(service)

    result = service.predict("हे product चांगले आहे!!!")
    metadata = service.metadata()

    assert metadata.state == "ready"
    assert metadata.smoke_test is True
    assert metadata.production_ready is False
    assert result.label == "neutral"
    assert set(result.probabilities) == {"positive", "negative", "neutral"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert tokenizer.calls[0][0] == preprocess_text("हे product चांगले आहे!!!").model_text
    assert tokenizer.calls[0][1]["truncation"] is True
    assert tokenizer.calls[0][1]["max_length"] == 256


def test_device_selection_auto_cpu_and_explicit_cuda_failure() -> None:
    assert MurilSentimentService._select_device(FakeTorch, "auto") == "cpu"
    with pytest.raises(MurilServiceError, match="CUDA was explicitly requested"):
        MurilSentimentService._select_device(FakeTorch, "cuda")


def test_inference_failure_is_not_silently_mocked(tmp_path: Path) -> None:
    service = make_service(make_artifact(tmp_path))
    fake_components(service, FailingModel())

    with pytest.raises(MurilServiceError, match="inference failed"):
        service.predict("सेवा खराब आहे")


def test_ready_model_info_and_analyze_expose_smoke_state_without_schema_change(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    service = make_service(artifact)
    fake_components(service)
    settings = Settings(
        sentiment_backend="muril",
        sentiment_model_path=str(artifact),
        allow_smoke_model=True,
        model_device="auto",
        max_text_length=100_000,
        allowed_origins=[],
    )
    orchestrator = AnalysisOrchestrator(
        settings,
        service,
        MockKeywordService(),
        MockTopicService(),
        MockSummaryService(),
    )
    client = TestClient(create_app(settings=settings, orchestrator=orchestrator))

    ready = client.get("/ready")
    info = client.get("/v1/model-info")
    analysis = client.post("/v1/analyze", json={"text": "हा phone चांगला आहे"})

    assert ready.status_code == 200
    assert ready.json()["status"] == "degraded"
    assert ready.json()["services"]["sentiment"] == {
        "state": "ready",
        "detail": "Smoke artifact is operational for development; not production-ready",
        "smoke_test": True,
        "production_ready": False,
    }
    assert info.status_code == 200
    assert info.json()["sentiment_model"]["version"] == "test-artifact-v1"
    assert info.json()["sentiment_model"]["smoke_test"] is True
    assert info.json()["sentiment_model"]["production_ready"] is False
    assert analysis.status_code == 200
    body = analysis.json()
    assert set(body) == {
        "request_id", "original_text", "model_text", "analysis_text", "language",
        "sentiment", "keywords", "topic", "summary", "meta",
    }
    assert body["meta"]["model_version"] == "test-artifact-v1"
    assert body["meta"]["warnings"]


def test_low_confidence_uses_configured_threshold(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    service = make_service(artifact)
    fake_components(service, FakeModel([[0.0, 0.0, 0.0]]))
    settings = Settings(
        sentiment_backend="muril",
        sentiment_model_path=str(artifact),
        allow_smoke_model=True,
        model_device="auto",
        low_confidence_threshold=0.60,
        max_text_length=100_000,
        allowed_origins=[],
    )
    orchestrator = AnalysisOrchestrator(
        settings, service, MockKeywordService(), MockTopicService(), MockSummaryService()
    )

    response = TestClient(create_app(settings=settings, orchestrator=orchestrator)).post(
        "/v1/analyze", json={"text": "साधारण माहिती"}
    )

    assert response.status_code == 200
    assert response.json()["sentiment"]["low_confidence"] is True


def test_invalid_artifact_is_not_ready(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    manifest["label_mapping"] = {"negative": 0, "neutral": 2, "positive": 1}
    (artifact / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    service = make_service(artifact)

    assert service.metadata().state == "not_ready"


def test_artifact_rejects_classifier_config_with_swapped_labels(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    config["label2id"] = {"negative": 2, "neutral": 1, "positive": 0}
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert make_service(artifact).metadata().state == "not_ready"


def test_artifact_integrity_paths_cannot_escape_artifact_directory(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    outside = tmp_path / "private.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = json.loads((artifact / "model_manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_integrity"] = {"../private.json": {"bytes": 2, "sha256": "irrelevant"}}
    (artifact / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert make_service(artifact).metadata().state == "not_ready"
