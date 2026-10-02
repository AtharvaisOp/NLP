from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.interfaces import ServiceMetadata
from backend.app.services.mocks import (
    MockKeywordService,
    MockSummaryService,
    MockTopicService,
    MockSentimentService,
)
from backend.app.services.orchestrator import AnalysisOrchestrator


def make_client(*, max_text_length: int = 100_000, orchestrator=None) -> TestClient:
    settings = Settings(
        max_text_length=max_text_length,
        allowed_origins=[],
        low_confidence_threshold=0.60,
    )
    return TestClient(create_app(settings=settings, orchestrator=orchestrator))


def test_health_is_lightweight_and_alive() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "mahapulse-api",
        "pipeline": "phase-2",
    }


def test_ready_reports_api_preprocessing_and_mocked_models() -> None:
    response = make_client().get("/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["services"]["api"]["state"] == "ready"
    assert body["services"]["preprocessing"]["state"] == "ready"
    assert body["services"]["sentiment"]["state"] == "mocked"


def test_model_info_is_structured_and_does_not_claim_loaded_models() -> None:
    response = make_client().get("/v1/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["sentiment_model"]["name"] == "MuRIL"
    assert body["sentiment_model"]["version"] == "mock-v0"
    assert body["sentiment_model"]["state"] == "mocked"
    assert body["labels"] == ["positive", "negative", "neutral"]
    assert body["keyword_service"]["name"] == "KeyBERT"
    assert body["topic_service"]["name"] == "BERTopic"
    assert body["summary_service"]["state"] == "mocked"


def test_valid_marathi_analysis_has_stable_response_contract() -> None:
    response = make_client().post("/v1/analyze", json={"text": "हे उत्पादन चांगले आहे!"})
    body = response.json()

    assert response.status_code == 200
    assert body["original_text"] == "हे उत्पादन चांगले आहे!"
    assert body["language"]["primary"] == "mr"
    assert body["sentiment"]["label"] in {"positive", "negative", "neutral"}
    assert body["meta"]["warnings"]


def test_code_mixed_analysis_reports_both_scripts() -> None:
    response = make_client().post(
        "/v1/analyze", json={"text": "हा product चांगला नाही!!!"}
    )
    body = response.json()

    assert response.status_code == 200
    assert body["language"]["is_code_mixed"] is True
    assert body["language"]["devanagari_ratio"] > 0
    assert body["language"]["latin_ratio"] > 0


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_or_whitespace_only_input_is_rejected(text: str) -> None:
    response = make_client().post("/v1/analyze", json={"text": text})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert response.json()["request_id"]


def test_oversized_input_is_rejected_using_configured_limit() -> None:
    response = make_client(max_text_length=5).post(
        "/v1/analyze", json={"text": "123456"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert response.json()["error"]["details"]["max_text_length"] == 5


def test_response_schema_and_probabilities_are_valid() -> None:
    response = make_client().post("/v1/analyze", json={"text": "डेटा छान आहे"})
    body = response.json()

    assert response.status_code == 200
    expected_keys = {
        "request_id",
        "original_text",
        "model_text",
        "analysis_text",
        "language",
        "sentiment",
        "keywords",
        "topic",
        "summary",
        "meta",
    }
    assert set(body) == expected_keys
    assert sum(body["sentiment"]["probabilities"].values()) == pytest.approx(1.0)
    assert body["meta"]["processing_ms"] >= 0


def test_request_id_is_present_and_echoed() -> None:
    response = make_client().post(
        "/v1/analyze",
        headers={"X-Request-ID": "contract-test-123"},
        json={"text": "नमस्कार"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "contract-test-123"
    assert response.headers["X-Request-ID"] == "contract-test-123"


def test_model_text_and_analysis_text_remain_separate() -> None:
    response = make_client().post(
        "/v1/analyze",
        json={"text": "हे product चांगले नाही!!! #review @user"},
    )
    body = response.json()

    assert "!!!" in body["model_text"]
    assert "#review" in body["model_text"]
    assert "<USER>" in body["model_text"]
    assert "नाही" in body["model_text"]
    assert "!" not in body["analysis_text"]
    assert "#" not in body["analysis_text"]
    assert "<USER>" not in body["analysis_text"]
    assert "नाही" in body["analysis_text"]


class FailingSentimentService(MockSentimentService):
    def metadata(self) -> ServiceMetadata:
        metadata = super().metadata()
        return ServiceMetadata(
            name=metadata.name,
            version=metadata.version,
            device=metadata.device,
            state=metadata.state,
        )

    def predict(self, model_text: str):
        del model_text
        raise RuntimeError("private model failure details")


def test_service_failure_is_safe_and_contains_no_stack_trace() -> None:
    settings = Settings(max_text_length=100_000, allowed_origins=[])
    orchestrator = AnalysisOrchestrator(
        settings,
        FailingSentimentService(),
        MockKeywordService(),
        MockTopicService(),
        MockSummaryService(),
    )
    response = TestClient(create_app(settings=settings, orchestrator=orchestrator)).post(
        "/v1/analyze", json={"text": "सेवा तपासणी"}
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "service_unavailable",
        "message": "Analysis services are temporarily unavailable",
        "details": None,
    }
    assert "private model failure details" not in response.text
    assert "Traceback" not in response.text
    assert response.json()["request_id"]
