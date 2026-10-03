from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.enrichment.bertopic import BertopicTopicService
from backend.app.services.enrichment.keybert import KeyBERTKeywordService
from backend.app.services.enrichment.summary import ExtractiveSummaryService
from backend.app.services.interfaces import (
    KeywordResult,
    ServiceMetadata,
    SentimentResult,
    SummaryResult,
    TopicResult,
)
from backend.app.services.mocks import MockSentimentService
from backend.app.services.orchestrator import AnalysisOrchestrator
from ml.preprocessing import build_analysis_tokens


def settings(**overrides) -> Settings:
    values = {"max_text_length": 100_000, "allowed_origins": []}
    values.update(overrides)
    return Settings(**values)


class FakeKeyBERT:
    def __init__(self) -> None:
        self.calls = []

    def extract_keywords(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [("बॅटरी backup", 1.5), ("खराब", -0.2), ("phone", 0.42)]


def test_keybert_result_mapping_preserves_unicode_and_normalizes_scores() -> None:
    service = KeyBERTKeywordService(settings(keyword_backend="keybert"))
    fake = FakeKeyBERT()
    service._keybert = fake
    result = service.extract("हा phone चांगला नाही आणि battery खराब आहे")

    assert result == [
        KeywordResult(text="बॅटरी backup", score=1.0),
        KeywordResult(text="खराब", score=0.0),
        KeywordResult(text="phone", score=0.42),
    ]
    assert fake.calls[0][0].startswith("हा phone")
    assert fake.calls[0][1]["stop_words"] is None


def test_keybert_empty_text_is_safe() -> None:
    service = KeyBERTKeywordService(settings(keyword_backend="keybert"))
    assert service.extract("   ") == []


def test_keyword_candidates_keep_marathi_words_and_negation_intact() -> None:
    tokens = build_analysis_tokens("हा मोबाईल चांगला नाही। पण battery backup खराब आहे॥")
    assert tokens == ("हा", "मोबाईल", "चांगला", "नाही", "पण", "battery", "backup", "खराब", "आहे")
    assert "ईल" not in tokens and "गल" not in tokens and "आह" not in tokens


class FakeTopicModel:
    def __init__(self, topics, probabilities) -> None:
        self.topics = topics
        self.probabilities = probabilities

    def transform(self, documents):
        assert len(documents) == 1
        return self.topics, self.probabilities


def test_bertopic_assigned_topic_uses_saved_label_and_probability() -> None:
    service = BertopicTopicService(
        settings(topic_backend="bertopic", topic_model_path="unused")
    )
    service._model = FakeTopicModel([2], [[0.1, 0.2, 0.7]])
    service._topic_labels = {2: "सेवा, अनुभव"}
    result = service.classify("सेवेचा अनुभव चांगला आहे")

    assert result == TopicResult(id=2, label="सेवा, अनुभव", probability=0.7)


def test_bertopic_outlier_is_honestly_unassigned() -> None:
    service = BertopicTopicService(
        settings(topic_backend="bertopic", topic_model_path="unused")
    )
    service._model = FakeTopicModel([-1], [[1.0]])
    assert service.classify("एक वेगळा मजकूर") == TopicResult(None, None, None)


@pytest.mark.parametrize(
    ("compact", "outliers", "probabilities"),
    [
        (True, 1, [[0.1, 0.2, 0.3, 0.7]]),
        (False, 1, [[0.1, 0.2, 0.7]]),
        (True, 0, [[0.1, 0.2, 0.7]]),
        (True, 1, [0.7]),
    ],
)
def test_bertopic_score_matches_assigned_topic_in_compact_and_full_models(
    compact: bool, outliers: int, probabilities: list
) -> None:
    """Compact cosine rows include -1; HDBSCAN membership rows do not."""
    service = BertopicTopicService(settings(topic_backend="bertopic"))
    fake = FakeTopicModel([2], probabilities)
    fake._outliers = outliers
    fake.hdbscan_model = (
        type("BaseCluster", (), {"__module__": "bertopic.cluster._base"})()
        if compact
        else SimpleNamespace()
    )
    service._model = fake
    service._topic_labels = {2: "सेवा, अनुभव"}

    assert service.classify("सेवेचा अनुभव चांगला आहे") == TopicResult(
        id=2, label="सेवा, अनुभव", probability=0.7
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_bertopic_nonfinite_score_is_not_returned_to_api(score: float) -> None:
    assert BertopicTopicService._probability_for([[score]], 0) is None


def test_missing_topic_artifact_is_unavailable(tmp_path: Path) -> None:
    service = BertopicTopicService(
        settings(topic_backend="bertopic", topic_model_path=str(tmp_path / "missing"))
    )
    metadata = service.metadata()
    assert metadata.state == "unavailable"
    assert metadata.production_ready is False


def test_extractive_summary_is_deterministic_and_source_only() -> None:
    service = ExtractiveSummaryService(settings(summary_backend="extractive"))
    text = "बॅटरी खूप चांगली आहे. हा phone वेगवान आहे. बॅटरी दिवसभर टिकते."
    first = service.summarize(text)
    second = service.summarize(text)

    assert first == second
    assert first.provider == "extractive"
    assert first.text is not None
    assert all(sentence in text for sentence in first.text.split(" "))


def test_extractive_summary_returns_null_for_short_text_and_keeps_code_mixing() -> None:
    service = ExtractiveSummaryService(settings(summary_backend="extractive"))
    assert service.summarize("हा phone चांगला आहे") == SummaryResult(None, "extractive")
    result = service.summarize("हा phone चांगला आहे. पण battery backup खराब आहे.")
    assert result.text is not None
    assert "phone" in result.text or "battery" in result.text


class FailingOptional:
    def __init__(self, name: str) -> None:
        self.name = name

    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(self.name, "test", "cpu", "unavailable")

    def extract(self, text):
        raise RuntimeError("private keyword detail")

    def classify(self, text):
        raise RuntimeError("private topic detail")

    def summarize(self, text):
        raise RuntimeError("private summary detail")


def test_optional_failures_are_isolated_and_warnings_are_aggregated() -> None:
    orchestrator = AnalysisOrchestrator(
        settings(),
        MockSentimentService(),
        FailingOptional("KeyBERT"),
        FailingOptional("BERTopic"),
        FailingOptional("summary"),
    )
    response = orchestrator.analyze("हा phone चांगला आहे", "enrichment-failure-test")

    assert response.sentiment.label == "neutral"
    assert response.keywords == []
    assert response.topic.id is None
    assert response.topic.label is None
    assert response.summary.text is None
    assert response.summary.provider is None
    assert response.meta.warnings.count("Keyword enrichment unavailable.") == 1
    assert response.meta.warnings.count("Topic enrichment unavailable.") == 1
    assert response.meta.warnings.count("Summary enrichment unavailable.") == 1


def test_disabled_services_report_disabled_without_loading_optional_dependencies() -> None:
    client = TestClient(
        create_app(
            settings(
                keyword_backend="disabled",
                topic_backend="disabled",
                summary_backend="disabled",
            )
        )
    )
    ready = client.get("/ready").json()
    info = client.get("/v1/model-info").json()
    analysis = client.post("/v1/analyze", json={"text": "नमस्कार"}).json()

    assert ready["services"]["keywords"]["state"] == "disabled"
    assert ready["services"]["topics"]["state"] == "disabled"
    assert ready["services"]["summary"]["state"] == "disabled"
    assert info["keyword_service"]["backend"] == "disabled"
    assert analysis["keywords"] == []
    assert analysis["topic"] == {"id": None, "label": None, "probability": None}
    assert analysis["summary"] == {"text": None, "provider": None}
