"""Deterministic, explicitly non-production service implementations."""

from __future__ import annotations

from .demo_sentiment import DEMO_VERSION, demo_sentiment
from .interfaces import (
    KeywordResult,
    KeywordService,
    SentimentResult,
    SentimentService,
    ServiceMetadata,
    SummaryResult,
    SummaryService,
    TopicResult,
    TopicService,
)


MOCK_VERSION = "mock-v0"


class MockSentimentService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="Rule-based demo",
            version=DEMO_VERSION,
            device="cpu",
            state="mocked",
            smoke_test=False,
            production_ready=False,
            backend="mock",
            provider="lexicon",
        )

    def predict(self, model_text: str) -> SentimentResult:
        return demo_sentiment(model_text)


class MockKeywordService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="KeyBERT",
            version=MOCK_VERSION,
            device="not-loaded",
            state="mocked",
            backend="mock",
            provider="KeyBERT",
        )

    def extract(self, analysis_text: str) -> list[KeywordResult]:
        unique_tokens = list(dict.fromkeys(token for token in analysis_text.split() if token))
        return [
            KeywordResult(text=token, score=round(max(0.1, 1.0 - index * 0.2), 3))
            for index, token in enumerate(unique_tokens[:5])
        ]


class MockTopicService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="BERTopic",
            version=MOCK_VERSION,
            device="not-loaded",
            state="mocked",
            backend="mock",
            provider="BERTopic",
        )

    def classify(self, analysis_text: str) -> TopicResult:
        del analysis_text
        return TopicResult(id=None, label=None, probability=None)


class MockSummaryService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="summarization",
            version=MOCK_VERSION,
            device="not-loaded",
            state="mocked",
            backend="mock",
        )

    def summarize(self, model_text: str) -> SummaryResult:
        del model_text
        return SummaryResult(text=None, provider=None)


def create_mock_services() -> tuple[
    SentimentService, KeywordService, TopicService, SummaryService
]:
    """Build the default service bundle without loading any model artifacts."""

    return (
        MockSentimentService(),
        MockKeywordService(),
        MockTopicService(),
        MockSummaryService(),
    )
