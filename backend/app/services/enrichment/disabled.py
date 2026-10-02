"""Explicitly disabled optional enrichment services."""

from __future__ import annotations

from ..interfaces import (
    KeywordResult,
    ServiceMetadata,
    SummaryResult,
    TopicResult,
)


class DisabledKeywordService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="KeyBERT",
            version="disabled",
            device="not-loaded",
            state="disabled",
            backend="disabled",
            provider="KeyBERT",
        )

    def extract(self, analysis_text: str) -> list[KeywordResult]:
        del analysis_text
        return []


class DisabledTopicService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="BERTopic",
            version="disabled",
            device="not-loaded",
            state="disabled",
            backend="disabled",
            provider="BERTopic",
        )

    def classify(self, analysis_text: str) -> TopicResult:
        del analysis_text
        return TopicResult(id=None, label=None, probability=None)


class DisabledSummaryService:
    def metadata(self) -> ServiceMetadata:
        return ServiceMetadata(
            name="summarization",
            version="disabled",
            device="not-loaded",
            state="disabled",
            backend="disabled",
        )

    def summarize(self, model_text: str) -> SummaryResult:
        del model_text
        return SummaryResult(text=None, provider=None)
