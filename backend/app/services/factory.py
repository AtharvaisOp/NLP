"""Application service assembly without model-specific route coupling."""

from __future__ import annotations

from .interfaces import KeywordService, SentimentService, SummaryService, TopicService
from .mocks import create_mock_services
from .orchestrator import AnalysisOrchestrator


def create_orchestrator(settings) -> AnalysisOrchestrator:
    mock_sentiment, _, _, _ = create_mock_services()
    sentiment: SentimentService = mock_sentiment
    if settings.sentiment_backend == "muril":
        from .sentiment import MurilSentimentService

        sentiment = MurilSentimentService(settings)
    if settings.keyword_backend == "keybert":
        from .enrichment.keybert import KeyBERTKeywordService

        keyword = KeyBERTKeywordService(settings)
    elif settings.keyword_backend == "disabled":
        from .enrichment.disabled import DisabledKeywordService

        keyword = DisabledKeywordService()
    else:
        from .mocks import MockKeywordService

        keyword = MockKeywordService()
    if settings.topic_backend == "bertopic":
        from .enrichment.bertopic import BertopicTopicService

        topic = BertopicTopicService(settings)
    elif settings.topic_backend == "disabled":
        from .enrichment.disabled import DisabledTopicService

        topic = DisabledTopicService()
    else:
        from .mocks import MockTopicService

        topic = MockTopicService()
    if settings.summary_backend == "extractive":
        from .enrichment.summary import ExtractiveSummaryService

        summary = ExtractiveSummaryService(settings)
    elif settings.summary_backend == "disabled":
        from .enrichment.disabled import DisabledSummaryService

        summary = DisabledSummaryService()
    else:
        from .mocks import MockSummaryService

        summary = MockSummaryService()
    return AnalysisOrchestrator(settings, sentiment, keyword, topic, summary)
