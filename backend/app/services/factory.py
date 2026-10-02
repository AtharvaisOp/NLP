"""Application service assembly without model-specific route coupling."""

from __future__ import annotations

from .interfaces import KeywordService, SentimentService, SummaryService, TopicService
from .mocks import create_mock_services
from .orchestrator import AnalysisOrchestrator


def create_orchestrator(settings) -> AnalysisOrchestrator:
    mock_sentiment, keyword, topic, summary = create_mock_services()
    sentiment: SentimentService = mock_sentiment
    if settings.sentiment_backend == "muril":
        from .sentiment import MurilSentimentService

        sentiment = MurilSentimentService(settings)
    return AnalysisOrchestrator(settings, sentiment, keyword, topic, summary)
