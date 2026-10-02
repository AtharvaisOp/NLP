"""Application service that preserves the two-path NLP architecture."""

from __future__ import annotations

import logging
from time import perf_counter

from ml.preprocessing import PreparedText, preprocess_text

from ..config import Settings
from ..exceptions import InputValidationError, ServiceFailure
from ..logging_config import LOGGER_NAME
from ..schemas import (
    AnalysisPreview,
    AnalysisResponse,
    AnalysisMeta,
    KeywordInfo,
    LanguageInfo,
    ModelInfoResponse,
    ReadinessState,
    ReadyResponse,
    SentimentInfo,
    ServiceInfo,
    ServiceReadiness,
    SummaryInfo,
    TopicInfo,
)
from .interfaces import (
    KeywordService,
    SentimentService,
    SummaryService,
    TopicService,
)
from .mocks import create_mock_services


class AnalysisOrchestrator:
    """Coordinates preprocessing and injected enrichment services."""

    def __init__(
        self,
        settings: Settings,
        sentiment_service: SentimentService,
        keyword_service: KeywordService,
        topic_service: TopicService,
        summary_service: SummaryService,
    ) -> None:
        self.settings = settings
        self.sentiment_service = sentiment_service
        self.keyword_service = keyword_service
        self.topic_service = topic_service
        self.summary_service = summary_service

    @classmethod
    def with_mocks(cls, settings: Settings) -> "AnalysisOrchestrator":
        return cls(settings, *create_mock_services())

    def preview(self, text: str) -> AnalysisPreview:
        prepared = self._prepare(text)
        return AnalysisPreview(
            model_text=prepared.model_text,
            analysis_text=prepared.analysis_text,
            analysis_tokens=list(prepared.analysis_tokens),
        )

    def analyze(self, text: str, request_id: str) -> AnalysisResponse:
        started = perf_counter()
        prepared = self._prepare(text)
        try:
            sentiment = self.sentiment_service.predict(prepared.model_text)
            keywords = self.keyword_service.extract(prepared.analysis_text)
            topic = self.topic_service.classify(prepared.analysis_text)
            summary = self.summary_service.summarize(prepared.model_text)
        except ServiceFailure:
            raise
        except Exception as exc:
            logging.getLogger(LOGGER_NAME).exception(
                "analysis service failed",
                extra={
                    "event": "analysis.service_failed",
                    "fields": {"request_id": request_id},
                },
                exc_info=exc,
            )
            raise ServiceFailure("Analysis services are temporarily unavailable") from exc

        warnings = [
            "ML service outputs are deterministic mocks; no models are loaded."
        ]
        return AnalysisResponse(
            request_id=request_id,
            original_text=text,
            model_text=prepared.model_text,
            analysis_text=prepared.analysis_text,
            language=self._language_info(text),
            sentiment=SentimentInfo(
                label=sentiment.label,
                confidence=sentiment.confidence,
                probabilities=sentiment.probabilities,
                low_confidence=sentiment.confidence < self.settings.low_confidence_threshold,
            ),
            keywords=[KeywordInfo(text=item.text, score=item.score) for item in keywords],
            topic=TopicInfo(
                id=topic.id,
                label=topic.label,
                probability=topic.probability,
            ),
            summary=SummaryInfo(text=summary.text, provider=summary.provider),
            meta=AnalysisMeta(
                model_version=self.sentiment_service.metadata().version,
                processing_ms=max(0, round((perf_counter() - started) * 1000)),
                warnings=warnings,
            ),
        )

    def readiness(self) -> ReadyResponse:
        services = {
            "api": ServiceReadiness(state="ready", detail="API process is running"),
            "preprocessing": ServiceReadiness(
                state="ready", detail="Deterministic preprocessing is available"
            ),
            "sentiment": self._readiness(self.sentiment_service.metadata()),
            "keywords": self._readiness(self.keyword_service.metadata()),
            "topics": self._readiness(self.topic_service.metadata()),
            "summary": self._readiness(self.summary_service.metadata()),
        }
        status: ReadinessState = "ready"
        if any(item.state != "ready" for item in services.values()):
            status = "mocked"
        return ReadyResponse(status="ready" if status == "ready" else "degraded", services=services)

    def model_info(self) -> ModelInfoResponse:
        sentiment_metadata = self.sentiment_service.metadata()
        return ModelInfoResponse(
            sentiment_model=self._service_info(sentiment_metadata),
            labels=["positive", "negative", "neutral"],
            keyword_service=self._service_info(self.keyword_service.metadata()),
            topic_service=self._service_info(self.topic_service.metadata()),
            summary_service=self._service_info(self.summary_service.metadata()),
            readiness=self.readiness().services,
        )

    def _prepare(self, text: str) -> PreparedText:
        if len(text) > self.settings.max_text_length:
            raise InputValidationError(
                "text exceeds the configured maximum length",
                details={"max_text_length": self.settings.max_text_length},
            )
        if not text.strip():
            raise InputValidationError("text must not be empty or whitespace-only")
        return preprocess_text(text)

    @staticmethod
    def _readiness(metadata) -> ServiceReadiness:
        return ServiceReadiness(
            state=metadata.state,
            detail=(
                "Service is mocked and no model is loaded"
                if metadata.state == "mocked"
                else f"{metadata.name} service is {metadata.state}"
            ),
        )

    @staticmethod
    def _service_info(metadata) -> ServiceInfo:
        return ServiceInfo(
            name=metadata.name,
            version=metadata.version,
            device=metadata.device,
            state=metadata.state,
        )

    @staticmethod
    def _language_info(text: str) -> LanguageInfo:
        letters = [character for character in text if character.isalpha()]
        devanagari = sum("\u0900" <= character <= "\u097f" for character in letters)
        latin = sum(("A" <= character <= "Z") or ("a" <= character <= "z") for character in letters)
        denominator = max(1, len(letters))
        devanagari_ratio = devanagari / denominator
        latin_ratio = latin / denominator
        return LanguageInfo(
            primary="mr",
            devanagari_ratio=round(devanagari_ratio, 6),
            latin_ratio=round(latin_ratio, 6),
            is_code_mixed=devanagari > 0 and latin > 0,
        )
