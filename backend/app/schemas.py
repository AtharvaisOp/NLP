"""Pydantic request and response contracts for the MahaPulse API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SentimentLabel = Literal["positive", "negative", "neutral"]
ReadinessState = Literal[
    "ready", "mocked", "disabled", "not_loaded", "not_ready", "unavailable"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeRequest(StrictModel):
    """Raw user text accepted by the analysis endpoint."""

    text: str = Field(description="Marathi or Marathi-English text to analyze")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class HealthResponse(StrictModel):
    status: str
    service: str
    pipeline: str


class LanguageInfo(StrictModel):
    primary: str
    devanagari_ratio: float = Field(ge=0, le=1)
    latin_ratio: float = Field(ge=0, le=1)
    is_code_mixed: bool


class SentimentInfo(StrictModel):
    label: SentimentLabel
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[SentimentLabel, float]
    low_confidence: bool


class KeywordInfo(StrictModel):
    text: str
    score: float = Field(ge=0, le=1)


class TopicInfo(StrictModel):
    id: int | None = None
    label: str | None = None
    probability: float | None = Field(default=None, ge=0, le=1)


class SummaryInfo(StrictModel):
    text: str | None = None
    provider: str | None = None


class AnalysisMeta(StrictModel):
    model_version: str
    processing_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(StrictModel):
    request_id: str
    original_text: str
    model_text: str
    analysis_text: str
    language: LanguageInfo
    sentiment: SentimentInfo
    keywords: list[KeywordInfo]
    topic: TopicInfo
    summary: SummaryInfo
    meta: AnalysisMeta


class ServiceReadiness(StrictModel):
    state: ReadinessState
    detail: str
    smoke_test: bool | None = None
    production_ready: bool | None = None


class ReadyResponse(StrictModel):
    status: Literal["ready", "degraded"]
    services: dict[str, ServiceReadiness]


class ServiceInfo(StrictModel):
    name: str
    version: str
    device: str | None = None
    state: ReadinessState
    smoke_test: bool | None = None
    production_ready: bool | None = None
    base_model: str | None = None
    preprocessing_version: str | None = None
    backend: str | None = None
    provider: str | None = None
    embedding_model: str | None = None


class ModelInfoResponse(StrictModel):
    sentiment_model: ServiceInfo
    labels: list[SentimentLabel]
    keyword_service: ServiceInfo
    topic_service: ServiceInfo
    summary_service: ServiceInfo
    readiness: dict[str, ServiceReadiness]


class AnalysisPreview(StrictModel):
    """Deterministic preprocessing output retained from Phase 1."""

    model_text: str
    analysis_text: str
    analysis_tokens: list[str]


class ErrorBody(StrictModel):
    code: str
    message: str
    details: object | None = None


class ErrorResponse(StrictModel):
    error: ErrorBody
    request_id: str


class BatchAnalysisResponse(StrictModel):
    session_id: str
    status: Literal["completed", "partial", "failed"]
    total_documents: int = Field(ge=0)
    successful_documents: int = Field(ge=0)
    failed_documents: int = Field(ge=0)
    processing_ms: int = Field(ge=0)
    model_version: str


class PersistedDocumentResponse(StrictModel):
    id: str
    row_index: int = Field(ge=0)
    status: Literal["success", "failed"]
    original_text: str | None = None
    model_text: str | None = None
    analysis_text: str | None = None
    language: LanguageInfo | None = None
    sentiment: SentimentInfo | None = None
    keywords: list[KeywordInfo] = Field(default_factory=list)
    topic: TopicInfo = Field(default_factory=TopicInfo)
    summary: SummaryInfo = Field(default_factory=SummaryInfo)
    processing_ms: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class SessionInfo(StrictModel):
    id: str
    source_type: Literal["single", "batch"]
    filename: str | None = None
    created_at: str
    completed_at: str | None = None
    status: Literal["processing", "completed", "partial", "failed"]
    total_documents: int = Field(ge=0)
    successful_documents: int = Field(ge=0)
    failed_documents: int = Field(ge=0)
    model_version: str


class AnalysisSessionResponse(StrictModel):
    session: SessionInfo
    documents: list[PersistedDocumentResponse]
    total_documents: int = Field(ge=0)
    limit: int = Field(ge=0)
    offset: int = Field(ge=0)


class SentimentAggregate(StrictModel):
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class ConfidenceAnalytics(StrictModel):
    average: float | None = Field(default=None, ge=0, le=1)
    minimum: float | None = Field(default=None, ge=0, le=1)
    maximum: float | None = Field(default=None, ge=0, le=1)
    low_confidence_count: int = Field(ge=0)


class LanguageAnalytics(StrictModel):
    code_mixed_count: int = Field(ge=0)
    code_mixed_percentage: float = Field(ge=0, le=100)


class KeywordAggregate(StrictModel):
    text: str
    count: int = Field(ge=0)
    average_score: float = Field(ge=0, le=1)


class TopicAggregate(StrictModel):
    id: int | None = None
    label: str | None = None
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class AnalyticsResponse(StrictModel):
    session_id: str
    total: int = Field(ge=0)
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    sentiment: dict[str, SentimentAggregate]
    confidence: ConfidenceAnalytics
    language: LanguageAnalytics
    keywords: list[KeywordAggregate]
    topics: list[TopicAggregate]
    null_topic_count: int = Field(ge=0)
    summary: None = None
