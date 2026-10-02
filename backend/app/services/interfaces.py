"""Stable service contracts for future real model integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ServiceMetadata:
    name: str
    version: str
    device: str | None
    state: str
    smoke_test: bool | None = None
    production_ready: bool | None = None
    base_model: str | None = None
    preprocessing_version: str | None = None
    backend: str | None = None
    provider: str | None = None
    embedding_model: str | None = None


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class KeywordResult:
    text: str
    score: float


@dataclass(frozen=True)
class TopicResult:
    id: int | None
    label: str | None
    probability: float | None


@dataclass(frozen=True)
class SummaryResult:
    text: str | None
    provider: str | None


class SentimentService(Protocol):
    def metadata(self) -> ServiceMetadata: ...

    def predict(self, model_text: str) -> SentimentResult: ...


class KeywordService(Protocol):
    def metadata(self) -> ServiceMetadata: ...

    def extract(self, analysis_text: str) -> list[KeywordResult]: ...


class TopicService(Protocol):
    def metadata(self) -> ServiceMetadata: ...

    def classify(self, analysis_text: str) -> TopicResult: ...


class SummaryService(Protocol):
    def metadata(self) -> ServiceMetadata: ...

    def summarize(self, model_text: str) -> SummaryResult: ...
