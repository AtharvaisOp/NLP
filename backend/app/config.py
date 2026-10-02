"""Environment-backed API configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    max_text_length: int = Field(default=100_000, gt=0)
    allowed_origins: list[str] = Field(default_factory=list)
    low_confidence_threshold: float = Field(default=0.60, ge=0, le=1)
    sentiment_backend: Literal["mock", "muril"] = "mock"
    sentiment_model_path: str | None = None
    allow_smoke_model: bool = False
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    keyword_backend: Literal["mock", "keybert", "disabled"] = "mock"
    keyword_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    keyword_top_n: int = Field(default=5, ge=1, le=50)
    keyword_ngram_min: int = Field(default=1, ge=1, le=5)
    keyword_ngram_max: int = Field(default=2, ge=1, le=5)
    keyword_use_mmr: bool = False
    keyword_diversity: float = Field(default=0.5, ge=0, le=1)
    keyword_timeout_seconds: float = Field(default=2.0, gt=0)
    topic_backend: Literal["mock", "bertopic", "disabled"] = "mock"
    topic_model_path: str | None = None
    topic_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    topic_timeout_seconds: float = Field(default=2.0, gt=0)
    summary_backend: Literal["mock", "extractive", "disabled"] = "mock"
    summary_max_sentences: int = Field(default=2, ge=1, le=5)
    summary_timeout_seconds: float = Field(default=1.0, gt=0)

    @field_validator("keyword_ngram_max")
    @classmethod
    def ngram_max_must_cover_min(cls, value: int, info) -> int:
        minimum = info.data.get("keyword_ngram_min", 1)
        if value < minimum:
            raise ValueError("keyword_ngram_max must be >= keyword_ngram_min")
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
