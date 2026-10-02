"""Pydantic request and response contracts for the API scaffold."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Raw user text accepted by the future full analysis endpoint."""

    text: str = Field(min_length=1, max_length=100_000)


class HealthResponse(BaseModel):
    status: str
    service: str
    pipeline: str


class AnalysisPreview(BaseModel):
    """Deterministic preprocessing output used to validate API wiring."""

    model_text: str
    analysis_text: str
    analysis_tokens: list[str]
