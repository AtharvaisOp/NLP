"""HTTP routes kept deliberately thin around the ML boundary."""

from fastapi import APIRouter

from ml.preprocessing import preprocess_text

from ..schemas import AnalysisPreview, AnalyzeRequest, HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Return a lightweight liveness response for Render health checks."""

    return HealthResponse(status="ok", service="mahapulse-api", pipeline="scaffold")


@router.post(
    "/api/v1/analyze/preview",
    response_model=AnalysisPreview,
    tags=["analysis"],
)
def analyze_preview(payload: AnalyzeRequest) -> AnalysisPreview:
    """Expose the two text contracts before model inference is integrated.

    This endpoint intentionally stops after deterministic preprocessing. The
    production analyze endpoint will add MuRIL, KeyBERT, BERTopic,
    summarization, persistence, and analytics behind this boundary.
    """

    prepared = preprocess_text(payload.text)
    return AnalysisPreview(
        model_text=prepared.model_text,
        analysis_text=prepared.analysis_text,
        analysis_tokens=prepared.analysis_tokens,
    )
