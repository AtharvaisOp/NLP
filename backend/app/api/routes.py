"""Thin HTTP routes over the injected analysis orchestrator."""

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_orchestrator
from ..schemas import (
    AnalysisPreview,
    AnalysisResponse,
    AnalyzeRequest,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
)
from ..services.orchestrator import AnalysisOrchestrator


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Return a lightweight liveness response with no model dependency."""

        return HealthResponse(status="ok", service="mahapulse-api", pipeline="phase-2")

    @router.get("/ready", response_model=ReadyResponse, tags=["operations"])
    def ready(
        orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    ) -> ReadyResponse:
        return orchestrator.readiness()

    @router.get("/v1/model-info", response_model=ModelInfoResponse, tags=["operations"])
    def model_info(
        orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    ) -> ModelInfoResponse:
        return orchestrator.model_info()

    @router.post("/v1/analyze", response_model=AnalysisResponse, tags=["analysis"])
    def analyze(
        payload: AnalyzeRequest,
        request: Request,
        orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    ) -> AnalysisResponse:
        return orchestrator.analyze(payload.text, request.state.request_id)

    @router.post(
        "/api/v1/analyze/preview",
        response_model=AnalysisPreview,
        tags=["analysis"],
    )
    def analyze_preview(
        payload: AnalyzeRequest,
        orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    ) -> AnalysisPreview:
        """Keep the Phase 1 preprocessing-preview endpoint compatible."""

        return orchestrator.preview(payload.text)

    return router
