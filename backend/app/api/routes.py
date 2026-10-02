"""Thin HTTP routes over the injected analysis orchestrator."""

from typing import Literal

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from ..dependencies import get_orchestrator, get_persistence
from ..schemas import (
    AnalysisPreview,
    AnalysisResponse,
    AnalysisSessionResponse,
    AnalyzeRequest,
    AnalyticsResponse,
    BatchAnalysisResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    ServiceReadiness,
)
from ..exceptions import InputValidationError, ServiceFailure
from ..services.orchestrator import AnalysisOrchestrator
from ..storage.batch import BatchProcessor
from ..storage.persistence import PersistenceService, StorageError


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Return a lightweight liveness response with no model dependency."""

        return HealthResponse(status="ok", service="mahapulse-api", pipeline="phase-2")

    @router.get("/ready", response_model=ReadyResponse, tags=["operations"])
    def ready(
        orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
        persistence: PersistenceService = Depends(get_persistence),
    ) -> ReadyResponse:
        response = orchestrator.readiness()
        database_state, database_detail = persistence.database.readiness()
        response.services["database"] = ServiceReadiness(
            state=database_state,
            detail=database_detail,
        )
        if database_state == "unavailable":
            response.status = "degraded"
        return response

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
        response = orchestrator.analyze(payload.text, request.state.request_id)
        if request.app.state.settings.persist_single_analysis:
            persistence: PersistenceService = request.app.state.persistence
            try:
                session_id = persistence.create_session(
                    orchestrator, source_type="single", filename=None
                )
                persistence.persist_success(
                    session_id,
                    row_index=0,
                    original_text=payload.text,
                    response=response,
                    orchestrator=orchestrator,
                )
                persistence.finish_session(session_id, total=1, successful=1, failed=0)
            except StorageError as exc:
                raise ServiceFailure("Analysis persistence is temporarily unavailable") from exc
        return response

    @router.post(
        "/v1/analyze/batch",
        response_model=BatchAnalysisResponse,
        tags=["analysis"],
    )
    async def analyze_batch(
        request: Request,
        file: UploadFile = File(...),
        text_column: str | None = Query(default=None),
        orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
        persistence: PersistenceService = Depends(get_persistence),
    ) -> BatchAnalysisResponse:
        settings = request.app.state.settings
        payload = await file.read(settings.max_upload_bytes + 1)
        try:
            result = await run_in_threadpool(
                BatchProcessor(settings, persistence).process,
                payload,
                filename=file.filename,
                text_column=text_column,
                orchestrator=orchestrator,
            )
        except InputValidationError:
            raise
        except StorageError as exc:
            raise ServiceFailure("Batch persistence is temporarily unavailable") from exc
        return BatchAnalysisResponse(**result)

    @router.get(
        "/v1/analyses/{session_id}",
        response_model=AnalysisSessionResponse,
        tags=["analysis"],
    )
    def get_analysis_session(
        session_id: str,
        request: Request,
        limit: int = Query(default=50, ge=1, le=1_000),
        offset: int = Query(default=0, ge=0),
        persistence: PersistenceService = Depends(get_persistence),
    ) -> AnalysisSessionResponse:
        if limit > request.app.state.settings.max_pagination_limit:
            raise InputValidationError(
                "limit exceeds the configured pagination maximum",
                details={"max_pagination_limit": request.app.state.settings.max_pagination_limit},
            )
        try:
            return persistence.get_session_response(session_id, limit=limit, offset=offset)
        except StorageError as exc:
            raise ServiceFailure("Analysis retrieval is temporarily unavailable") from exc

    @router.get(
        "/v1/analyses/{session_id}/analytics",
        response_model=AnalyticsResponse,
        tags=["analytics"],
    )
    def get_analytics(
        session_id: str,
        persistence: PersistenceService = Depends(get_persistence),
    ) -> AnalyticsResponse:
        try:
            return persistence.analytics(session_id)
        except StorageError as exc:
            raise ServiceFailure("Analysis analytics are temporarily unavailable") from exc

    @router.get("/v1/analyses/{session_id}/export", tags=["analytics"])
    def export_analysis(
        session_id: str,
        format: Literal["csv", "json"] = Query(default="json"),
        persistence: PersistenceService = Depends(get_persistence),
    ) -> Response:
        try:
            content, media_type, filename = persistence.export(session_id, format)
        except StorageError as exc:
            raise ServiceFailure("Analysis export is temporarily unavailable") from exc
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

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
