"""FastAPI dependency providers for application services."""

from fastapi import Request

from .services.orchestrator import AnalysisOrchestrator


def get_orchestrator(request: Request) -> AnalysisOrchestrator:
    return request.app.state.orchestrator
