"""FastAPI dependency providers for application services."""

from fastapi import Request

from .services.orchestrator import AnalysisOrchestrator
from .storage.persistence import PersistenceService


def get_orchestrator(request: Request) -> AnalysisOrchestrator:
    return request.app.state.orchestrator


def get_persistence(request: Request) -> PersistenceService:
    return request.app.state.persistence
