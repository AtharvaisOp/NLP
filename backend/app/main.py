"""FastAPI application factory and production-shaped API assembly."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import create_router
from .config import Settings, get_settings
from .errors import register_exception_handlers
from .logging_config import configure_logging
from .middleware import RequestContextMiddleware
from .services.orchestrator import AnalysisOrchestrator


def create_app(
    settings: Settings | None = None,
    orchestrator: AnalysisOrchestrator | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    configure_logging()
    app = FastAPI(
        title="MahaPulse NLP API",
        description="Stable API boundary for Marathi and Marathi-English NLP analysis.",
        version="0.2.0",
    )
    app.state.settings = runtime_settings
    app.state.orchestrator = orchestrator or AnalysisOrchestrator.with_mocks(
        runtime_settings
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.include_router(create_router())
    register_exception_handlers(app)
    return app


app = create_app()
