"""Security and deployment regressions without downloading any models."""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.logging_config import JSONFormatter
from backend.app.main import create_app
from backend.app.middleware import RequestBodyLimitMiddleware
from backend.app.storage.persistence import _safe_csv_cell


def test_environment_origins_and_render_postgresql_urls(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://mahapulse.vercel.app,http://localhost:5500/")
    monkeypatch.setenv("DATABASE_URL", "postgres://example:placeholder@localhost/mahapulse")
    settings = Settings(_env_file=None)
    assert settings.allowed_origins == ["https://mahapulse.vercel.app", "http://localhost:5500"]
    assert settings.database_url.startswith("postgresql+psycopg://")
    monkeypatch.setenv("ALLOWED_ORIGINS", '["https://mahapulse.vercel.app"]')
    assert Settings(_env_file=None).allowed_origins == ["https://mahapulse.vercel.app"]
    with pytest.raises(ValidationError, match="explicit frontend origins"):
        Settings(allowed_origins=["*"])


def test_configured_database_requires_migrated_schema(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'unmigrated.db'}")
    with TestClient(create_app(settings)) as client:
        result = client.get("/ready").json()
    assert result["services"]["database"]["state"] == "unavailable"
    assert "Alembic" in result["services"]["database"]["detail"]


def test_oversized_http_body_rejected_before_json_parser() -> None:
    with TestClient(create_app(Settings(max_text_length=5))) as client:
        response = client.post(
            "/v1/analyze",
            content=b"{" + b"a" * 17_000,
            headers={"Content-Type": "application/json", "X-Request-ID": "oversized-input"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert response.json()["request_id"] == "oversized-input"


def test_chunked_upload_limit_does_not_rely_on_content_length() -> None:
    called = False
    output = []
    messages = iter([
        {"type": "http.request", "body": b"a" * 40_000, "more_body": True},
        {"type": "http.request", "body": b"a" * 40_000, "more_body": False},
    ])

    async def application(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return next(messages)

    async def send(message):
        output.append(message)

    middleware = RequestBodyLimitMiddleware(application, Settings(max_upload_bytes=10))
    asyncio.run(middleware(
        {"type": "http", "method": "POST", "path": "/v1/analyze/batch", "headers": []},
        receive,
        send,
    ))
    assert called is False
    assert output[0]["status"] == 413


@pytest.mark.parametrize("value", ["=SUM(A1:A2)", " \t=1+1", "\ttext", "\n@SUM(1)", "\ufeff+1+1"])
def test_csv_export_protects_formula_prefixes_and_control_characters(value) -> None:
    assert _safe_csv_cell(value) == "'" + value
    assert _safe_csv_cell("हा मोबाईल चांगला आहे") == "हा मोबाईल चांगला आहे"


def test_exception_logging_does_not_store_user_text_or_sql_parameters() -> None:
    try:
        raise RuntimeError("PRIVATE Marathi user text; SQL parameters={'password':'secret'}")
    except RuntimeError:
        import sys
        record = logging.LogRecord("mahapulse", logging.ERROR, __file__, 1, "operation failed", (), sys.exc_info())
    output = JSONFormatter().format(record)
    assert "RuntimeError" in output
    assert "PRIVATE" not in output
    assert "secret" not in output
    assert "Traceback" not in output
