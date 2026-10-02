"""HTTP middleware for request correlation and structured access logs."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging_config import LOGGER_NAME


def _request_id(header_value: str | None) -> str:
    candidate = (header_value or "").strip()
    if candidate and len(candidate) <= 128 and all(
        character.isalnum() or character in "-_." for character in candidate
    ):
        return candidate
    return str(uuid4())


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = _request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logging.getLogger(LOGGER_NAME).info(
            "request completed",
            extra={
                "event": "request.completed",
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_ms": round((perf_counter() - started) * 1000, 2),
                },
            },
        )
        return response
