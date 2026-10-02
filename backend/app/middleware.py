"""HTTP middleware for request correlation and structured access logs."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings
from .logging_config import LOGGER_NAME


def _request_id(header_value: str | None) -> str:
    candidate = (header_value or "").strip()
    if candidate and len(candidate) <= 128 and all(
        character.isalnum() or character in "-_." for character in candidate
    ):
        return candidate
    return str(uuid4())


class RequestBodyLimitMiddleware:
    """Bound bodies before JSON or multipart parsers allocate/spool them."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path == "/v1/analyze/batch":
            maximum = self.settings.max_upload_bytes + 65_536
        elif path in {"/v1/analyze", "/api/v1/analyze/preview"}:
            # JSON can encode each Unicode character using six ASCII bytes.
            maximum = self.settings.max_text_length * 6 + 16_384
        else:
            await self.app(scope, receive, send)
            return

        async def reject() -> None:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "request_too_large",
                        "message": "Request body exceeds the configured maximum size",
                        "details": {"max_request_bytes": maximum},
                    },
                    "request_id": scope.get("state", {}).get("request_id", "unknown"),
                },
            )
            await response(scope, receive, send)

        headers = dict(scope.get("headers", []))
        try:
            declared_size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared_size = 0
        if declared_size > maximum:
            await reject()
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > maximum:
                await reject()
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


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
