"""Centralized safe exception handlers."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .exceptions import APIError
from .logging_config import LOGGER_NAME


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message, "details": details},
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        logging.getLogger(LOGGER_NAME).warning(
            "request failed",
            extra={
                "event": "request.failed",
                "fields": {"request_id": _request_id(request), "code": exc.code},
            },
        )
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "location": list(error.get("loc", ())),
                "message": error.get("msg", "invalid value"),
            }
            for error in exc.errors()
        ]
        return _error_response(
            request,
            status_code=422,
            code="request_validation_error",
            message="Request validation failed",
            details=details,
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(LOGGER_NAME).exception(
            "unhandled request failure",
            extra={
                "event": "request.unhandled_error",
                "fields": {"request_id": _request_id(request)},
            },
            exc_info=exc,
        )
        return _error_response(
            request,
            status_code=500,
            code="internal_error",
            message="An internal error occurred",
        )
