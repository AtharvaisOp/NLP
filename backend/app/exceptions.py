"""Typed application errors exposed through safe JSON responses."""


class APIError(Exception):
    """Base error with a client-safe status, code, and message."""

    status_code = 500
    code = "api_error"

    def __init__(self, message: str, *, details: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InputValidationError(APIError):
    status_code = 422
    code = "invalid_input"


class ServiceFailure(APIError):
    status_code = 503
    code = "service_unavailable"
