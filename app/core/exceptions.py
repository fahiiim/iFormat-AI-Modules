"""Service exceptions and their standardized HTTP representations."""

from http import HTTPStatus
from typing import NoReturn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


class AIServiceException(Exception):
    """Base class for expected failures in an AI integration service."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: str = "AI_SERVICE_ERROR"
    public_message: str = "The AI service could not complete the request."

    def __init__(self, message: str | None = None) -> None:
        """Initialize the exception with a safe client-facing message.

        Args:
            message: Optional safe message that may be returned to an API
                client. Provider exception details should only be logged.
        """

        self.public_message = message or self.public_message
        super().__init__(self.public_message)


class BedrockInvocationException(AIServiceException):
    """Raised when Amazon Bedrock rejects or cannot execute a request."""

    error_code = "BEDROCK_INVOCATION_FAILED"
    public_message = "Amazon Bedrock could not complete the request."


class BedrockThrottlingException(BedrockInvocationException):
    """Raised when Amazon Bedrock throttles or is temporarily unavailable."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    error_code = "BEDROCK_THROTTLED"
    public_message = "Amazon Bedrock is temporarily unavailable. Please retry."


class BedrockUnavailableException(BedrockInvocationException):
    """Raised when Bedrock cannot be reached due to a transient AWS failure."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    error_code = "BEDROCK_UNAVAILABLE"
    public_message = "Amazon Bedrock is temporarily unavailable. Please retry."


class BedrockResponseException(AIServiceException):
    """Raised when Bedrock returns malformed or contract-invalid output."""

    error_code = "BEDROCK_INVALID_RESPONSE"
    public_message = "Amazon Bedrock returned an invalid response."


class RAGServiceException(AIServiceException):
    """Raised when the career-advisor retrieval pipeline fails."""

    error_code = "RAG_SERVICE_ERROR"
    public_message = "The career advisor could not complete the request."


class RAGNotReadyException(RAGServiceException):
    """Raised when the career-advisor knowledge base is unavailable."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    error_code = "RAG_NOT_READY"
    public_message = "The career advisor knowledge base is not available."


def _error_payload(exc: AIServiceException) -> dict[str, dict[str, str]]:
    """Build the stable JSON error envelope for a service exception."""

    return {
        "detail": {
            "code": exc.error_code,
            "message": exc.public_message,
        }
    }


def raise_http_exception_for_service_error(
    exc: AIServiceException,
) -> NoReturn:
    """Translate a service-layer exception into a FastAPI HTTP exception.

    Args:
        exc: The expected service error raised by a business service.

    Raises:
        HTTPException: Always raised with the service error's standard status
            and error detail.
    """

    headers = {"Retry-After": "2"} if exc.status_code == 503 else None
    raise HTTPException(
        status_code=exc.status_code,
        detail=_error_payload(exc)["detail"],
        headers=headers,
    ) from exc


async def ai_service_exception_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle uncaught service errors using the same stable API envelope."""

    if not isinstance(exc, AIServiceException):
        raise exc
    headers = {"Retry-After": "2"} if exc.status_code == 503 else None
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on a FastAPI application.

    Args:
        app: Application instance that should own the handlers.
    """

    app.add_exception_handler(AIServiceException, ai_service_exception_handler)
