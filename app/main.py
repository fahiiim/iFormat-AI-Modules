"""FastAPI application factory for the iFormat AI API."""

from typing import Literal, TypedDict

from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.exceptions import register_exception_handlers


class HealthResponse(TypedDict):
    """Public liveness response used by AWS App Runner."""

    status: Literal["ok"]


def create_application() -> FastAPI:
    """Create and configure the iFormat AI FastAPI application.

    Returns:
        FastAPI: Fully wired ASGI application.
    """

    application = FastAPI(
        title="iFormat Job Portal AI Services",
        description="AI screening, writing, recommendation, and career-guide APIs.",
        version="1.0.0",
    )
    register_exception_handlers(application)
    application.include_router(api_router)

    @application.get(
        "/health",
        response_model=HealthResponse,
        include_in_schema=False,
    )
    async def health_check() -> HealthResponse:
        """Return process liveness without invoking AWS or business services."""

        return {"status": "ok"}

    return application


app = create_application()
