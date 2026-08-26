"""FastAPI application factory for the iFormat AI API."""

from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.exceptions import register_exception_handlers


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
    return application


app = create_application()
