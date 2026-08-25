"""FastAPI application factory and lifespan management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.exceptions import AIServiceException, register_exception_handlers
from app.services.rag_service import get_default_rag_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and release the process-wide career-advisor RAG service.

    A missing or temporarily unavailable knowledge base does not prevent the
    other AI endpoints from starting. The chat endpoint reports a standardized
    503 until a subsequent process start can initialize the index.

    Args:
        app: FastAPI application entering its ASGI lifespan.

    Yields:
        None: Control while the application is serving requests.
    """

    rag_service = get_default_rag_service()
    app.state.rag_service = rag_service
    try:
        await rag_service.initialize()
    except AIServiceException as exc:
        logger.warning("Career-advisor startup deferred: %s", exc)

    try:
        yield
    finally:
        await rag_service.close()


def create_application() -> FastAPI:
    """Create and configure the iFormat AI FastAPI application.

    Returns:
        FastAPI: Fully wired ASGI application.
    """

    application = FastAPI(
        title="iFormat Job Portal AI Services",
        description="AI screening, writing, recommendation, and career RAG APIs.",
        version="1.0.0",
        lifespan=lifespan,
    )
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_application()
