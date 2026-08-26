"""FastAPI dependency providers for configuration and AI services."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.bedrock_service import (
    BedrockRuntimeClient,
    BedrockService,
    create_bedrock_runtime_client,
)
from app.services.cv_builder_service import CVBuilderService
from app.services.rag_service import RAGService, get_default_rag_service
from app.services.resume_service import ResumeOptimizationService


@lru_cache(maxsize=1)
def _get_cached_bedrock_client() -> BedrockRuntimeClient:
    """Create and cache one thread-safe boto3 client per process."""

    return create_bedrock_runtime_client(get_settings())


def get_bedrock_client() -> BedrockRuntimeClient:
    """Provide the shared Bedrock Runtime client to request dependencies.

    Returns:
        BedrockRuntimeClient: Cached boto3 Bedrock Runtime client.
    """

    return _get_cached_bedrock_client()


def get_bedrock_service(
    client: Annotated[BedrockRuntimeClient, Depends(get_bedrock_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BedrockService:
    """Build a request-scoped structured Bedrock service.

    Args:
        client: Injected process-wide Bedrock client.
        settings: Injected immutable runtime settings.

    Returns:
        BedrockService: Service configured for the current request.
    """

    return BedrockService(client=client, settings=settings)


def get_rag_service(request: Request) -> RAGService:
    """Provide the RAG service initialized by the application lifespan.

    Args:
        request: Current FastAPI request, used to access lifespan state.

    Returns:
        RAGService: Process-wide career-advisor retrieval service.
    """

    service = getattr(request.app.state, "rag_service", None)
    if isinstance(service, RAGService):
        return service
    return get_default_rag_service()


def get_resume_optimization_service(
    bedrock_service: Annotated[BedrockService, Depends(get_bedrock_service)],
) -> ResumeOptimizationService:
    """Build the resume PDF orchestration service for a request.

    Args:
        bedrock_service: Injected structured Bedrock integration.

    Returns:
        ResumeOptimizationService: Resume extraction and generation service.
    """

    return ResumeOptimizationService(bedrock_service=bedrock_service)


def get_cv_builder_service(
    bedrock_service: Annotated[BedrockService, Depends(get_bedrock_service)],
) -> CVBuilderService:
    """Build the ATS CV PDF orchestration service for a request.

    Args:
        bedrock_service: Injected structured Bedrock integration.

    Returns:
        CVBuilderService: CV data merge and PDF generation service.
    """

    return CVBuilderService(bedrock_service=bedrock_service)


BedrockServiceDependency = Annotated[BedrockService, Depends(get_bedrock_service)]
RAGServiceDependency = Annotated[RAGService, Depends(get_rag_service)]
ResumeOptimizationServiceDependency = Annotated[
    ResumeOptimizationService,
    Depends(get_resume_optimization_service),
]
CVBuilderServiceDependency = Annotated[
    CVBuilderService,
    Depends(get_cv_builder_service),
]
