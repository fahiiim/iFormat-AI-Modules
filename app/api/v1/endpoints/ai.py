"""HTTP-only route handlers for the iFormat AI API."""

from typing import Any

from fastapi import APIRouter, status

from app.api.dependencies import BedrockServiceDependency, RAGServiceDependency
from app.core.exceptions import (
    AIServiceException,
    raise_http_exception_for_service_error,
)
from app.schemas.ai_schemas import (
    CareerChatRequest,
    CareerChatResponse,
    ColdEmailRequest,
    ColdEmailResponse,
    CoverLetterRequest,
    CoverLetterResponse,
    CVBuilderRequest,
    CVBuilderResponse,
    ProductRecommenderRequest,
    ProductRecommenderResponse,
    ResumeOptimizerRequest,
    ResumeOptimizerResponse,
    ScreeningRequest,
    ScreeningResponse,
)

router = APIRouter(prefix="/ai", tags=["AI Services"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "description": "The AI provider returned an invalid response or failed."
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "description": "The AI provider or knowledge base is temporarily unavailable."
    },
}


@router.post(
    "/screen",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def screen_candidate(
    payload: ScreeningRequest,
    service: BedrockServiceDependency,
) -> ScreeningResponse:
    """Screen a candidate CV against a job description."""

    try:
        result = await service.screen_candidate(payload)
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return ScreeningResponse.model_validate(result)


@router.post(
    "/cover-letter",
    response_model=CoverLetterResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def generate_cover_letter(
    payload: CoverLetterRequest,
    service: BedrockServiceDependency,
) -> CoverLetterResponse:
    """Generate a tailored cover letter."""

    try:
        result = await service.generate_cover_letter(payload)
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return CoverLetterResponse.model_validate(result)


@router.post(
    "/email",
    response_model=ColdEmailResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def generate_cold_email(
    payload: ColdEmailRequest,
    service: BedrockServiceDependency,
) -> ColdEmailResponse:
    """Generate a professional cold outreach email."""

    try:
        result = await service.generate_cold_email(payload)
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return ColdEmailResponse.model_validate(result)


@router.post(
    "/resume/optimize",
    response_model=ResumeOptimizerResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def optimize_resume(
    payload: ResumeOptimizerRequest,
    service: BedrockServiceDependency,
) -> ResumeOptimizerResponse:
    """Optimize raw resume content for a target role and industry."""

    try:
        result = await service.optimize_resume(payload)
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return ResumeOptimizerResponse.model_validate(result)


@router.post(
    "/cv/build",
    response_model=CVBuilderResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def build_cv(
    payload: CVBuilderRequest,
    service: BedrockServiceDependency,
) -> CVBuilderResponse:
    """Build normalized CV sections from unstructured career notes."""

    try:
        result = await service.build_cv(payload)
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return CVBuilderResponse.model_validate(result)


@router.post(
    "/recommend",
    response_model=ProductRecommenderResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def recommend_products(
    payload: ProductRecommenderRequest,
    service: BedrockServiceDependency,
) -> ProductRecommenderResponse:
    """Recommend iFormat products for a candidate profile."""

    try:
        result = await service.recommend_products(payload)
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return ProductRecommenderResponse.model_validate(result)


@router.post(
    "/chat",
    response_model=CareerChatResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def query_career_advisor(
    payload: CareerChatRequest,
    service: RAGServiceDependency,
) -> CareerChatResponse:
    """Answer a career question using the iFormat RAG knowledge base."""

    try:
        result = await service.query_career_advisor(
            payload.query,
            payload.chat_history,
        )
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    return CareerChatResponse.model_validate(result)
