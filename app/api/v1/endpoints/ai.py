"""HTTP-only route handlers for the iFormat AI API."""

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile, status

from app.api.dependencies import (
    BedrockServiceDependency,
    CVBuilderServiceDependency,
    RAGServiceDependency,
    ResumeOptimizationServiceDependency,
)
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
    ResumeOptimizerResponse,
    ScreeningRequest,
    ScreeningResponse,
)
from app.services.resume_service import MAX_RESUME_PDF_BYTES

router = APIRouter(prefix="/ai", tags=["AI Services"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "description": "The AI provider returned an invalid response or failed."
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "description": "The AI provider or knowledge base is temporarily unavailable."
    },
}

RESUME_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "description": "The uploaded resume PDF exceeds 10 MB."
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": "The upload is not a readable, text-based resume PDF."
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
    responses=RESUME_ERROR_RESPONSES,
)
async def optimize_resume(
    resume: Annotated[
        UploadFile,
        File(description="Original text-based resume in PDF format."),
    ],
    target_role: Annotated[
        str,
        Form(alias="targetRole", min_length=1, max_length=300),
    ],
    target_industry: Annotated[
        str,
        Form(alias="targetIndustry", min_length=1, max_length=300),
    ],
    service: ResumeOptimizationServiceDependency,
) -> ResumeOptimizerResponse:
    """Extract an uploaded PDF and return a newly optimized resume PDF."""

    try:
        pdf_bytes = await resume.read(MAX_RESUME_PDF_BYTES + 1)
        result = await service.optimize_resume_pdf(
            pdf_bytes=pdf_bytes,
            original_filename=resume.filename,
            target_role=target_role,
            target_industry=target_industry,
        )
    except AIServiceException as exc:
        raise_http_exception_for_service_error(exc)
    finally:
        await resume.close()
    return ResumeOptimizerResponse.model_validate(result)


@router.post(
    "/cv/build",
    response_model=CVBuilderResponse,
    status_code=status.HTTP_200_OK,
    responses=ERROR_RESPONSES,
)
async def build_cv(
    payload: CVBuilderRequest,
    service: CVBuilderServiceDependency,
) -> CVBuilderResponse:
    """Merge backend data and notes into an encoded ATS-friendly CV PDF."""

    try:
        result = await service.build_cv_pdf(payload)
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
