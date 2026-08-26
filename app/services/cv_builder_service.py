"""ATS CV construction orchestration and PDF encoding service."""

import asyncio
import base64
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.exceptions import BedrockResponseException, ResumeGenerationException
from app.schemas.ai_schemas import (
    CVBuilderRequest,
    CVBuilderResponse,
    ResumeOptimizerAIResponse,
)
from app.services.resume_service import (
    build_cv_download_filename,
    render_ats_resume_pdf,
)


class CVContentBuilder(Protocol):
    """Structural type for the AI operation used by CV construction."""

    async def build_cv(self, request: CVBuilderRequest) -> dict[str, Any]:
        """Return normalized ATS resume content."""


class CVBuilderService:
    """Merge backend/user inputs and return an encoded ATS-friendly CV PDF."""

    def __init__(self, bedrock_service: CVContentBuilder) -> None:
        """Initialize the CV construction service.

        Args:
            bedrock_service: Structured AI service used to assemble CV data.
        """

        self._bedrock_service = bedrock_service

    async def build_cv_pdf(self, request: CVBuilderRequest) -> dict[str, Any]:
        """Generate normalized CV sections and an ATS-friendly PDF.

        Args:
            request: Backend user profile and user-authored career notes.

        Returns:
            dict[str, Any]: CV sections, Base64 PDF data, model, and token usage.

        Raises:
            BedrockResponseException: If AI output violates the CV contract.
            ResumeGenerationException: If the PDF cannot be rendered.
        """

        ai_result = await self._bedrock_service.build_cv(request)
        try:
            content = ResumeOptimizerAIResponse.model_validate(ai_result)
        except ValidationError as exc:
            raise BedrockResponseException() from exc

        try:
            pdf_data = await asyncio.to_thread(
                render_ats_resume_pdf,
                content,
                content.personal.headline or "Professional CV",
            )
        except ResumeGenerationException:
            raise
        except Exception as exc:
            raise ResumeGenerationException() from exc

        response = CVBuilderResponse(
            personal=content.personal.model_dump(by_alias=True),
            experiences=[
                item.model_dump(by_alias=True) for item in content.experiences
            ],
            education=[item.model_dump(by_alias=True) for item in content.education],
            skills=content.core_skills,
            fileName=build_cv_download_filename(content.personal.name),
            contentType="application/pdf",
            pdfBase64=base64.b64encode(pdf_data).decode("ascii"),
            model=content.model,
            tokensUsed=content.tokens_used,
        )
        return response.model_dump(by_alias=True)
