"""Unit tests for ATS CV construction and PDF encoding."""

import base64
from io import BytesIO
from typing import Any

import pytest
from pypdf import PdfReader

from app.schemas.ai_schemas import CVBuilderRequest
from app.services.cv_builder_service import CVBuilderService


class FakeCVContentBuilder:
    """Return deterministic normalized CV content without calling AWS."""

    request: CVBuilderRequest | None = None

    async def build_cv(self, request: CVBuilderRequest) -> dict[str, Any]:
        """Capture inputs and return a complete internal CV contract."""

        self.request = request
        return {
            "personal": {
                "name": "Ada Lovelace",
                "headline": "Python Backend Engineer",
                "email": "ada@example.com",
                "phone": "+1 555 0100",
                "location": "London",
                "links": ["github.com/ada"],
            },
            "professionalSummary": (
                "Python backend engineer building reliable API platforms."
            ),
            "coreSkills": ["Python", "FastAPI", "AWS"],
            "experiences": [
                {
                    "title": "Software Engineer",
                    "company": "Analytical Engines",
                    "location": "London",
                    "startDate": "2022",
                    "endDate": "Present",
                    "bullets": ["Delivered reliable FastAPI services."],
                }
            ],
            "education": [
                {
                    "qualification": "BSc Computer Science",
                    "institution": "University of London",
                    "location": "London",
                    "completionDate": "2021",
                    "details": ["Artificial Intelligence concentration"],
                }
            ],
            "projects": [],
            "certifications": ["AWS Certified Developer"],
            "model": "test-model",
            "tokensUsed": 77,
        }


@pytest.mark.asyncio
async def test_cv_builder_merges_inputs_and_returns_valid_pdf() -> None:
    """CV construction should return ATS sections and a readable PDF."""

    content_builder = FakeCVContentBuilder()
    service = CVBuilderService(content_builder)
    request = CVBuilderRequest(
        user_info={"name": "Ada Lovelace", "email": "ada@example.com"},
        raw_notes="Built Python APIs for career products.",
    )

    result = await service.build_cv_pdf(request)

    assert content_builder.request == request
    assert result["fileName"] == "Ada-Lovelace-ats-cv.pdf"
    assert result["contentType"] == "application/pdf"
    assert result["skills"] == ["Python", "FastAPI", "AWS"]
    assert result["model"] == "test-model"
    assert result["tokensUsed"] == 77

    pdf_data = base64.b64decode(result["pdfBase64"])
    reader = PdfReader(BytesIO(pdf_data))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert pdf_data.startswith(b"%PDF-")
    assert "Ada Lovelace" in extracted
    assert "PROFESSIONAL EXPERIENCE" in extracted
    assert "Delivered reliable FastAPI services" in extracted
