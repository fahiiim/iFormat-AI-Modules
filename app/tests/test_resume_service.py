"""Unit tests for resume PDF extraction and generation."""

import base64
from io import BytesIO
from typing import Any

import pytest
from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas

from app.core.exceptions import InvalidResumePDFException
from app.services.resume_service import ResumeOptimizationService


class FakeResumeContentOptimizer:
    """Return deterministic normalized resume content without calling AWS."""

    extracted_text = ""
    job_description = ""

    async def optimize_resume(
        self,
        raw_text: str,
        target_role: str,
        target_industry: str,
        job_description: str,
    ) -> dict[str, Any]:
        """Capture extracted text and return a complete model response."""

        del target_role, target_industry
        self.extracted_text = raw_text
        self.job_description = job_description
        return {
            "personal": {
                "name": "Ada Lovelace",
                "headline": "Backend Engineer",
                "email": "ada@example.com",
                "phone": "+1 555 0100",
                "location": "London",
                "links": ["github.com/ada"],
            },
            "professionalSummary": (
                "Backend engineer building reliable Python services."
            ),
            "coreSkills": ["Python", "FastAPI", "AWS"],
            "experiences": [
                {
                    "title": "Software Engineer",
                    "company": "Analytical Engines",
                    "location": "London",
                    "startDate": "2022",
                    "endDate": "Present",
                    "bullets": ["Built reliable API services."],
                }
            ],
            "education": [
                {
                    "qualification": "BSc Computer Science",
                    "institution": "University of London",
                    "location": "London",
                    "completionDate": "2021",
                    "details": "Major: Artificial Intelligence",
                }
            ],
            "projects": [
                {
                    "name": "Job Portal",
                    "technologies": ["Python", "FAISS"],
                    "bullets": ["Implemented semantic career search."],
                }
            ],
            "certifications": ["AWS Certified Developer"],
            "model": "test-model",
            "tokensUsed": 123,
        }


def make_text_pdf() -> bytes:
    """Create a minimal text-based source resume PDF."""

    buffer = BytesIO()
    canvas = Canvas(buffer)
    canvas.drawString(72, 760, "Ada Lovelace - Python Backend Engineer")
    canvas.drawString(72, 740, "Built FastAPI services for job-search workflows.")
    canvas.save()
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_resume_service_extracts_and_generates_valid_pdf() -> None:
    """A text resume should become a readable optimized PDF response."""

    optimizer = FakeResumeContentOptimizer()
    service = ResumeOptimizationService(optimizer)

    result = await service.optimize_resume_pdf(
        pdf_bytes=make_text_pdf(),
        original_filename="Ada Resume.pdf",
        target_role="Backend Engineer",
        target_industry="Technology",
        job_description="Build reliable Python backend services.",
    )

    assert "FastAPI services" in optimizer.extracted_text
    assert "reliable Python" in optimizer.job_description
    assert result["fileName"] == "Ada-Resume-optimized.pdf"
    assert result["model"] == "test-model"
    assert result["tokensUsed"] == 123
    generated_pdf = base64.b64decode(result["pdfBase64"])
    reader = PdfReader(BytesIO(generated_pdf))
    generated_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) >= 1
    assert "Ada Lovelace" in generated_text
    assert "PROFESSIONAL EXPERIENCE" in generated_text
    assert "Major: Artificial Intelligence" in generated_text


@pytest.mark.asyncio
async def test_resume_service_rejects_non_pdf_bytes() -> None:
    """A non-PDF upload should fail before invoking the model."""

    service = ResumeOptimizationService(FakeResumeContentOptimizer())

    with pytest.raises(InvalidResumePDFException):
        await service.optimize_resume_pdf(
            pdf_bytes=b"not a pdf",
            original_filename="resume.pdf",
            target_role="Backend Engineer",
            target_industry="Technology",
            job_description="Build reliable Python backend services.",
        )
