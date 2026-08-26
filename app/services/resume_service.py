"""Resume PDF extraction, AI optimization, and PDF generation service."""

import asyncio
import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from xml.sax.saxutils import escape

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.core.exceptions import (
    AIServiceException,
    InvalidResumePDFException,
    ResumeGenerationException,
    ResumePDFTooLargeException,
)
from app.schemas.ai_schemas import (
    ResumeEducation,
    ResumeExperience,
    ResumeOptimizerAIResponse,
    ResumeOptimizerResponse,
    ResumeProject,
)

MAX_RESUME_PDF_BYTES = 10 * 1024 * 1024
MAX_RESUME_PAGES = 30
MAX_EXTRACTED_CHARACTERS = 100_000


class ResumeContentOptimizer(Protocol):
    """Structural type for the AI operation used by resume orchestration."""

    async def optimize_resume(
        self,
        raw_text: str,
        target_role: str,
        target_industry: str,
    ) -> dict[str, Any]:
        """Return normalized optimized resume content."""


class ResumeOptimizationService:
    """Convert an uploaded resume PDF into an optimized downloadable PDF."""

    def __init__(self, bedrock_service: ResumeContentOptimizer) -> None:
        """Initialize the orchestration service.

        Args:
            bedrock_service: Structured AI service used to rewrite the resume.
        """

        self._bedrock_service = bedrock_service

    async def optimize_resume_pdf(
        self,
        pdf_bytes: bytes,
        original_filename: str | None,
        target_role: str,
        target_industry: str,
    ) -> dict[str, Any]:
        """Extract, optimize, render, and encode a resume PDF.

        Args:
            pdf_bytes: Raw bytes received from the uploaded PDF.
            original_filename: Client-supplied filename used only to derive a
                safe download filename.
            target_role: Role for which the document should be tailored.
            target_industry: Target industry terminology and context.

        Returns:
            dict[str, Any]: Base64 PDF data, filename, summary, model, and token
            usage matching ``ResumeOptimizerResponse``.

        Raises:
            ResumePDFTooLargeException: If the PDF exceeds the upload limit.
            InvalidResumePDFException: If text cannot be safely extracted.
            ResumeGenerationException: If the optimized PDF cannot be built.
            AIServiceException: If the Bedrock optimization call fails.
        """

        raw_text = await asyncio.to_thread(self._extract_pdf_text, pdf_bytes)
        ai_result = await self._bedrock_service.optimize_resume(
            raw_text=raw_text,
            target_role=target_role,
            target_industry=target_industry,
        )
        optimized_content = ResumeOptimizerAIResponse.model_validate(ai_result)

        try:
            optimized_pdf = await asyncio.to_thread(
                render_ats_resume_pdf,
                optimized_content,
                target_role,
            )
        except AIServiceException:
            raise
        except Exception as exc:
            raise ResumeGenerationException() from exc

        response = ResumeOptimizerResponse(
            summary=optimized_content.professional_summary,
            fileName=self._build_download_filename(original_filename),
            contentType="application/pdf",
            pdfBase64=base64.b64encode(optimized_pdf).decode("ascii"),
            model=optimized_content.model,
            tokensUsed=optimized_content.tokens_used,
        )
        return response.model_dump(by_alias=True)

    @staticmethod
    def _extract_pdf_text(pdf_bytes: bytes) -> str:
        """Validate a PDF and extract normalized text from every page."""

        if len(pdf_bytes) > MAX_RESUME_PDF_BYTES:
            raise ResumePDFTooLargeException(
                f"Resume PDFs must not exceed {MAX_RESUME_PDF_BYTES // 1024 // 1024} MB."
            )
        if not pdf_bytes or not pdf_bytes.lstrip().startswith(b"%PDF-"):
            raise InvalidResumePDFException()

        try:
            reader = PdfReader(BytesIO(pdf_bytes), strict=True)
            if reader.is_encrypted:
                raise InvalidResumePDFException(
                    "Password-protected resume PDFs are not supported."
                )
            if not reader.pages:
                raise InvalidResumePDFException("The resume PDF has no pages.")
            if len(reader.pages) > MAX_RESUME_PAGES:
                raise InvalidResumePDFException(
                    f"Resume PDFs must not exceed {MAX_RESUME_PAGES} pages."
                )

            page_text = [page.extract_text() or "" for page in reader.pages]
        except InvalidResumePDFException:
            raise
        except (PdfReadError, OSError, TypeError, ValueError) as exc:
            raise InvalidResumePDFException() from exc

        normalized_text = "\n\n".join(page_text).replace("\x00", " ").strip()
        if not normalized_text:
            raise InvalidResumePDFException(
                "No selectable text was found. Run OCR before uploading this resume."
            )
        if len(normalized_text) > MAX_EXTRACTED_CHARACTERS:
            raise InvalidResumePDFException(
                "The extracted resume text exceeds the supported length."
            )
        return normalized_text

    @classmethod
    def _render_resume_pdf(
        cls,
        content: ResumeOptimizerAIResponse,
        target_role: str,
    ) -> bytes:
        """Render structured resume content as a polished, ATS-friendly PDF."""

        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=17 * mm,
            leftMargin=17 * mm,
            topMargin=15 * mm,
            bottomMargin=16 * mm,
            title=f"Optimized Resume - {content.personal.name or target_role}",
            author="iFormat Job Portal",
        )
        styles = cls._build_styles()
        story: list[Any] = []

        display_name = content.personal.name or "Professional Resume"
        story.append(Paragraph(cls._safe_markup(display_name), styles["ResumeName"]))

        headline = content.personal.headline or target_role
        if headline:
            story.append(
                Paragraph(cls._safe_markup(headline), styles["ResumeHeadline"])
            )

        contact_items = [
            content.personal.email,
            content.personal.phone,
            content.personal.location,
            *content.personal.links,
        ]
        contact_line = " | ".join(item for item in contact_items if item)
        if contact_line:
            story.append(
                Paragraph(cls._safe_markup(contact_line), styles["ResumeContact"])
            )
        story.append(Spacer(1, 4 * mm))

        cls._append_section_heading(story, "PROFESSIONAL SUMMARY", styles)
        story.append(
            Paragraph(
                cls._safe_markup(content.professional_summary),
                styles["ResumeBody"],
            )
        )

        if content.core_skills:
            cls._append_section_heading(story, "CORE SKILLS", styles)
            story.append(
                Paragraph(
                    cls._safe_markup(" | ".join(content.core_skills)),
                    styles["ResumeBody"],
                )
            )

        if content.experiences:
            cls._append_section_heading(story, "PROFESSIONAL EXPERIENCE", styles)
            for experience in content.experiences:
                cls._append_experience(story, experience, styles)

        if content.projects:
            cls._append_section_heading(story, "PROJECTS", styles)
            for project in content.projects:
                cls._append_project(story, project, styles)

        if content.education:
            cls._append_section_heading(story, "EDUCATION", styles)
            for education in content.education:
                cls._append_education(story, education, styles)

        if content.certifications:
            cls._append_section_heading(story, "CERTIFICATIONS", styles)
            for certification in content.certifications:
                cls._append_bullet(story, certification, styles)

        document.build(
            story,
            onFirstPage=cls._draw_page_footer,
            onLaterPages=cls._draw_page_footer,
        )
        pdf_data = buffer.getvalue()
        cls._validate_generated_pdf(pdf_data)
        return pdf_data

    @staticmethod
    def _build_styles() -> dict[str, ParagraphStyle]:
        """Create the typography system used by generated resumes."""

        base = getSampleStyleSheet()
        navy = HexColor("#16324F")
        teal = HexColor("#167D8D")
        gray = HexColor("#4A5568")
        return {
            "ResumeName": ParagraphStyle(
                "ResumeName",
                parent=base["Title"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=23,
                textColor=navy,
                alignment=TA_CENTER,
                spaceAfter=2,
            ),
            "ResumeHeadline": ParagraphStyle(
                "ResumeHeadline",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=10.5,
                leading=13,
                textColor=teal,
                alignment=TA_CENTER,
                spaceAfter=2,
            ),
            "ResumeContact": ParagraphStyle(
                "ResumeContact",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=8.5,
                leading=11,
                textColor=gray,
                alignment=TA_CENTER,
            ),
            "SectionHeading": ParagraphStyle(
                "SectionHeading",
                parent=base["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=10,
                leading=12,
                textColor=navy,
                spaceBefore=9,
                spaceAfter=4,
                borderWidth=0,
                borderPadding=(0, 0, 2, 0),
            ),
            "EntryTitle": ParagraphStyle(
                "EntryTitle",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=9.5,
                leading=12,
                textColor=navy,
                spaceBefore=3,
                spaceAfter=1,
            ),
            "EntryMeta": ParagraphStyle(
                "EntryMeta",
                parent=base["Normal"],
                fontName="Helvetica-Oblique",
                fontSize=8.5,
                leading=10.5,
                textColor=gray,
                spaceAfter=2,
            ),
            "ResumeBody": ParagraphStyle(
                "ResumeBody",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=HexColor("#1F2933"),
                spaceAfter=3,
            ),
            "ResumeBullet": ParagraphStyle(
                "ResumeBullet",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=9,
                leading=11.5,
                leftIndent=10,
                firstLineIndent=-8,
                textColor=HexColor("#1F2933"),
                spaceAfter=2,
            ),
        }

    @classmethod
    def _append_section_heading(
        cls,
        story: list[Any],
        title: str,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append a consistently styled section heading."""

        story.append(Paragraph(cls._safe_markup(title), styles["SectionHeading"]))

    @classmethod
    def _append_experience(
        cls,
        story: list[Any],
        experience: ResumeExperience,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append one work-experience entry."""

        title_parts = [experience.title, experience.company]
        title = " | ".join(part for part in title_parts if part)
        if title:
            story.append(
                Paragraph(f"<b>{cls._safe_markup(title)}</b>", styles["EntryTitle"])
            )
        meta_parts = [
            " - ".join(
                part for part in (experience.start_date, experience.end_date) if part
            ),
            experience.location,
        ]
        meta = " | ".join(part for part in meta_parts if part)
        if meta:
            story.append(Paragraph(cls._safe_markup(meta), styles["EntryMeta"]))
        for bullet in experience.bullets:
            cls._append_bullet(story, bullet, styles)

    @classmethod
    def _append_education(
        cls,
        story: list[Any],
        education: ResumeEducation,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append one education entry."""

        title = " | ".join(
            part for part in (education.qualification, education.institution) if part
        )
        if title:
            story.append(
                Paragraph(f"<b>{cls._safe_markup(title)}</b>", styles["EntryTitle"])
            )
        meta = " | ".join(
            part for part in (education.completion_date, education.location) if part
        )
        if meta:
            story.append(Paragraph(cls._safe_markup(meta), styles["EntryMeta"]))
        for detail in education.details:
            cls._append_bullet(story, detail, styles)

    @classmethod
    def _append_project(
        cls,
        story: list[Any],
        project: ResumeProject,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append one project entry."""

        if project.name:
            story.append(
                Paragraph(
                    f"<b>{cls._safe_markup(project.name)}</b>",
                    styles["EntryTitle"],
                )
            )
        if project.technologies:
            story.append(
                Paragraph(
                    cls._safe_markup(" | ".join(project.technologies)),
                    styles["EntryMeta"],
                )
            )
        for bullet in project.bullets:
            cls._append_bullet(story, bullet, styles)

    @classmethod
    def _append_bullet(
        cls,
        story: list[Any],
        text: str,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append one ASCII-hyphen bullet paragraph."""

        if text.strip():
            story.append(
                Paragraph(f"- {cls._safe_markup(text)}", styles["ResumeBullet"])
            )

    @staticmethod
    def _safe_markup(value: str) -> str:
        """Normalize unsafe Unicode punctuation and escape ReportLab markup."""

        normalized = (
            value.replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2022", "-")
            .replace("\x00", " ")
        )
        return escape(normalized.strip())

    @staticmethod
    def _draw_page_footer(canvas: Canvas, document: Any) -> None:
        """Draw a discreet brand marker and page number."""

        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(HexColor("#718096"))
        canvas.drawString(17 * mm, 8 * mm, "iFormat ATS Resume")
        canvas.drawRightString(
            A4[0] - 17 * mm,
            8 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    @staticmethod
    def _validate_generated_pdf(pdf_data: bytes) -> None:
        """Reopen a generated PDF and verify pages and extractable text."""

        try:
            reader = PdfReader(BytesIO(pdf_data), strict=True)
            extracted = "".join(page.extract_text() or "" for page in reader.pages)
            if not reader.pages or not extracted.strip():
                raise ResumeGenerationException()
        except ResumeGenerationException:
            raise
        except (PdfReadError, OSError, TypeError, ValueError) as exc:
            raise ResumeGenerationException() from exc

    @staticmethod
    def _build_download_filename(original_filename: str | None) -> str:
        """Build a safe and stable optimized-PDF download filename."""

        source_stem = Path(original_filename or "resume").stem
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", source_stem).strip("-_")
        return f"{safe_stem or 'resume'}-optimized.pdf"


def render_ats_resume_pdf(
    content: ResumeOptimizerAIResponse,
    fallback_headline: str,
) -> bytes:
    """Render validated resume content as an extractable ATS-friendly PDF.

    Args:
        content: Validated structured resume sections and AI usage metadata.
        fallback_headline: Heading used when no professional headline exists.

    Returns:
        bytes: Complete PDF document bytes.
    """

    return ResumeOptimizationService._render_resume_pdf(content, fallback_headline)


def build_cv_download_filename(candidate_name: str) -> str:
    """Build a safe filename for a newly generated CV.

    Args:
        candidate_name: Candidate name returned in the normalized CV.

    Returns:
        str: Stable filename ending in ``-ats-cv.pdf``.
    """

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate_name).strip("-_")
    return f"{safe_name or 'iformat'}-ats-cv.pdf"
