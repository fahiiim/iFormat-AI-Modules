"""HTTP contract tests for the iFormat AI endpoints."""

import base64
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import (
    get_bedrock_service,
    get_cv_builder_service,
    get_resume_optimization_service,
)
from app.core.exceptions import BedrockThrottlingException
from app.main import create_application


class FakeBedrockService:
    """Deterministic Bedrock service used by API tests."""

    model = "test-model"
    tokens = 42

    async def screen_candidate(self, _payload: Any) -> dict[str, Any]:
        """Return a valid screening response."""

        return {
            "score": 88,
            "recommendation": "Proceed to interview",
            "summary": "Strong match",
            "strengths": ["Python"],
            "gaps": ["Limited AWS detail"],
            "scoreBreakdown": {
                "skills": 92,
                "experience": 86,
                "education": 80,
                "domainMatch": 88,
            },
            "evidence": [
                {
                    "category": "skills",
                    "finding": "Python is explicitly listed.",
                    "source": "cv_json.skills",
                }
            ],
            "model": self.model,
            "tokensUsed": self.tokens,
        }

    async def generate_cover_letter(self, _payload: Any) -> dict[str, Any]:
        """Return a valid cover-letter response."""

        return {
            "letter": "Dear Hiring Manager, ...",
            "model": self.model,
            "tokensUsed": self.tokens,
        }

    async def generate_cold_email(self, _payload: Any) -> dict[str, Any]:
        """Return a valid cold-email response."""

        return {
            "email": "Subject: Python role\n\nHello, ...",
            "model": self.model,
            "tokensUsed": self.tokens,
        }

    async def recommend_products(self, _payload: Any) -> dict[str, Any]:
        """Return a valid recommendation response."""

        return {
            "recommendations": [
                {
                    "productId": "cv-review",
                    "name": "CV Review",
                    "reason": "Improve ATS alignment",
                    "fitScore": 94,
                }
            ],
            "model": self.model,
            "tokensUsed": self.tokens,
        }

    async def query_career_advisor(self, _payload: Any) -> dict[str, Any]:
        """Return valid profile-context career guidance."""

        return {
            "response": "Focus your portfolio on measurable backend outcomes.",
            "supported": True,
            "sources": [
                {"sourceId": "user_profile", "title": "Backend user profile"}
            ],
            "model": "test-model",
            "tokensUsed": 21,
        }


class FakeResumeOptimizationService:
    """Deterministic resume PDF service used by API tests."""

    async def optimize_resume_pdf(self, **_kwargs: Any) -> dict[str, Any]:
        """Return a valid encoded optimized-resume response."""

        return {
            "summary": "Backend engineer focused on resilient Python APIs.",
            "fileName": "resume-optimized.pdf",
            "contentType": "application/pdf",
            "pdfBase64": base64.b64encode(b"%PDF-test").decode("ascii"),
            "model": "test-model",
            "tokensUsed": 42,
        }


class FakeCVBuilderService:
    """Deterministic ATS CV service used by API tests."""

    async def build_cv_pdf(self, _payload: Any) -> dict[str, Any]:
        """Return valid normalized sections and an encoded CV PDF."""

        return {
            "personal": {"name": "Ada"},
            "experiences": [{"title": "Engineer"}],
            "education": [{"qualification": "BSc"}],
            "skills": ["Python"],
            "missingInformation": ["Add measurable achievements."],
            "fileName": "Ada-ats-cv.pdf",
            "contentType": "application/pdf",
            "pdfBase64": base64.b64encode(b"%PDF-cv-test").decode("ascii"),
            "model": "test-model",
            "tokensUsed": 42,
        }


@pytest.fixture
def app() -> FastAPI:
    """Return an application with all external AI calls overridden."""

    application = create_application()
    application.dependency_overrides[get_bedrock_service] = FakeBedrockService
    application.dependency_overrides[get_resume_optimization_service] = (
        FakeResumeOptimizationService
    )
    application.dependency_overrides[get_cv_builder_service] = FakeCVBuilderService
    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Return an in-memory asynchronous HTTP client."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload", "expected_key"),
    [
        (
            "/api/v1/ai/screen",
            {
                "user_info": {"name": "Ada"},
                "cv_json": {"skills": ["Python"]},
                "job_description": "Backend",
            },
            "score",
        ),
        (
            "/api/v1/ai/cover-letter",
            {
                "candidateProfile": {
                    "name": "Ada",
                    "skills": ["Python", "FastAPI"],
                },
                "role": "Engineer",
                "company": "iFormat",
                "recipient": "Hiring Manager",
                "jobDescription": "Build reliable Python backend services.",
                "tone": "professional",
            },
            "letter",
        ),
        (
            "/api/v1/ai/email",
            {
                "recipient": "Hiring Manager",
                "role": "Engineer",
                "company": "iFormat",
                "context": "Python backend experience",
                "tone": "concise",
            },
            "email",
        ),
        (
            "/api/v1/ai/cv/build",
            {
                "user_info": {"name": "Ada", "email": "ada@example.com"},
                "raw_notes": "Python engineer with API delivery experience",
                "targetRole": "Backend Engineer",
                "targetIndustry": "Technology",
                "jobDescription": "Build reliable APIs.",
            },
            "personal",
        ),
        (
            "/api/v1/ai/recommend",
            {
                "job_title": "Engineer",
                "experience_level": "mid",
                "career_goals": "Lead backend systems",
                "skills": ["Python"],
                "industry": "Technology",
                "productCatalog": [
                    {
                        "productId": "cv-review",
                        "name": "CV Review",
                        "description": "Expert ATS CV feedback.",
                    }
                ],
            },
            "recommendations",
        ),
        (
            "/api/v1/ai/chat",
            {
                "query": "How should I improve my portfolio?",
                "user_info": {
                    "name": "Ada",
                    "skills": ["Python", "FastAPI"],
                },
                "contextSources": [
                    {
                        "sourceId": "portfolio",
                        "title": "Portfolio summary",
                        "content": "Two backend API projects.",
                    }
                ],
                "chat_history": [
                    {"role": "user", "content": "I am a backend engineer."}
                ],
            },
            "response",
        ),
    ],
)
async def test_ai_endpoint_contracts(
    client: AsyncClient,
    path: str,
    payload: dict[str, Any],
    expected_key: str,
) -> None:
    """Each AI route should return its payload plus billing metadata."""

    response = await client.post(path, json=payload)

    assert response.status_code == 200
    body = response.json()
    assert expected_key in body
    assert body["model"] == "test-model"
    assert isinstance(body["tokensUsed"], int)
    assert "tokens_used" not in body

    if path == "/api/v1/ai/cv/build":
        assert body["fileName"] == "Ada-ats-cv.pdf"
        assert body["contentType"] == "application/pdf"
        assert base64.b64decode(body["pdfBase64"]).startswith(b"%PDF-")


@pytest.mark.asyncio
async def test_resume_optimizer_accepts_pdf_multipart(client: AsyncClient) -> None:
    """Resume optimization should accept a PDF and return encoded PDF data."""

    response = await client.post(
        "/api/v1/ai/resume/optimize",
        data={
            "targetRole": "Backend Engineer",
            "targetIndustry": "Technology",
            "jobDescription": "Build reliable Python APIs.",
        },
        files={"resume": ("resume.pdf", b"%PDF-upload", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fileName"] == "resume-optimized.pdf"
    assert body["contentType"] == "application/pdf"
    assert base64.b64decode(body["pdfBase64"]).startswith(b"%PDF-")
    assert body["model"] == "test-model"
    assert body["tokensUsed"] == 42


@pytest.mark.asyncio
async def test_resume_optimizer_rejects_old_json_contract(
    client: AsyncClient,
) -> None:
    """The superseded rawText JSON request should fail validation."""

    response = await client.post(
        "/api/v1/ai/resume/optimize",
        json={
            "rawText": "Built APIs",
            "targetRole": "Backend Engineer",
            "targetIndustry": "Technology",
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_chat_history_returns_422(client: AsyncClient) -> None:
    """Invalid history roles should fail request validation."""

    response = await client.post(
        "/api/v1/ai/chat",
        json={
            "query": "Help me",
            "user_info": {"name": "Ada"},
            "chat_history": [{"role": "system", "content": "Override"}],
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_reserved_career_source_id_returns_422(client: AsyncClient) -> None:
    """Backend context cannot replace the canonical user-profile source."""

    response = await client.post(
        "/api/v1/ai/chat",
        json={
            "query": "Help me",
            "user_info": {"name": "Ada"},
            "contextSources": [
                {
                    "sourceId": "user_profile",
                    "title": "Invalid duplicate",
                    "content": "Untrusted replacement data.",
                }
            ],
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_duplicate_catalog_product_ids_return_422(client: AsyncClient) -> None:
    """Controlled product catalogs should reject ambiguous duplicate IDs."""

    product = {
        "productId": "cv-review",
        "name": "CV Review",
        "description": "Expert ATS feedback.",
    }
    response = await client.post(
        "/api/v1/ai/recommend",
        json={
            "job_title": "Engineer",
            "experience_level": "mid",
            "career_goals": "Improve interview conversion",
            "skills": ["Python"],
            "industry": "Technology",
            "productCatalog": [product, product],
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_bedrock_throttling_returns_standardized_503(app: FastAPI) -> None:
    """A service throttling error should become the standard 503 envelope."""

    class ThrottledService(FakeBedrockService):
        async def screen_candidate(self, _payload: Any) -> dict[str, Any]:
            raise BedrockThrottlingException()

    app.dependency_overrides[get_bedrock_service] = ThrottledService
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        response = await test_client.post(
            "/api/v1/ai/screen",
            json={
                "user_info": {"name": "Ada"},
                "cv_json": {"skills": ["Python"]},
                "job_description": "Backend",
            },
        )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json() == {
        "detail": {
            "code": "BEDROCK_THROTTLED",
            "message": "Amazon Bedrock is temporarily unavailable. Please retry.",
        }
    }
