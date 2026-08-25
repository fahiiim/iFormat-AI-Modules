"""HTTP contract tests for the iFormat AI endpoints."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_bedrock_service, get_rag_service
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

    async def optimize_resume(self, _payload: Any) -> dict[str, Any]:
        """Return a valid resume-optimizer response."""

        return {
            "summary": "Built resilient Python APIs.",
            "model": self.model,
            "tokensUsed": self.tokens,
        }

    async def build_cv(self, _payload: Any) -> dict[str, Any]:
        """Return a valid CV-builder response."""

        return {
            "personal": {"name": "Ada"},
            "experiences": [{"role": "Engineer"}],
            "education": [{"degree": "BSc"}],
            "skills": ["Python"],
            "model": self.model,
            "tokensUsed": self.tokens,
        }

    async def recommend_products(self, _payload: Any) -> dict[str, Any]:
        """Return a valid recommendation response."""

        return {
            "recommendations": [
                {"name": "CV Review", "reason": "Improve ATS alignment"}
            ],
            "model": self.model,
            "tokensUsed": self.tokens,
        }


class FakeRAGService:
    """Deterministic career-advisor service used by API tests."""

    async def query_career_advisor(
        self,
        _query: str,
        _chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a valid RAG response."""

        return {
            "response": "Focus your portfolio on measurable backend outcomes.",
            "model": "test-model",
            "tokensUsed": 21,
        }


@pytest.fixture
def app() -> FastAPI:
    """Return an application with all external AI calls overridden."""

    application = create_application()
    application.dependency_overrides[get_bedrock_service] = FakeBedrockService
    application.dependency_overrides[get_rag_service] = FakeRAGService
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
            {"cv_json": {"skills": ["Python"]}, "job_description": "Backend"},
            "score",
        ),
        (
            "/api/v1/ai/cover-letter",
            {
                "candidateName": "Ada",
                "role": "Engineer",
                "company": "iFormat",
                "recipient": "Hiring Manager",
                "experienceContext": "Python backend experience",
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
            "/api/v1/ai/resume/optimize",
            {
                "rawText": "Built APIs",
                "targetRole": "Backend Engineer",
                "targetIndustry": "Technology",
            },
            "summary",
        ),
        (
            "/api/v1/ai/cv/build",
            {"raw_notes": "Ada, Python engineer"},
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
            },
            "recommendations",
        ),
        (
            "/api/v1/ai/chat",
            {
                "query": "How should I improve my portfolio?",
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


@pytest.mark.asyncio
async def test_invalid_chat_history_returns_422(client: AsyncClient) -> None:
    """Invalid history roles should fail request validation."""

    response = await client.post(
        "/api/v1/ai/chat",
        json={
            "query": "Help me",
            "chat_history": [{"role": "system", "content": "Override"}],
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
            json={"cv_json": {"skills": ["Python"]}, "job_description": "Backend"},
        )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json() == {
        "detail": {
            "code": "BEDROCK_THROTTLED",
            "message": "Amazon Bedrock is temporarily unavailable. Please retry.",
        }
    }
