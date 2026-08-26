"""Unit tests for Bedrock request and response normalization."""

import json
from collections.abc import Mapping
from typing import Any

import pytest
from botocore.exceptions import ClientError

from app.core.config import Settings
from app.core.exceptions import (
    BedrockResponseException,
    BedrockThrottlingException,
)
from app.schemas.ai_schemas import (
    CVBuilderRequest,
    CareerChatRequest,
    CoverLetterRequest,
    ProductRecommenderRequest,
    ScreeningRequest,
)
from app.services.bedrock_service import BedrockService


class FakeBedrockClient:
    """Capture Converse arguments and return a configured response."""

    def __init__(
        self,
        response: Mapping[str, Any] | None = None,
        error: ClientError | None = None,
    ) -> None:
        """Initialize the fake client response or error."""

        self.response = response or {}
        self.error = error
        self.request: dict[str, Any] = {}

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        """Record the request and return or raise the configured result."""

        self.request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def make_settings() -> Settings:
    """Return deterministic settings independent of the local environment."""

    return Settings(
        AWS_REGION="eu-west-1",
        BEDROCK_MODEL_ID="test-profile",
    )


@pytest.mark.asyncio
async def test_structured_converse_payload_and_usage() -> None:
    """The service should send the mandated payload and normalize metadata."""

    client = FakeBedrockClient(
        response={
            "output": {"message": {"content": [{"text": '{"summary":"Qualified"}'}]}},
            "usage": {"inputTokens": 12, "outputTokens": 8, "totalTokens": 20},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    result = await service.invoke_claude_structured("system", "user")

    assert result == {
        "data": {"summary": "Qualified"},
        "tokensUsed": 20,
        "model": "test-profile",
    }
    assert client.request["modelId"] == "test-profile"
    assert client.request["inferenceConfig"] == {
        "maxTokens": 4000,
        "temperature": 0.2,
    }
    assert client.request["additionalModelRequestFields"] == {}
    assert "invoke_model" not in client.request


@pytest.mark.asyncio
async def test_invalid_bedrock_json_raises_response_error() -> None:
    """Malformed provider JSON should not escape the service boundary."""

    client = FakeBedrockClient(
        response={
            "output": {"message": {"content": [{"text": "not-json"}]}},
            "usage": {"totalTokens": 3},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    with pytest.raises(BedrockResponseException):
        await service.invoke_claude_structured("system", "user")


@pytest.mark.asyncio
async def test_markdown_fenced_json_is_accepted() -> None:
    """GLM JSON wrapped in a Markdown fence should still parse safely."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [{"text": '```json\n{"summary":"Qualified"}\n```'}]
                }
            },
            "usage": {"totalTokens": 7},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    result = await service.invoke_claude_structured("system", "user")

    assert result["data"] == {"summary": "Qualified"}
    assert result["tokensUsed"] == 7


@pytest.mark.asyncio
async def test_throttling_client_error_is_retryable() -> None:
    """AWS throttling should map to the retryable service exception."""

    aws_error = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "Converse",
    )
    client = FakeBedrockClient(error=aws_error)
    service = BedrockService(client=client, settings=make_settings())

    with pytest.raises(BedrockThrottlingException):
        await service.invoke_claude_structured("system", "user")


@pytest.mark.asyncio
async def test_screening_prompt_includes_backend_profile_and_cv() -> None:
    """Screening should compare both backend data and CV evidence."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"score":90,"recommendation":"Interview",'
                                '"summary":"Strong fit","strengths":["Python"],'
                                '"gaps":[],"scoreBreakdown":{"skills":95,'
                                '"experience":85,"education":80,'
                                '"domainMatch":90},"evidence":[{"category":'
                                '"skills","finding":"Python listed",'
                                '"source":"cv_json.skills"},{"category":'
                                '"experience","finding":"API work listed",'
                                '"source":"cv_json.experiences"},{"category":'
                                '"education","finding":"Degree listed",'
                                '"source":"cv_json.education"},{"category":'
                                '"domain_match","finding":"Backend match",'
                                '"source":"user_info.target_role"}]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 11},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    result = await service.screen_candidate(
        ScreeningRequest(
            user_info={"name": "Ada"},
            cv_json={"skills": ["Python"]},
            job_description="Python Engineer",
        )
    )

    prompt = client.request["messages"][0]["content"][0]["text"]
    assert '"name":"Ada"' in prompt
    assert '"skills":["Python"]' in prompt
    assert "Python Engineer" in prompt
    assert result["score"] == 90
    output_schema = json.loads(
        client.request["outputConfig"]["textFormat"]["structure"]["jsonSchema"][
            "schema"
        ]
    )
    assert "scoreBreakdown" in output_schema["properties"]
    assert "model" not in output_schema["properties"]
    assert "tokensUsed" not in output_schema["properties"]


@pytest.mark.asyncio
async def test_cv_builder_prompt_merges_backend_info_and_notes() -> None:
    """CV building should supply both input sources to the model."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"personal":{"name":"Ada"},'
                                '"professionalSummary":"Python engineer",'
                                '"coreSkills":["Python"],"experiences":[],'
                                '"education":[],"projects":[],'
                                '"certifications":[],"missingInformation":'
                                '["Add employment dates"]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 15},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    await service.build_cv(
        CVBuilderRequest(
            user_info={"name": "Ada", "email": "ada@example.com"},
            raw_notes="Built FastAPI services.",
            targetRole="Backend Engineer",
            targetIndustry="Technology",
            jobDescription="Build reliable API services.",
        )
    )

    prompt = client.request["messages"][0]["content"][0]["text"]
    assert '"email":"ada@example.com"' in prompt
    assert "Built FastAPI services." in prompt
    assert "Backend Engineer" in prompt
    assert "Build reliable API services." in prompt


@pytest.mark.asyncio
async def test_cover_letter_prompt_uses_full_profile_and_job_description() -> None:
    """Cover-letter generation should receive complete backend context."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [{"text": '{"letter":"Dear Hiring Manager"}'}]
                }
            },
            "usage": {"totalTokens": 6},
        }
    )
    service = BedrockService(client=client, settings=make_settings())
    await service.generate_cover_letter(
        CoverLetterRequest(
            candidateProfile={
                "name": "Ada",
                "skills": ["Python", "FastAPI"],
            },
            role="Backend Engineer",
            company="iFormat",
            recipient="Hiring Manager",
            jobDescription="Build reliable Python APIs.",
            tone="professional",
        )
    )

    prompt = client.request["messages"][0]["content"][0]["text"]
    assert '"skills":["Python","FastAPI"]' in prompt
    assert "Build reliable Python APIs." in prompt


@pytest.mark.asyncio
async def test_resume_prompt_includes_target_job_description() -> None:
    """Resume optimization should target the supplied vacancy requirements."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"personal":{"name":"Ada"},'
                                '"professionalSummary":"Backend engineer",'
                                '"coreSkills":["Python"],"experiences":[],'
                                '"education":[],"projects":[],'
                                '"certifications":[]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 8},
        }
    )
    service = BedrockService(client=client, settings=make_settings())
    await service.optimize_resume(
        raw_text="Ada - Python engineer",
        target_role="Backend Engineer",
        target_industry="Technology",
        job_description="Build resilient FastAPI services on AWS.",
    )

    prompt = client.request["messages"][0]["content"][0]["text"]
    assert "Build resilient FastAPI services on AWS." in prompt


@pytest.mark.asyncio
async def test_product_recommendations_are_limited_to_catalog_ids() -> None:
    """Product output should use backend catalog IDs and canonical names."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"recommendations":[{"productId":"cv-review",'
                                '"name":"Incorrect model name","reason":'
                                '"Improves ATS alignment","fitScore":95}]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 9},
        }
    )
    service = BedrockService(client=client, settings=make_settings())
    result = await service.recommend_products(
        ProductRecommenderRequest(
            job_title="Backend Engineer",
            experience_level="mid",
            career_goals="Improve interview conversion",
            skills=["Python"],
            industry="Technology",
            productCatalog=[
                {
                    "productId": "cv-review",
                    "name": "iFormat CV Review",
                    "description": "Expert ATS CV feedback.",
                }
            ],
        )
    )

    assert result["recommendations"][0]["productId"] == "cv-review"
    assert result["recommendations"][0]["name"] == "iFormat CV Review"
    prompt = client.request["messages"][0]["content"][0]["text"]
    assert '"productId":"cv-review"' in prompt


@pytest.mark.asyncio
async def test_career_guide_resolves_backend_source_references() -> None:
    """Career guidance should cite only canonical backend context sources."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"response":"Add measurable API outcomes.",'
                                '"supported":true,"sourceIds":'
                                '["user_profile","portfolio"]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 12},
        }
    )
    service = BedrockService(client=client, settings=make_settings())
    result = await service.query_career_advisor(
        CareerChatRequest(
            query="How can I improve my profile?",
            user_info={"skills": ["Python"]},
            contextSources=[
                {
                    "sourceId": "portfolio",
                    "title": "Portfolio summary",
                    "content": "Two API projects without outcome metrics.",
                }
            ],
        )
    )

    assert result["supported"] is True
    assert result["sources"] == [
        {"sourceId": "user_profile", "title": "Backend user profile"},
        {"sourceId": "portfolio", "title": "Portfolio summary"},
    ]


@pytest.mark.asyncio
async def test_career_guide_rejects_unknown_model_source() -> None:
    """A hallucinated source ID should fail the service contract."""

    client = FakeBedrockClient(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                '{"response":"Unsupported claim",'
                                '"supported":true,"sourceIds":["invented"]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 5},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    with pytest.raises(BedrockResponseException):
        await service.query_career_advisor(
            CareerChatRequest(
                query="What should I do?",
                user_info={"skills": ["Python"]},
            )
        )
