"""Unit tests for Bedrock request and response normalization."""

from collections.abc import Mapping
from typing import Any

import pytest
from botocore.exceptions import ClientError

from app.core.config import Settings
from app.core.exceptions import (
    BedrockResponseException,
    BedrockThrottlingException,
)
from app.services.bedrock_service import BedrockService
from app.schemas.ai_schemas import CVBuilderRequest, ScreeningRequest


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
        EMBEDDING_MODEL_ID="test-embedding",
        KNOWLEDGE_BASE_PATH="./missing-test-kb",
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
                                '"gaps":[]}'
                            )
                        }
                    ]
                }
            },
            "usage": {"totalTokens": 11},
        }
    )
    service = BedrockService(client=client, settings=make_settings())

    await service.screen_candidate(
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
                                '"certifications":[]}'
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
        )
    )

    prompt = client.request["messages"][0]["content"][0]["text"]
    assert '"email":"ada@example.com"' in prompt
    assert "Built FastAPI services." in prompt
