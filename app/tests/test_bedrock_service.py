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
    assert client.request["additionalModelRequestFields"] == {
        "response_format": {"type": "json_object"}
    }
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
