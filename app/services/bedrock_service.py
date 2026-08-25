"""Amazon Bedrock Converse integration and structured AI use cases."""

import asyncio
import json
import logging
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Protocol, TypeVar, cast

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    BedrockInvocationException,
    BedrockResponseException,
    BedrockThrottlingException,
    BedrockUnavailableException,
)
from app.schemas.ai_schemas import (
    AIResponse,
    ColdEmailRequest,
    ColdEmailResponse,
    CoverLetterRequest,
    CoverLetterResponse,
    CVBuilderRequest,
    CVBuilderResponse,
    ProductRecommenderRequest,
    ProductRecommenderResponse,
    ResumeOptimizerRequest,
    ResumeOptimizerResponse,
    ScreeningRequest,
    ScreeningResponse,
)
from app.utils.prompts import (
    COLD_EMAIL_SYSTEM_PROMPT,
    COLD_EMAIL_USER_PROMPT,
    COVER_LETTER_SYSTEM_PROMPT,
    COVER_LETTER_USER_PROMPT,
    CV_BUILDER_SYSTEM_PROMPT,
    CV_BUILDER_USER_PROMPT,
    PRODUCT_RECOMMENDER_SYSTEM_PROMPT,
    PRODUCT_RECOMMENDER_USER_PROMPT,
    RESUME_OPTIMIZE_PROMPT,
    RESUME_OPTIMIZE_USER_PROMPT,
    SCREENING_SYSTEM_PROMPT,
    SCREENING_USER_PROMPT,
)

logger = logging.getLogger(__name__)

THROTTLING_ERROR_CODES = frozenset(
    {
        "ModelNotReadyException",
        "ModelTimeoutException",
        "ServiceQuotaExceededException",
        "ServiceUnavailableException",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)


class BedrockRuntimeClient(Protocol):
    """Minimal structural type required from a Bedrock Runtime client."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        """Send a message through the Bedrock Converse API."""


ResponseModelT = TypeVar("ResponseModelT", bound=AIResponse)


def create_bedrock_runtime_client(
    settings: Settings | None = None,
) -> BedrockRuntimeClient:
    """Create a configured boto3 Bedrock Runtime client.

    Args:
        settings: Optional settings override, primarily for dependency
            injection and tests.

    Returns:
        BedrockRuntimeClient: A client using boto3's standard AWS credential
        provider chain.
    """

    resolved_settings = settings or get_settings()
    client = boto3.client(
        "bedrock-runtime",
        region_name=resolved_settings.AWS_REGION,
        config=Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )
    return cast(BedrockRuntimeClient, client)


class BedrockService:
    """Execute structured Bedrock requests and validate feature contracts."""

    def __init__(
        self,
        client: BedrockRuntimeClient,
        settings: Settings,
    ) -> None:
        """Initialize the service.

        Args:
            client: Injected Bedrock Runtime client.
            settings: Validated runtime configuration.
        """

        self._client = client
        self._settings = settings

    async def invoke_claude_structured(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Invoke Claude through Converse and parse its enforced JSON output.

        boto3 is synchronous, so the network call is delegated to a worker
        thread to avoid blocking FastAPI's event loop.

        Args:
            system_prompt: Instruction defining Claude's role and JSON schema.
            user_prompt: User-specific content for the request.

        Returns:
            dict[str, Any]: An envelope containing ``data``, ``tokensUsed``,
            and ``model``.

        Raises:
            BedrockThrottlingException: If AWS asks the caller to retry later.
            BedrockInvocationException: If AWS rejects the request.
            BedrockUnavailableException: If the AWS SDK cannot reach Bedrock.
            BedrockResponseException: If the response is missing data or does
                not contain a valid JSON object.
        """

        try:
            response = await asyncio.to_thread(
                self._client.converse,
                modelId=self._settings.BEDROCK_MODEL_ID,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                inferenceConfig={"maxTokens": 4000, "temperature": 0.2},
                additionalModelRequestFields={
                    "response_format": {"type": "json_object"}
                },
            )
        except ClientError as exc:
            error_code = str(
                exc.response.get("Error", {}).get("Code", "UnknownClientError")
            )
            logger.exception("Bedrock Converse returned AWS error %s", error_code)
            if error_code in THROTTLING_ERROR_CODES:
                raise BedrockThrottlingException() from exc
            raise BedrockInvocationException() from exc
        except BotoCoreError as exc:
            logger.exception("Bedrock Converse transport or credential failure")
            raise BedrockUnavailableException() from exc

        try:
            output = response["output"]
            message = cast(Mapping[str, Any], output)["message"]
            content = cast(Mapping[str, Any], message)["content"]
            text_blocks = [
                block["text"]
                for block in cast(list[Mapping[str, Any]], content)
                if isinstance(block.get("text"), str)
            ]
            if not text_blocks:
                raise KeyError("No text block in Bedrock response")
            parsed_json = json.loads("".join(text_blocks))
            if not isinstance(parsed_json, dict):
                raise TypeError("Structured Bedrock output must be a JSON object")
            tokens_used = self._extract_total_tokens(response.get("usage", {}))
        except json.JSONDecodeError as exc:
            logger.exception("Bedrock returned invalid JSON")
            raise BedrockResponseException() from exc
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("Bedrock response envelope was malformed")
            raise BedrockResponseException() from exc

        return {
            "data": parsed_json,
            "tokensUsed": tokens_used,
            "model": self._settings.BEDROCK_MODEL_ID,
        }

    async def screen_candidate(self, request: ScreeningRequest) -> dict[str, Any]:
        """Score a candidate CV against a job description."""

        user_prompt = SCREENING_USER_PROMPT.format(
            cv_json=self._json(request.cv_json),
            job_description=request.job_description,
        )
        return await self._invoke_for_contract(
            ScreeningResponse,
            SCREENING_SYSTEM_PROMPT,
            user_prompt,
        )

    async def generate_cover_letter(
        self,
        request: CoverLetterRequest,
    ) -> dict[str, Any]:
        """Generate a tailored cover letter."""

        user_prompt = COVER_LETTER_USER_PROMPT.format(
            candidate_name=request.candidate_name,
            role=request.role,
            company=request.company,
            recipient=request.recipient,
            experience_context=request.experience_context,
            tone=request.tone,
        )
        return await self._invoke_for_contract(
            CoverLetterResponse,
            COVER_LETTER_SYSTEM_PROMPT,
            user_prompt,
        )

    async def generate_cold_email(
        self,
        request: ColdEmailRequest,
    ) -> dict[str, Any]:
        """Generate a concise professional cold email."""

        user_prompt = COLD_EMAIL_USER_PROMPT.format(
            recipient=request.recipient,
            role=request.role,
            company=request.company,
            context=request.context,
            tone=request.tone,
        )
        return await self._invoke_for_contract(
            ColdEmailResponse,
            COLD_EMAIL_SYSTEM_PROMPT,
            user_prompt,
        )

    async def optimize_resume(
        self,
        request: ResumeOptimizerRequest,
    ) -> dict[str, Any]:
        """Rewrite resume content for ATS relevance and impact."""

        user_prompt = RESUME_OPTIMIZE_USER_PROMPT.format(
            raw_text=request.raw_text,
            target_role=request.target_role,
            target_industry=request.target_industry,
        )
        return await self._invoke_for_contract(
            ResumeOptimizerResponse,
            RESUME_OPTIMIZE_PROMPT,
            user_prompt,
        )

    async def build_cv(self, request: CVBuilderRequest) -> dict[str, Any]:
        """Normalize unstructured notes into CV sections."""

        user_prompt = CV_BUILDER_USER_PROMPT.format(raw_notes=request.raw_notes)
        return await self._invoke_for_contract(
            CVBuilderResponse,
            CV_BUILDER_SYSTEM_PROMPT,
            user_prompt,
        )

    async def recommend_products(
        self,
        request: ProductRecommenderRequest,
    ) -> dict[str, Any]:
        """Recommend iFormat products for a candidate profile."""

        user_prompt = PRODUCT_RECOMMENDER_USER_PROMPT.format(
            job_title=request.job_title,
            experience_level=request.experience_level,
            career_goals=request.career_goals,
            skills=self._json(request.skills),
            industry=request.industry,
        )
        return await self._invoke_for_contract(
            ProductRecommenderResponse,
            PRODUCT_RECOMMENDER_SYSTEM_PROMPT,
            user_prompt,
        )

    async def _invoke_for_contract(
        self,
        response_model: type[ResponseModelT],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Invoke Bedrock and validate the merged response contract."""

        result = await self.invoke_claude_structured(system_prompt, user_prompt)
        candidate = {
            **result["data"],
            "model": result["model"],
            "tokensUsed": result["tokensUsed"],
        }
        try:
            validated = response_model.model_validate(candidate)
        except ValidationError as exc:
            logger.exception(
                "Bedrock JSON did not match %s",
                response_model.__name__,
            )
            raise BedrockResponseException() from exc
        return validated.model_dump(by_alias=True)

    @staticmethod
    def _extract_total_tokens(usage: Any) -> int:
        """Extract normalized token usage from a Converse response."""

        if not isinstance(usage, Mapping):
            return 0
        total = usage.get("totalTokens")
        if isinstance(total, int) and not isinstance(total, bool):
            return max(total, 0)
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        input_count = input_tokens if isinstance(input_tokens, int) else 0
        output_count = output_tokens if isinstance(output_tokens, int) else 0
        return max(input_count + output_count, 0)

    @staticmethod
    def _json(value: Any) -> str:
        """Serialize user data consistently for inclusion in a prompt."""

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1)
def get_default_bedrock_service() -> BedrockService:
    """Return the lazily constructed process-wide default Bedrock service."""

    settings = get_settings()
    return BedrockService(create_bedrock_runtime_client(settings), settings)


async def invoke_claude_structured(
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Invoke Claude using the default service and return structured output.

    Args:
        system_prompt: Instruction defining the task and JSON contract.
        user_prompt: Feature-specific user content.

    Returns:
        dict[str, Any]: Parsed JSON data, total tokens, and model identifier.
    """

    return await get_default_bedrock_service().invoke_claude_structured(
        system_prompt,
        user_prompt,
    )
