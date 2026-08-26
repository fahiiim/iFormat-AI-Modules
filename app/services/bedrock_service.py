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
    CareerChatAIResponse,
    CareerChatRequest,
    CareerChatResponse,
    CareerChatSourceReference,
    ColdEmailRequest,
    ColdEmailResponse,
    CoverLetterRequest,
    CoverLetterResponse,
    CVBuilderAIResponse,
    CVBuilderRequest,
    ProductRecommenderRequest,
    ProductRecommenderResponse,
    ResumeOptimizerAIResponse,
    ScreeningRequest,
    ScreeningResponse,
)
from app.utils.prompts import (
    CAREER_GUIDE_SYSTEM_PROMPT,
    CAREER_GUIDE_USER_PROMPT,
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
    client_options: dict[str, Any] = {
        "region_name": resolved_settings.AWS_REGION,
        "config": Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    }
    if (
        resolved_settings.AWS_ACCESS_KEY_ID is not None
        and resolved_settings.AWS_SECRET_ACCESS_KEY is not None
    ):
        client_options["aws_access_key_id"] = (
            resolved_settings.AWS_ACCESS_KEY_ID.get_secret_value()
        )
        client_options["aws_secret_access_key"] = (
            resolved_settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
        )
        if resolved_settings.AWS_SESSION_TOKEN is not None:
            client_options["aws_session_token"] = (
                resolved_settings.AWS_SESSION_TOKEN.get_secret_value()
            )

    client = boto3.client("bedrock-runtime", **client_options)
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
        response_model: type[AIResponse] | None = None,
    ) -> dict[str, Any]:
        """Invoke the configured chat model and parse its JSON output.

        boto3 is synchronous, so the network call is delegated to a worker
        thread to avoid blocking FastAPI's event loop.

        Args:
            system_prompt: Instruction defining Claude's role and JSON schema.
            user_prompt: User-specific content for the request.
            response_model: Optional contract used to enforce Bedrock JSON
                Schema structured output.

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

        converse_request: dict[str, Any] = {
            "modelId": self._settings.BEDROCK_MODEL_ID,
            "system": [{"text": system_prompt}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            "inferenceConfig": {"maxTokens": 4000, "temperature": 0.2},
            "additionalModelRequestFields": {},
        }
        if response_model is not None:
            converse_request["outputConfig"] = self._build_output_config(
                response_model
            )

        try:
            response = await asyncio.to_thread(
                self._client.converse,
                **converse_request,
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
            parsed_json = self._parse_json_object("".join(text_blocks))
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
            user_info=self._json(request.user_info),
            cv_json=self._json(request.cv_json),
            job_description=request.job_description,
        )
        result = await self._invoke_for_contract(
            ScreeningResponse,
            SCREENING_SYSTEM_PROMPT,
            user_prompt,
        )
        breakdown = result["scoreBreakdown"]
        result["score"] = round(
            (breakdown["skills"] * 0.40)
            + (breakdown["experience"] * 0.30)
            + (breakdown["education"] * 0.10)
            + (breakdown["domainMatch"] * 0.20)
        )
        required_categories = {
            "skills",
            "experience",
            "education",
            "domain_match",
        }
        evidence_categories = {
            item["category"] for item in result["evidence"]
        }
        valid_sources = all(
            item["source"].startswith(("cv_json", "user_info"))
            for item in result["evidence"]
        )
        if evidence_categories != required_categories or not valid_sources:
            raise BedrockResponseException()
        return ScreeningResponse.model_validate(result).model_dump(by_alias=True)

    async def generate_cover_letter(
        self,
        request: CoverLetterRequest,
    ) -> dict[str, Any]:
        """Generate a tailored cover letter."""

        user_prompt = COVER_LETTER_USER_PROMPT.format(
            candidate_profile=self._json(request.candidate_profile),
            role=request.role,
            company=request.company,
            recipient=request.recipient,
            job_description=request.job_description,
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
        raw_text: str,
        target_role: str,
        target_industry: str,
        job_description: str,
    ) -> dict[str, Any]:
        """Rebuild extracted resume content into normalized ATS sections.

        Args:
            raw_text: Text extracted from the uploaded resume PDF.
            target_role: Role for which the resume should be tailored.
            target_industry: Industry whose terminology should be considered.
            job_description: Target vacancy requirements and responsibilities.

        Returns:
            dict[str, Any]: Structured resume content with usage metadata.
        """

        user_prompt = RESUME_OPTIMIZE_USER_PROMPT.format(
            raw_text=raw_text,
            target_role=target_role,
            target_industry=target_industry,
            job_description=job_description,
        )
        return await self._invoke_for_contract(
            ResumeOptimizerAIResponse,
            RESUME_OPTIMIZE_PROMPT,
            user_prompt,
        )

    async def build_cv(self, request: CVBuilderRequest) -> dict[str, Any]:
        """Merge backend profile data and notes into ATS resume sections."""

        user_prompt = CV_BUILDER_USER_PROMPT.format(
            user_info=self._json(request.user_info),
            raw_notes=request.raw_notes,
            target_role=request.target_role,
            target_industry=request.target_industry,
            job_description=request.job_description or "Not provided",
        )
        return await self._invoke_for_contract(
            CVBuilderAIResponse,
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
            product_catalog=self._json(
                [item.model_dump(by_alias=True) for item in request.product_catalog]
            ),
        )
        result = await self._invoke_for_contract(
            ProductRecommenderResponse,
            PRODUCT_RECOMMENDER_SYSTEM_PROMPT,
            user_prompt,
        )
        catalog_names = {
            item.product_id: item.name for item in request.product_catalog
        }
        seen_ids: set[str] = set()
        for recommendation in result["recommendations"]:
            product_id = recommendation["productId"]
            if product_id not in catalog_names or product_id in seen_ids:
                raise BedrockResponseException()
            recommendation["name"] = catalog_names[product_id]
            seen_ids.add(product_id)
        return ProductRecommenderResponse.model_validate(result).model_dump(
            by_alias=True
        )

    async def query_career_advisor(
        self,
        request: CareerChatRequest,
    ) -> dict[str, Any]:
        """Guide a user using only backend-supplied profile context.

        Args:
            request: User profile, backend context sources, chat history, and
                current career question.

        Returns:
            dict[str, Any]: Grounded guidance, canonical sources, and usage.

        Raises:
            BedrockResponseException: If the model cites an unknown source or
                claims support without citing backend context.
        """

        source_titles = {"user_profile": "Backend user profile"}
        source_titles.update(
            {source.source_id: source.title for source in request.context_sources}
        )
        user_prompt = CAREER_GUIDE_USER_PROMPT.format(
            user_info=self._json(request.user_info),
            context_sources=self._json(
                [
                    source.model_dump(by_alias=True)
                    for source in request.context_sources
                ]
            ),
            allowed_source_ids=self._json(list(source_titles)),
            chat_history=self._json(
                [
                    message.model_dump(by_alias=True)
                    for message in (request.chat_history or [])
                ]
            ),
            query=request.query,
        )
        result = await self._invoke_for_contract(
            CareerChatAIResponse,
            CAREER_GUIDE_SYSTEM_PROMPT,
            user_prompt,
        )

        source_ids = list(dict.fromkeys(result["sourceIds"]))
        if any(source_id not in source_titles for source_id in source_ids):
            raise BedrockResponseException()
        if result["supported"] and not source_ids:
            raise BedrockResponseException()

        response = CareerChatResponse(
            response=result["response"],
            supported=result["supported"],
            sources=[
                CareerChatSourceReference(
                    sourceId=source_id,
                    title=source_titles[source_id],
                )
                for source_id in source_ids
            ],
            model=result["model"],
            tokensUsed=result["tokensUsed"],
        )
        return response.model_dump(by_alias=True)

    async def _invoke_for_contract(
        self,
        response_model: type[ResponseModelT],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Invoke Bedrock and validate the merged response contract."""

        result = await self.invoke_claude_structured(
            system_prompt,
            user_prompt,
            response_model,
        )
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
    def _parse_json_object(raw_text: str) -> dict[str, Any]:
        """Parse a JSON object, tolerating an optional Markdown JSON fence."""

        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed_json = json.loads(text)
        if not isinstance(parsed_json, dict):
            raise TypeError("Structured Bedrock output must be a JSON object")
        return parsed_json

    @staticmethod
    def _json(value: Any) -> str:
        """Serialize user data consistently for inclusion in a prompt."""

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _build_output_config(
        cls,
        response_model: type[AIResponse],
    ) -> dict[str, Any]:
        """Build a Bedrock Converse JSON Schema output configuration."""

        schema = response_model.model_json_schema(by_alias=True)
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            properties.pop("model", None)
            properties.pop("tokensUsed", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [
                item for item in required if item not in {"model", "tokensUsed"}
            ]

        sanitized_schema = cls._sanitize_bedrock_schema(schema)
        schema_name = response_model.__name__.removesuffix("Response").lower()
        return {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(
                            sanitized_schema,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "name": schema_name[:64],
                        "description": (
                            f"Structured output for {response_model.__name__}"
                        ),
                    }
                },
            }
        }

    @classmethod
    def _sanitize_bedrock_schema(cls, value: Any) -> Any:
        """Remove JSON Schema keywords unsupported by Bedrock grammars."""

        unsupported_keywords = {
            "default",
            "examples",
            "exclusiveMaximum",
            "exclusiveMinimum",
            "maxItems",
            "maxLength",
            "maximum",
            "minLength",
            "minimum",
            "multipleOf",
            "pattern",
            "title",
        }
        if isinstance(value, dict):
            return {
                key: cls._sanitize_bedrock_schema(item)
                for key, item in value.items()
                if key not in unsupported_keywords
            }
        if isinstance(value, list):
            return [cls._sanitize_bedrock_schema(item) for item in value]
        return value


@lru_cache(maxsize=1)
def get_default_bedrock_service() -> BedrockService:
    """Return the lazily constructed process-wide default Bedrock service."""

    settings = get_settings()
    return BedrockService(create_bedrock_runtime_client(settings), settings)


async def invoke_claude_structured(
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Invoke the configured model and return structured output.

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
