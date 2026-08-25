"""Pydantic V2 contracts for every iFormat AI endpoint."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AISchema(BaseModel):
    """Base contract with strict keys and predictable alias serialization."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class AIResponse(AISchema):
    """Metadata returned by all billable AI operations."""

    model: str = Field(min_length=1, description="Bedrock model or profile ID used.")
    tokens_used: int = Field(
        alias="tokensUsed",
        ge=0,
        description="Total input and output tokens consumed.",
    )


class ScreeningRequest(AISchema):
    """Candidate CV and job description to compare."""

    cv_json: dict[str, Any] = Field(min_length=1)
    job_description: str = Field(min_length=1, max_length=50_000)


class ScreeningResponse(AIResponse):
    """Structured candidate-to-role screening assessment."""

    score: int = Field(ge=0, le=100)
    recommendation: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class CoverLetterRequest(AISchema):
    """Inputs required to generate a tailored cover letter."""

    candidate_name: str = Field(alias="candidateName", min_length=1, max_length=200)
    role: str = Field(min_length=1, max_length=300)
    company: str = Field(min_length=1, max_length=300)
    recipient: str = Field(min_length=1, max_length=300)
    experience_context: str = Field(
        alias="experienceContext",
        min_length=1,
        max_length=50_000,
    )
    tone: str = Field(min_length=1, max_length=100)


class CoverLetterResponse(AIResponse):
    """Generated cover letter and usage metadata."""

    letter: str = Field(min_length=1)


class ColdEmailRequest(AISchema):
    """Inputs required to generate a professional cold email."""

    recipient: str = Field(min_length=1, max_length=300)
    role: str = Field(min_length=1, max_length=300)
    company: str = Field(min_length=1, max_length=300)
    context: str = Field(min_length=1, max_length=30_000)
    tone: str = Field(min_length=1, max_length=100)


class ColdEmailResponse(AIResponse):
    """Generated cold email and usage metadata."""

    email: str = Field(min_length=1)


class ResumeOptimizerRequest(AISchema):
    """Raw resume content and the desired target market."""

    raw_text: str = Field(alias="rawText", min_length=1, max_length=100_000)
    target_role: str = Field(alias="targetRole", min_length=1, max_length=300)
    target_industry: str = Field(
        alias="targetIndustry",
        min_length=1,
        max_length=300,
    )


class ResumeOptimizerResponse(AIResponse):
    """ATS-optimized resume text and usage metadata."""

    summary: str = Field(min_length=1)


class CVBuilderRequest(AISchema):
    """Unstructured notes from which to assemble a CV."""

    raw_notes: str = Field(min_length=1, max_length=100_000)


class CVBuilderResponse(AIResponse):
    """Normalized CV sections and usage metadata."""

    personal: dict[str, Any]
    experiences: list[dict[str, Any]] = Field(default_factory=list)
    education: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class ProductRecommenderRequest(AISchema):
    """Candidate profile used to recommend iFormat products."""

    job_title: str = Field(min_length=1, max_length=300)
    experience_level: str = Field(min_length=1, max_length=200)
    career_goals: str = Field(min_length=1, max_length=20_000)
    skills: list[str] = Field(default_factory=list, max_length=200)
    industry: str = Field(min_length=1, max_length=300)


class ProductRecommenderResponse(AIResponse):
    """Ranked product suggestions and usage metadata."""

    recommendations: list[dict[str, Any]] = Field(default_factory=list)


class CareerChatRequest(AISchema):
    """Career-advisor query with optional prior conversation messages."""

    query: str = Field(min_length=1, max_length=20_000)
    chat_history: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=100,
    )

    @field_validator("chat_history")
    @classmethod
    def validate_chat_history(
        cls,
        value: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Require each history item to contain a supported role and content."""

        if value is None:
            return None
        supported_roles = {"ai", "assistant", "human", "user"}
        for index, message in enumerate(value):
            role = message.get("role")
            content = message.get("content")
            if role not in supported_roles:
                raise ValueError(
                    f"chat_history[{index}].role must be user or assistant"
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"chat_history[{index}].content must be a non-empty string"
                )
        return value


class CareerChatResponse(AIResponse):
    """Grounded career-advisor answer and usage metadata."""

    response: str = Field(min_length=1)
