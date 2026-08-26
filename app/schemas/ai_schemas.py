"""Pydantic V2 contracts for every iFormat AI endpoint."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_string_list(value: Any) -> Any:
    """Normalize common model variations for fields declared as string lists."""

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return value


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
    """Backend candidate data, CV content, and job description to compare."""

    user_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Authoritative candidate profile supplied by the backend.",
    )
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
    """Targeting metadata submitted alongside a resume PDF upload."""

    target_role: str = Field(alias="targetRole", min_length=1, max_length=300)
    target_industry: str = Field(
        alias="targetIndustry",
        min_length=1,
        max_length=300,
    )


class ResumeOptimizerResponse(AIResponse):
    """Generated resume PDF payload and usage metadata."""

    summary: str = Field(min_length=1)
    file_name: str = Field(alias="fileName", min_length=1)
    content_type: str = Field(alias="contentType", default="application/pdf")
    pdf_base64: str = Field(
        alias="pdfBase64",
        min_length=1,
        description="Base64-encoded optimized resume PDF.",
    )


class ResumePersonalDetails(AISchema):
    """Contact and identity details extracted from the uploaded resume."""

    name: str = ""
    headline: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)

    @field_validator("links", mode="before")
    @classmethod
    def normalize_links(cls, value: Any) -> Any:
        """Accept a single link string as a one-item list."""

        return _normalize_string_list(value)


class ResumeExperience(AISchema):
    """One normalized work-experience entry."""

    title: str = ""
    company: str = ""
    location: str = ""
    start_date: str = Field(alias="startDate", default="")
    end_date: str = Field(alias="endDate", default="")
    bullets: list[str] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def normalize_bullets(cls, value: Any) -> Any:
        """Accept a single achievement string as a one-item list."""

        return _normalize_string_list(value)


class ResumeEducation(AISchema):
    """One normalized education entry."""

    qualification: str = ""
    institution: str = ""
    location: str = ""
    completion_date: str = Field(alias="completionDate", default="")
    details: list[str] = Field(default_factory=list)

    @field_validator("details", mode="before")
    @classmethod
    def normalize_details(cls, value: Any) -> Any:
        """Accept a single education detail as a one-item list."""

        return _normalize_string_list(value)


class ResumeProject(AISchema):
    """One normalized project entry."""

    name: str = ""
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)

    @field_validator("technologies", "bullets", mode="before")
    @classmethod
    def normalize_project_lists(cls, value: Any) -> Any:
        """Accept single project technology or bullet strings as lists."""

        return _normalize_string_list(value)


class ResumeOptimizerAIResponse(AIResponse):
    """Internal structured content returned by the optimization model."""

    personal: ResumePersonalDetails
    professional_summary: str = Field(alias="professionalSummary", min_length=1)
    core_skills: list[str] = Field(alias="coreSkills", default_factory=list)
    experiences: list[ResumeExperience] = Field(default_factory=list)
    education: list[ResumeEducation] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    @field_validator("core_skills", "certifications", mode="before")
    @classmethod
    def normalize_top_level_string_lists(cls, value: Any) -> Any:
        """Accept single skill or certification strings as lists."""

        return _normalize_string_list(value)

    @field_validator("experiences", "education", "projects", mode="before")
    @classmethod
    def normalize_section_collections(cls, value: Any) -> Any:
        """Accept one section object when the model omits the surrounding list."""

        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


class CVBuilderRequest(AISchema):
    """Backend profile data and user notes from which to assemble a CV."""

    user_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Authoritative user profile supplied by the backend.",
    )
    raw_notes: str = Field(min_length=1, max_length=100_000)


class CVBuilderResponse(AIResponse):
    """Normalized CV sections, an ATS PDF, and usage metadata."""

    personal: dict[str, Any]
    experiences: list[dict[str, Any]] = Field(default_factory=list)
    education: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    file_name: str = Field(alias="fileName", min_length=1)
    content_type: str = Field(alias="contentType", default="application/pdf")
    pdf_base64: str = Field(
        alias="pdfBase64",
        min_length=1,
        description="Base64-encoded ATS-friendly CV PDF.",
    )


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
