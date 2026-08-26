"""Pydantic V2 contracts for every iFormat AI endpoint."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
        min_length=1,
        description="Authoritative candidate profile supplied by the backend.",
    )
    cv_json: dict[str, Any] = Field(min_length=1)
    job_description: str = Field(min_length=1, max_length=50_000)


class ScreeningScoreBreakdown(AISchema):
    """Category-level job-match scores supporting the overall score."""

    skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)
    domain_match: int = Field(alias="domainMatch", ge=0, le=100)


class ScreeningEvidence(AISchema):
    """One source-backed finding used in a screening decision."""

    category: Literal["skills", "experience", "education", "domain_match"]
    finding: str = Field(min_length=1)
    source: str = Field(
        min_length=1,
        description="CV or backend-profile field supporting the finding.",
    )


class ScreeningResponse(AIResponse):
    """Structured candidate-to-role screening assessment."""

    score: int = Field(ge=0, le=100)
    recommendation: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    strengths: list[str]
    gaps: list[str]
    score_breakdown: ScreeningScoreBreakdown = Field(alias="scoreBreakdown")
    evidence: list[ScreeningEvidence] = Field(min_length=1)


class CoverLetterRequest(AISchema):
    """Inputs required to generate a tailored cover letter."""

    candidate_profile: dict[str, Any] = Field(
        alias="candidateProfile",
        min_length=1,
        description="Complete authoritative candidate profile from the backend.",
    )
    role: str = Field(min_length=1, max_length=300)
    company: str = Field(min_length=1, max_length=300)
    recipient: str = Field(min_length=1, max_length=300)
    job_description: str = Field(
        alias="jobDescription",
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
    job_description: str = Field(
        alias="jobDescription",
        min_length=1,
        max_length=50_000,
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
    core_skills: list[str] = Field(alias="coreSkills")
    experiences: list[ResumeExperience]
    education: list[ResumeEducation]
    projects: list[ResumeProject]
    certifications: list[str]

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


class CVBuilderAIResponse(ResumeOptimizerAIResponse):
    """Internal CV-builder content including incomplete-profile feedback."""

    missing_information: list[str] = Field(
        alias="missingInformation",
    )

    @field_validator("missing_information", mode="before")
    @classmethod
    def normalize_missing_information(cls, value: Any) -> Any:
        """Accept one missing-information message as a one-item list."""

        return _normalize_string_list(value)


class CVBuilderRequest(AISchema):
    """Backend profile data and user notes from which to assemble a CV."""

    user_info: dict[str, Any] = Field(
        min_length=1,
        description="Authoritative user profile supplied by the backend.",
    )
    raw_notes: str = Field(min_length=1, max_length=100_000)
    target_role: str = Field(alias="targetRole", min_length=1, max_length=300)
    target_industry: str = Field(
        alias="targetIndustry",
        min_length=1,
        max_length=300,
    )
    job_description: str | None = Field(
        alias="jobDescription",
        default=None,
        min_length=1,
        max_length=50_000,
    )


class CVBuilderResponse(AIResponse):
    """Normalized CV sections, an ATS PDF, and usage metadata."""

    personal: dict[str, Any]
    experiences: list[dict[str, Any]]
    education: list[dict[str, Any]]
    skills: list[str]
    missing_information: list[str] = Field(
        alias="missingInformation",
    )
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
    product_catalog: list["ProductCatalogItem"] = Field(
        alias="productCatalog",
        min_length=1,
        max_length=500,
        description="Controlled backend product catalog eligible for recommendation.",
    )

    @model_validator(mode="after")
    def validate_unique_product_ids(self) -> "ProductRecommenderRequest":
        """Require every controlled catalog product ID to be unique."""

        product_ids = [item.product_id for item in self.product_catalog]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("productCatalog must contain unique productId values")
        return self


class ProductCatalogItem(AISchema):
    """One backend-controlled product available for recommendation."""

    product_id: str = Field(alias="productId", min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=10_000)
    target_roles: list[str] = Field(alias="targetRoles", default_factory=list)
    target_levels: list[str] = Field(alias="targetLevels", default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProductRecommendation(AISchema):
    """One catalog-grounded product recommendation."""

    product_id: str = Field(alias="productId", min_length=1)
    name: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    fit_score: int = Field(alias="fitScore", ge=0, le=100)


class ProductRecommenderResponse(AIResponse):
    """Ranked product suggestions and usage metadata."""

    recommendations: list[ProductRecommendation]


class CareerContextSource(AISchema):
    """Backend-provided career context that may support a chat response."""

    source_id: str = Field(alias="sourceId", min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=50_000)


class CareerChatMessage(AISchema):
    """One validated prior user or assistant chat message."""

    role: Literal["ai", "assistant", "human", "user"]
    content: str = Field(min_length=1, max_length=20_000)


class CareerChatRequest(AISchema):
    """Career-guide query grounded in backend-provided user context."""

    query: str = Field(min_length=1, max_length=20_000)
    user_info: dict[str, Any] = Field(
        min_length=1,
        description="Complete authoritative user profile supplied by the backend.",
    )
    context_sources: list[CareerContextSource] = Field(
        alias="contextSources",
        default_factory=list,
        max_length=100,
    )
    chat_history: list[CareerChatMessage] | None = Field(
        default=None,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_context_source_ids(self) -> "CareerChatRequest":
        """Reserve the profile source ID and reject ambiguous duplicates."""

        source_ids = [source.source_id for source in self.context_sources]
        if "user_profile" in source_ids:
            raise ValueError("contextSources cannot use reserved ID 'user_profile'")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("contextSources must contain unique sourceId values")
        return self


class CareerChatAIResponse(AIResponse):
    """Internal career-guide output before source IDs are resolved."""

    response: str = Field(min_length=1)
    supported: bool
    source_ids: list[str] = Field(alias="sourceIds")


class CareerChatSourceReference(AISchema):
    """Canonical backend source cited by a career-guide response."""

    source_id: str = Field(alias="sourceId", min_length=1)
    title: str = Field(min_length=1)


class CareerChatResponse(AIResponse):
    """Context-grounded career-guide answer and usage metadata."""

    response: str = Field(min_length=1)
    supported: bool
    sources: list[CareerChatSourceReference]
