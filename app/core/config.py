"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the iFormat AI service.

    Values are read from process environment variables or a local ``.env``
    file. AWS credentials intentionally remain outside this model so boto3 can
    use its standard credential provider chain (IAM role, profile, or
    environment variables).
    """

    AWS_REGION: str = Field(default="eu-west-1", min_length=1)
    BEDROCK_MODEL_ID: str = Field(
        default=(
            "arn:aws:bedrock:eu-west-1:952409747578:inference-profile/"
            "eu.anthropic.claude-opus-5"
        ),
        min_length=1,
    )
    EMBEDDING_MODEL_ID: str = Field(
        default="amazon.titan-embed-text-v2:0",
        min_length=1,
    )
    KNOWLEDGE_BASE_PATH: str = Field(
        default="./app/data/iformat_kb",
        min_length=1,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        frozen=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance.

    Returns:
        Settings: Validated application configuration.
    """

    return Settings()
