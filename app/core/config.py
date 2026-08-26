"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the iFormat AI service.

    Values are read from process environment variables or a local ``.env``
    file. Explicit credentials are supported for local development; production
    deployments should rely on boto3's IAM role credential provider.
    """

    AWS_REGION: str = Field(default="eu-west-1", min_length=1)
    BEDROCK_MODEL_ID: str = Field(
        default="zai.glm-4.7-flash",
        min_length=1,
    )
    AWS_ACCESS_KEY_ID: SecretStr | None = Field(
        default=None,
        description="Optional local-development AWS access key.",
    )
    AWS_SECRET_ACCESS_KEY: SecretStr | None = Field(
        default=None,
        description="Optional local-development AWS secret access key.",
    )
    AWS_SESSION_TOKEN: SecretStr | None = Field(
        default=None,
        description="Optional token for temporary AWS credentials.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_aws_credential_pair(self) -> "Settings":
        """Require the access key and secret key to be configured together."""

        has_access_key = self.AWS_ACCESS_KEY_ID is not None
        has_secret_key = self.AWS_SECRET_ACCESS_KEY is not None
        if has_access_key != has_secret_key:
            raise ValueError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must both be set"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance.

    Returns:
        Settings: Validated application configuration.
    """

    return Settings()
