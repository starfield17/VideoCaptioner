"""Strict configuration for the Phase 2 LLM boundary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class LlmOptions(BaseModel):
    """Provider and retry settings for one synchronous pipeline run."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["fake", "openai-compatible"] = "fake"
    api_key: SecretStr | None = None
    base_url: str | None = None
    base_url_env: str = Field(default="CAPTIONER_LLM_BASE_URL", min_length=1)
    api_key_env: str = Field(default="CAPTIONER_LLM_API_KEY", min_length=1)
    model: str = Field(default="your-model-name", min_length=1)
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    timeout_seconds: float = Field(default=120, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    backoff_base_seconds: float = Field(default=0.5, ge=0)
    backoff_max_seconds: float = Field(default=8, ge=0)
