"""Provider-independent transcription requests."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class TimestampRequirement(StrEnum):
    """The timing granularity requested by the application."""

    REQUIRED = "required"
    PREFERRED = "preferred"
    DISABLED = "disabled"


class TranscriptionRequest(BaseModel):
    """Small public request that contains no provider-specific parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio_path: Path
    language: str | None = None
    initial_prompt: str | None = None
    timestamps: TimestampRequirement = TimestampRequirement.REQUIRED
