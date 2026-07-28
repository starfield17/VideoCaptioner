"""ASR capability declarations."""

from pydantic import BaseModel, ConfigDict


class AsrCapabilities(BaseModel):
    """Capabilities reported by an ASR worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    native_word_timestamps: bool
    forced_alignment: bool
    language_detection: bool
    initial_prompt: bool
    internal_vad: bool
    supported_languages: tuple[str, ...] | None
