"""Strict pipeline configuration models."""

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from captioner.llm.config import LlmOptions
from captioner.media.voice_separation import VoiceSeparationOptions
from captioner.shared.errors import ConfigurationError
from captioner.subtitles.glossary import Glossary
from captioner.subtitles.quality import QualityOptions
from captioner.transcription.api import FasterWhisperConfig, NemoConfig, Qwen3Config
from captioner.transcription.requests import TimestampRequirement


class OutputFormat(StrEnum):
    """Formats implemented by the current pipeline."""

    SRT = "srt"
    BILINGUAL_SRT = "bilingual_srt"
    VTT = "vtt"
    JSON = "json"


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continue_on_error: bool = True
    keep_workdir: bool = False


class AudioOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(default=16_000, gt=0)
    channels: int = Field(default=1, gt=0)
    voice_separation: VoiceSeparationOptions = Field(
        default_factory=VoiceSeparationOptions
    )


class ContextAnalysisOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_characters: int = Field(default=50_000, ge=1)


class CleanupOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fillers: tuple[str, ...] = ()
    non_speech_markers: tuple[str, ...] = ()
    collapse_repetitions: bool = False


class FakeAsrOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["fake"] = "fake"
    language: str = "auto"
    initial_prompt: str = ""
    timestamps: TimestampRequirement = TimestampRequirement.REQUIRED


class FasterWhisperAsrOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["faster-whisper"]
    language: str = "auto"
    initial_prompt: str = ""
    timestamps: TimestampRequirement = TimestampRequirement.REQUIRED
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)


class Qwen3AsrOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["qwen3-asr"]
    language: str = "auto"
    initial_prompt: str = ""
    timestamps: TimestampRequirement = TimestampRequirement.REQUIRED
    qwen3: Qwen3Config = Field(default_factory=Qwen3Config)


class NemoAsrOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["nemo-asr"]
    language: str = "auto"
    initial_prompt: str = ""
    timestamps: TimestampRequirement = TimestampRequirement.REQUIRED
    nemo: NemoConfig = Field(default_factory=NemoConfig)


type AsrOptions = Annotated[
    FakeAsrOptions | FasterWhisperAsrOptions | Qwen3AsrOptions | NemoAsrOptions,
    Field(discriminator="provider"),
]


class SegmentationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    batch_tokens: int = Field(default=800, ge=1)
    overlap_tokens: int = Field(default=0, ge=0)
    silence_boundary_ms: int = Field(default=700, ge=0)
    parallelism: int = Field(default=4, ge=1, le=100)

    @model_validator(mode="after")
    def overlap_is_smaller_than_owner_batch(self) -> Self:
        if self.overlap_tokens >= self.batch_tokens:
            raise ValueError("overlap_tokens must be smaller than batch_tokens")
        return self


class CorrectionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    batch_size: int = Field(default=30, ge=1)
    parallelism: int = Field(default=8, ge=1, le=100)
    max_change_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class TranslationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    target_language: str = Field(default="en", min_length=1)
    batch_size: int = Field(default=30, ge=1)
    parallelism: int = Field(default=16, ge=1, le=100)
    allow_partial: bool = True


class RepairOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    batch_size: int = Field(default=20, ge=1)
    parallelism: int = Field(default=8, ge=1, le=100)


class OutputOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formats: tuple[OutputFormat, ...] = (OutputFormat.SRT, OutputFormat.JSON)
    bilingual: bool = True
    overwrite: Literal["error", "replace"] = "error"

    @field_validator("formats")
    @classmethod
    def require_supported_format(
        cls, value: tuple[OutputFormat, ...]
    ) -> tuple[OutputFormat, ...]:
        if not value:
            raise ValueError("at least one output format is required")
        if len(value) != len(set(value)):
            raise ValueError("output formats must be unique")
        return value


class PipelineOptions(BaseModel):
    """All pipeline options, with unknown fields rejected."""

    model_config = ConfigDict(extra="forbid")

    run: RunOptions = Field(default_factory=RunOptions)
    audio: AudioOptions = Field(default_factory=AudioOptions)
    asr: AsrOptions = Field(default_factory=FakeAsrOptions)
    context_analysis: ContextAnalysisOptions = Field(
        default_factory=ContextAnalysisOptions
    )
    segmentation: SegmentationOptions = Field(default_factory=SegmentationOptions)
    correction: CorrectionOptions = Field(default_factory=CorrectionOptions)
    cleanup: CleanupOptions = Field(default_factory=CleanupOptions)
    translation: TranslationOptions = Field(default_factory=TranslationOptions)
    repair: RepairOptions = Field(default_factory=RepairOptions)
    llm: LlmOptions = Field(default_factory=LlmOptions)
    subtitle: QualityOptions = Field(default_factory=QualityOptions)
    glossary: Glossary = Field(default_factory=Glossary)
    output: OutputOptions = Field(default_factory=OutputOptions)

    @model_validator(mode="before")
    @classmethod
    def default_fake_provider(cls, value: object) -> object:
        """Keep partial legacy Phase 0 configs defaulting to Fake ASR."""

        if not isinstance(value, dict):
            return value
        raw = dict(cast(dict[str, object], value))
        raw_asr = raw.get("asr")
        if isinstance(raw_asr, dict) and "provider" not in raw_asr:
            asr = dict(cast(dict[str, object], raw_asr))
            asr["provider"] = "fake"
            raw["asr"] = asr
        return raw

    @classmethod
    def validate_for_phase0(cls, options: Self) -> Self:
        if not options.segmentation.enabled:
            raise ConfigurationError("the pipeline requires segmentation to be enabled")
        if options.asr.timestamps is TimestampRequirement.DISABLED:
            raise ConfigurationError("the pipeline requires transcript timestamps")
        return options


def load_options(config_path: Path | None = None) -> PipelineOptions:
    """Load strict TOML options or return the documented defaults."""

    if config_path is None:
        return PipelineOptions()
    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
        options = PipelineOptions.model_validate(raw_config)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid configuration: {config_path}") from exc
    return PipelineOptions.validate_for_phase0(options)


def with_keep_workdir(options: PipelineOptions, keep_workdir: bool) -> PipelineOptions:
    """Apply the one explicit CLI lifecycle override."""

    return options.model_copy(
        update={"run": options.run.model_copy(update={"keep_workdir": keep_workdir})}
    )
