"""Immutable subtitle documents and deterministic quality reports."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from captioner.subtitles.glossary import Glossary, GlossaryEntry

_REPAIRABLE_CODES = {
    "empty_translation",
    "glossary_missing",
    "number_missing",
    "protected_content_missing",
    "target_language_unexpected",
    "translation_missing",
    "translation_too_long",
    "translation_unchanged",
}


class SubtitleCue(BaseModel):
    """One stable subtitle cue with independent text variants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    corrected_text: str | None = None
    translated_text: str | None = None
    source_word_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def has_positive_duration(self) -> Self:
        if self.start_ms >= self.end_ms:
            raise ValueError("SubtitleCue requires start_ms < end_ms")
        for field_name in ("corrected_text", "translated_text"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} cannot be blank")
        return self


class SubtitleDocument(BaseModel):
    """The immutable document passed between subtitle stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["subtitle.v1"] = "subtitle.v1"
    source_language: str = Field(min_length=1)
    target_language: str | None = None
    cues: tuple[SubtitleCue, ...] = ()

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        cue_ids = [cue.id for cue in self.cues]
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("SubtitleDocument cue IDs must be unique")
        for previous, current in zip(self.cues, self.cues[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ValueError("Subtitle cues must be chronologically ordered")
            if current.start_ms < previous.end_ms:
                raise ValueError("Subtitle cues cannot overlap")
        return self


class QualitySeverity(StrEnum):
    """Severity used by deterministic subtitle QC."""

    WARNING = "warning"
    ERROR = "error"


class QualityIssue(BaseModel):
    """One deterministic QC finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    severity: QualitySeverity
    cue_id: str | None = None
    message: str = Field(min_length=1)


class QualityReport(BaseModel):
    """Stable QC output used by the pipeline and exported artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issues: tuple[QualityIssue, ...] = ()

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is QualitySeverity.ERROR for issue in self.issues)

    @property
    def has_repairable_issues(self) -> bool:
        return any(
            issue.code in _REPAIRABLE_CODES and issue.cue_id is not None
            for issue in self.issues
        )

    @property
    def repairable_cue_ids(self) -> tuple[str, ...]:
        """Return stable cue IDs eligible for the single Repair stage."""

        return tuple(
            issue.cue_id
            for issue in self.issues
            if issue.code in _REPAIRABLE_CODES and issue.cue_id is not None
        )


__all__ = [
    "Glossary",
    "GlossaryEntry",
    "QualityIssue",
    "QualityReport",
    "QualitySeverity",
    "SubtitleCue",
    "SubtitleDocument",
]
