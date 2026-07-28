"""Structured, timing-free LLM request and response models."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LlmToken(BaseModel):
    """Read-only token input for semantic segmentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    gap_after_ms: int = Field(ge=0)


class BoundarySelection(BaseModel):
    """Structured segmentation output containing IDs only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    break_after: tuple[str, ...]

    @model_validator(mode="after")
    def has_unique_ids(self) -> Self:
        if any(not value for value in self.break_after):
            raise ValueError("segmentation boundary IDs cannot be blank")
        if len(self.break_after) != len(set(self.break_after)):
            raise ValueError("segmentation boundary IDs must be unique")
        return self


class LlmTextItem(BaseModel):
    """Text input identified by a stable subtitle or batch-local ID."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class TextUpdate(BaseModel):
    """One structured text-only LLM result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class TextUpdateBatch(BaseModel):
    """A complete ID-keyed correction or translation response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[TextUpdate, ...]

    @model_validator(mode="after")
    def has_unique_ids(self) -> Self:
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("LLM update IDs must be unique")
        return self
