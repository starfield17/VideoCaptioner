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


class LlmGlossaryEntry(BaseModel):
    """One terminology hint safe to send to a cloud LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    target: str | None = None
    note: str | None = None


class ContentContext(BaseModel):
    """One immutable file-level analysis shared by later LLM stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str = ""
    domain: str | None = None
    tone: str | None = None
    entities: tuple[str, ...] = ()
    glossary: tuple[LlmGlossaryEntry, ...] = ()


class LlmStageContext(BaseModel):
    """Read-only context for one segmentation or text-update request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: ContentContext = Field(default_factory=ContentContext)
    glossary: tuple[LlmGlossaryEntry, ...] = ()
    before: tuple[LlmTextItem, ...] = ()
    after: tuple[LlmTextItem, ...] = ()
    style_rules: tuple[str, ...] = ()
    max_chars_cjk: int = Field(default=24, ge=1)
    max_words_latin: int = Field(default=14, ge=1)


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


class LlmTextBatch(BaseModel):
    """One immutable owner batch and its read-only neighboring context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[LlmTextItem, ...]
    context: LlmStageContext = Field(default_factory=LlmStageContext)
