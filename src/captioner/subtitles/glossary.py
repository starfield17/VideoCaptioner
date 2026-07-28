"""Deterministic glossary terms applied at the subtitle boundary."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GlossaryEntry(BaseModel):
    """One source phrase and its preferred rendered term."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    target: str | None = None
    note: str | None = None


class Glossary(BaseModel):
    """Ordered, immutable glossary used by translation and repair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[GlossaryEntry, ...] = ()

    @model_validator(mode="after")
    def has_unique_sources(self) -> Self:
        sources = [entry.source for entry in self.entries]
        if len(sources) != len(set(sources)):
            raise ValueError("glossary source terms must be unique")
        return self

    def apply(self, text: str) -> str:
        """Apply entries in configuration order without changing line breaks."""

        result = text
        for entry in self.entries:
            if entry.target is not None:
                result = result.replace(entry.source, entry.target)
        return result


__all__ = ["Glossary", "GlossaryEntry"]
