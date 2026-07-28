"""Immutable provider-independent transcript models."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimingOrigin(StrEnum):
    """The authoritative source of timestamps in a transcript."""

    ASR_NATIVE = "asr_native"
    FORCED_ALIGNMENT = "forced_alignment"
    SEGMENT_NATIVE = "segment_native"


class TimedWord(BaseModel):
    """A word whose timing came from ASR or a forced aligner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def has_positive_duration(self) -> Self:
        if self.start_ms >= self.end_ms:
            raise ValueError("TimedWord requires start_ms < end_ms")
        return self


class TranscriptSegment(BaseModel):
    """A provider-native transcript segment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    word_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def has_positive_duration(self) -> Self:
        if self.start_ms >= self.end_ms:
            raise ValueError("TranscriptSegment requires start_ms < end_ms")
        return self


class TranscriptDocument(BaseModel):
    """The stable transcript contract consumed by subtitle processing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["transcript.v1"] = "transcript.v1"
    language: str = Field(min_length=1)
    text: str = Field(min_length=1)
    timing_origin: TimingOrigin
    words: tuple[TimedWord, ...] = ()
    segments: tuple[TranscriptSegment, ...] = ()
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        word_ids = [word.id for word in self.words]
        segment_ids = [segment.id for segment in self.segments]
        if len(word_ids) != len(set(word_ids)):
            raise ValueError("TranscriptDocument word IDs must be unique")
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("TranscriptDocument segment IDs must be unique")

        if self.timing_origin is TimingOrigin.SEGMENT_NATIVE and self.words:
            raise ValueError("SEGMENT_NATIVE transcripts cannot contain words")
        if self.words and self.timing_origin not in {
            TimingOrigin.ASR_NATIVE,
            TimingOrigin.FORCED_ALIGNMENT,
        }:
            raise ValueError("word timestamps require an ASR or aligner origin")

        for previous, current in zip(self.words, self.words[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ValueError("Transcript word timestamps must be monotonic")
            if current.end_ms < previous.end_ms:
                raise ValueError("Transcript word end times must be monotonic")

        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ValueError("Transcript segment timestamps must be monotonic")
            if current.end_ms < previous.end_ms:
                raise ValueError("Transcript segment end times must be monotonic")

        words_by_id = {word.id: word for word in self.words}
        for segment in self.segments:
            if len(segment.word_ids) != len(set(segment.word_ids)):
                raise ValueError("Segment word IDs must be unique")
            if self.words and not segment.word_ids:
                raise ValueError("word-timed segments must reference their words")
            referenced = [words_by_id.get(word_id) for word_id in segment.word_ids]
            if any(word is None for word in referenced):
                raise ValueError("Segment references an unknown word ID")
            concrete_words = [word for word in referenced if word is not None]
            if concrete_words:
                if concrete_words[0].start_ms < segment.start_ms:
                    raise ValueError("Segment does not cover its first word")
                if concrete_words[-1].end_ms > segment.end_ms:
                    raise ValueError("Segment does not cover its last word")
        return self
