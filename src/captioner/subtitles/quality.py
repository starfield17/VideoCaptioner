"""Deterministic subtitle readability and completeness checks."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from captioner.subtitles.models import (
    QualityIssue,
    QualityReport,
    QualitySeverity,
    SubtitleCue,
    SubtitleDocument,
)


class QualityOptions(BaseModel):
    """Stable thresholds for subtitle readability QC."""

    model_config = ConfigDict(extra="forbid")

    min_duration_ms: int = Field(default=700, ge=0)
    max_duration_ms: int = Field(default=7_000, gt=0)
    max_cps: float = Field(default=17.0, gt=0)
    max_line_chars: int = Field(default=42, ge=1)
    max_lines: int = Field(default=2, ge=1)
    max_chars_cjk: int = Field(default=24, ge=1)
    max_words_latin: int = Field(default=14, ge=1)

    @model_validator(mode="after")
    def has_valid_duration_range(self) -> Self:
        if self.max_duration_ms < self.min_duration_ms:
            raise ValueError("max_duration_ms must be >= min_duration_ms")
        return self


def check_quality(
    document: SubtitleDocument,
    options: QualityOptions | None = None,
) -> QualityReport:
    """Return stable issues for text, timing, overlap, CPS, and line layout."""

    policy = options or QualityOptions()
    issues: list[QualityIssue] = []
    previous_end: int | None = None
    previous_id: str | None = None
    for cue in document.cues:
        duration_ms = cue.end_ms - cue.start_ms
        if not cue.source_text.strip():
            issues.append(
                QualityIssue(
                    code="empty_source",
                    severity=QualitySeverity.ERROR,
                    cue_id=cue.id,
                    message="source text is empty",
                )
            )
        if cue.corrected_text is not None and not cue.corrected_text.strip():
            issues.append(
                QualityIssue(
                    code="empty_correction",
                    severity=QualitySeverity.ERROR,
                    cue_id=cue.id,
                    message="corrected text is empty",
                )
            )
        if cue.translated_text is not None and not cue.translated_text.strip():
            issues.append(
                QualityIssue(
                    code="empty_translation",
                    severity=QualitySeverity.ERROR,
                    cue_id=cue.id,
                    message="translated text is empty",
                )
            )
        if document.target_language and cue.translated_text is None:
            issues.append(
                QualityIssue(
                    code="translation_missing",
                    severity=QualitySeverity.WARNING,
                    cue_id=cue.id,
                    message="translation is missing",
                )
            )
        if duration_ms <= 0:
            issues.append(
                QualityIssue(
                    code="invalid_duration",
                    severity=QualitySeverity.ERROR,
                    cue_id=cue.id,
                    message="cue duration must be positive",
                )
            )
        else:
            if duration_ms < policy.min_duration_ms:
                issues.append(
                    QualityIssue(
                        code="duration_too_short",
                        severity=QualitySeverity.WARNING,
                        cue_id=cue.id,
                        message=f"duration is below {policy.min_duration_ms} ms",
                    )
                )
            if duration_ms > policy.max_duration_ms:
                issues.append(
                    QualityIssue(
                        code="duration_too_long",
                        severity=QualitySeverity.WARNING,
                        cue_id=cue.id,
                        message=f"duration exceeds {policy.max_duration_ms} ms",
                    )
                )
        if previous_end is not None and cue.start_ms < previous_end:
            issues.append(
                QualityIssue(
                    code="overlap",
                    severity=QualitySeverity.ERROR,
                    cue_id=cue.id,
                    message=f"cue overlaps previous cue {previous_id}",
                )
            )
        for label, text in _text_variants(cue):
            lines = _lines(text)
            if len(lines) > policy.max_lines:
                issues.append(
                    QualityIssue(
                        code="line_count_exceeded",
                        severity=QualitySeverity.WARNING,
                        cue_id=cue.id,
                        message=(
                            f"{label} has {len(lines)} lines; "
                            f"maximum is {policy.max_lines}"
                        ),
                    )
                )
            longest_line = max((len(line) for line in lines), default=0)
            if longest_line > policy.max_line_chars:
                issues.append(
                    QualityIssue(
                        code="line_too_long",
                        severity=QualitySeverity.WARNING,
                        cue_id=cue.id,
                        message=(
                            f"{label} line has {longest_line} characters; "
                            f"maximum is {policy.max_line_chars}"
                        ),
                    )
                )
            for line in lines:
                cjk_count = sum(1 for character in line if _is_cjk(character))
                if cjk_count > policy.max_chars_cjk:
                    issues.append(
                        QualityIssue(
                            code="cjk_line_too_long",
                            severity=QualitySeverity.WARNING,
                            cue_id=cue.id,
                            message=(
                                f"{label} line has {cjk_count} CJK characters; "
                                f"maximum is {policy.max_chars_cjk}"
                            ),
                        )
                    )
                latin_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", line))
                if latin_words > policy.max_words_latin:
                    issues.append(
                        QualityIssue(
                            code="latin_words_exceeded",
                            severity=QualitySeverity.WARNING,
                            cue_id=cue.id,
                            message=(
                                f"{label} line has {latin_words} Latin words; "
                                f"maximum is {policy.max_words_latin}"
                            ),
                        )
                    )
            if duration_ms > 0:
                characters = len("".join(lines))
                cps = characters / (duration_ms / 1_000)
                if cps > policy.max_cps:
                    issues.append(
                        QualityIssue(
                            code="cps_exceeded",
                            severity=QualitySeverity.WARNING,
                            cue_id=cue.id,
                            message=(
                                f"{label} CPS is {cps:.2f}; "
                                f"maximum is {policy.max_cps:.2f}"
                            ),
                        )
                    )
        previous_end = cue.end_ms
        previous_id = cue.id
    return QualityReport(issues=tuple(issues))


def _text_variants(cue: SubtitleCue) -> tuple[tuple[str, str], ...]:
    source_text = cue.source_text
    corrected_text = cue.corrected_text
    translated_text = cue.translated_text
    variants: list[tuple[str, str]] = [("source", source_text)]
    if corrected_text is not None:
        variants.append(("corrected", corrected_text))
    if translated_text is not None:
        variants.append(("translated", translated_text))
    return tuple(variants)


def _lines(text: str) -> tuple[str, ...]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return tuple(normalized.split("\n")) or ("",)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3000 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


__all__ = ["QualityOptions", "check_quality"]
