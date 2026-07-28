"""Strict, synchronous SRT importer for the refine workflow."""

import re
from pathlib import Path

from captioner.shared.errors import SubtitleValidationError
from captioner.shared.ids import make_cue_id
from captioner.subtitles.models import SubtitleCue, SubtitleDocument

_TIMESTAMP = re.compile(
    r"^(?P<start>\d{2,}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2,}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def read_srt(
    input_path: Path,
    *,
    source_language: str = "und",
    bilingual: bool = False,
) -> SubtitleDocument:
    """Read one existing SRT into the canonical immutable subtitle model."""

    try:
        text = input_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SubtitleValidationError(f"could not read SRT: {input_path}") from exc
    return parse_srt(
        text,
        source_language=source_language,
        bilingual=bilingual,
    )


def parse_srt(
    text: str,
    *,
    source_language: str = "und",
    bilingual: bool = False,
) -> SubtitleDocument:
    """Parse SRT blocks while preserving cue order and line breaks."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = tuple(block for block in re.split(r"\n{2,}", normalized) if block.strip())
    if not blocks:
        raise SubtitleValidationError("SRT contains no subtitle cues")
    cues: list[SubtitleCue] = []
    try:
        for index, block in enumerate(blocks, start=1):
            lines = block.split("\n")
            timestamp_index = 1 if lines and lines[0].strip().isdigit() else 0
            if timestamp_index >= len(lines):
                raise SubtitleValidationError("SRT cue is missing its timestamp")
            match = _TIMESTAMP.match(lines[timestamp_index].strip())
            if match is None:
                raise SubtitleValidationError("SRT cue has an invalid timestamp")
            body = "\n".join(lines[timestamp_index + 1 :]).strip()
            if not body:
                raise SubtitleValidationError("SRT cue has empty text")
            source_text = body
            translated_text: str | None = None
            if bilingual and "\n" in body:
                source_text, translated_text = body.split("\n", maxsplit=1)
            cues.append(
                SubtitleCue(
                    id=make_cue_id(index),
                    start_ms=_parse_timestamp(match["start"]),
                    end_ms=_parse_timestamp(match["end"]),
                    source_text=source_text,
                    translated_text=translated_text,
                )
            )
        return SubtitleDocument(
            source_language=source_language,
            target_language=None,
            cues=tuple(cues),
        )
    except SubtitleValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SubtitleValidationError(
            "SRT violates subtitle timing invariants"
        ) from exc


def _parse_timestamp(value: str) -> int:
    hours, minutes, seconds_millis = value.replace(",", ".").split(":")
    seconds, millis = seconds_millis.split(".")
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError("invalid SRT timestamp range")
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1_000
        + int(millis)
    )


__all__ = ["parse_srt", "read_srt"]
