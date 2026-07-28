"""WebVTT exporter with deterministic cue order and bilingual line order."""

from pathlib import Path

from captioner.shared.errors import ExportError
from captioner.subtitles.models import SubtitleCue, SubtitleDocument


def render_vtt(document: SubtitleDocument, bilingual: bool = True) -> str:
    """Render WebVTT without changing cue timing or business text."""

    blocks = ["WEBVTT"]
    blocks.extend(
        _render_cue(index, cue, bilingual)
        for index, cue in enumerate(document.cues, start=1)
    )
    return "\n\n".join(blocks) + "\n"


def write_vtt(
    document: SubtitleDocument,
    output_path: Path,
    bilingual: bool = True,
) -> Path:
    """Atomically write a WebVTT artifact."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            render_vtt(document, bilingual=bilingual), encoding="utf-8"
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        raise ExportError(f"could not write VTT: {output_path}") from exc
    return output_path


def _render_cue(index: int, cue: SubtitleCue, bilingual: bool) -> str:
    source_text = cue.corrected_text or cue.source_text
    text_lines = _text_lines(source_text)
    if bilingual and cue.translated_text:
        text_lines += _text_lines(cue.translated_text)
    return "\n".join(
        (
            str(index),
            f"{_format_vtt_timestamp(cue.start_ms)} --> "
            f"{_format_vtt_timestamp(cue.end_ms)}",
            *text_lines,
        )
    )


def _text_lines(text: str) -> tuple[str, ...]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return tuple(normalized.split("\n")) or ("",)


def _format_vtt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("VTT timestamps cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


__all__ = ["render_vtt", "write_vtt"]
