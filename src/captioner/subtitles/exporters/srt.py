"""SRT exporter."""

from pathlib import Path

from captioner.shared.errors import ExportError
from captioner.shared.timecode import format_srt_timestamp
from captioner.subtitles.models import SubtitleCue, SubtitleDocument


def render_srt(document: SubtitleDocument, bilingual: bool = True) -> str:
    """Render a document without changing any timing or domain text."""

    blocks = [
        _render_cue(index, cue, bilingual)
        for index, cue in enumerate(document.cues, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(
    document: SubtitleDocument,
    output_path: Path,
    bilingual: bool = True,
) -> Path:
    """Atomically write an SRT artifact."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            render_srt(document, bilingual=bilingual), encoding="utf-8"
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        raise ExportError(f"could not write SRT: {output_path}") from exc
    return output_path


def _render_cue(index: int, cue: SubtitleCue, bilingual: bool) -> str:
    source_text = cue.corrected_text or cue.source_text
    text_lines = list(_text_lines(source_text))
    if bilingual and cue.translated_text:
        text_lines.extend(_text_lines(cue.translated_text))
    return "\n".join(
        (
            str(index),
            f"{format_srt_timestamp(cue.start_ms)} --> "
            f"{format_srt_timestamp(cue.end_ms)}",
            *text_lines,
        )
    )


def _text_lines(text: str) -> tuple[str, ...]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return tuple(normalized.split("\n")) or ("",)
