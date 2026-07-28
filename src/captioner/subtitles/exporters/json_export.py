"""Subtitle JSON exporter."""

from pathlib import Path

from captioner.shared.errors import ExportError
from captioner.subtitles.models import SubtitleDocument


def write_json(
    document: SubtitleDocument,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write the canonical subtitle.v1 JSON document."""

    try:
        if output_path.exists() and not overwrite:
            raise ExportError(f"output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            document.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        raise ExportError(f"could not write subtitle JSON: {output_path}") from exc
    return output_path
