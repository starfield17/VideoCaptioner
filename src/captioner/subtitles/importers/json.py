"""Canonical subtitle JSON importer."""

from pathlib import Path

from pydantic import ValidationError

from captioner.shared.errors import SubtitleValidationError
from captioner.subtitles.models import SubtitleDocument


def read_json(input_path: Path) -> SubtitleDocument:
    """Read and validate a subtitle.v1 JSON document."""

    try:
        return SubtitleDocument.model_validate_json(input_path.read_text("utf-8"))
    except (OSError, ValidationError) as exc:
        raise SubtitleValidationError(
            f"could not read subtitle JSON: {input_path}"
        ) from exc


__all__ = ["read_json"]
