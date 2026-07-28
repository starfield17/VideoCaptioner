"""Public media preparation API and optional voice-separation ports."""

import shutil
from pathlib import Path
from typing import Protocol

from captioner.media.ffmpeg_adapter import FfmpegMediaService
from captioner.media.models import AudioAsset
from captioner.media.voice_separation import (
    CommandVoiceSeparator,
    VoiceSeparationError,
    VoiceSeparationOptions,
    VoiceSeparator,
)
from captioner.shared.errors import MediaPreparationError


class MediaService(Protocol):
    """Prepare one input into an ASR-consumable asset."""

    def prepare_audio(self, input_path: Path, output_dir: Path) -> AudioAsset:
        """Prepare a media input in the supplied per-file workspace."""
        ...


class FakeMediaService:
    """Copy a JSON fixture into the temporary workspace."""

    def __init__(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self._sample_rate = sample_rate
        self._channels = channels

    def prepare_audio(self, input_path: Path, output_dir: Path) -> AudioAsset:
        if not input_path.is_file():
            raise MediaPreparationError(f"input fixture does not exist: {input_path}")
        if input_path.suffix.lower() != ".json":
            raise MediaPreparationError(
                f"Phase 0 Fake Media accepts JSON fixtures only: {input_path.name}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = output_dir / "prepared.fake.json"
        temporary_path = output_dir / "prepared.fake.json.tmp"
        try:
            shutil.copyfile(input_path, temporary_path)
            temporary_path.replace(prepared_path)
        except OSError as exc:
            raise MediaPreparationError(
                f"could not prepare fixture {input_path}: {exc}"
            ) from exc
        return AudioAsset(
            source_path=input_path,
            path=prepared_path,
            sample_rate=self._sample_rate,
            channels=self._channels,
        )


__all__ = [
    "AudioAsset",
    "CommandVoiceSeparator",
    "FakeMediaService",
    "FfmpegMediaService",
    "MediaService",
    "VoiceSeparationError",
    "VoiceSeparationOptions",
    "VoiceSeparator",
]
