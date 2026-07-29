"""FFmpeg-backed audio preparation for the real ASR provider."""

import subprocess
from pathlib import Path

from captioner.media.models import AudioAsset
from captioner.shared.app_paths import bundled_executable
from captioner.shared.errors import MediaPreparationError


class FfmpegMediaService:
    """Extract and normalize the first audio stream serially."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        ffmpeg_binary: str | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._ffmpeg = ffmpeg_binary or bundled_executable("ffmpeg")

    def prepare_audio(self, input_path: Path, output_dir: Path) -> AudioAsset:
        if not input_path.is_file():
            raise MediaPreparationError(f"input media does not exist: {input_path}")
        if self._ffmpeg is None:
            raise MediaPreparationError("FFmpeg is not available")
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = output_dir / "prepared.wav"
        temporary_path = output_dir / "prepared.tmp.wav"
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            str(self._channels),
            "-ar",
            str(self._sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(temporary_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise MediaPreparationError(f"could not execute FFmpeg: {exc}") from exc
        if completed.returncode != 0 or not temporary_path.is_file():
            detail = completed.stderr.strip() or "unknown FFmpeg error"
            raise MediaPreparationError(
                f"FFmpeg audio preparation failed for {input_path}: {detail[-1_000:]}"
            )
        try:
            temporary_path.replace(prepared_path)
        except OSError as exc:
            raise MediaPreparationError(
                f"could not finalize prepared audio: {prepared_path}"
            ) from exc
        return AudioAsset(
            source_path=input_path,
            path=prepared_path,
            sample_rate=self._sample_rate,
            channels=self._channels,
        )
