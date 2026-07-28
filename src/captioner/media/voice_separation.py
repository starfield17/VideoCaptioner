"""Optional synchronous voice-separation adapter at the media boundary."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from captioner.media.models import AudioAsset
from captioner.shared.errors import CaptionerError


class VoiceSeparationError(CaptionerError):
    """A configured voice-separation adapter could not produce its output."""


class VoiceSeparationOptions(BaseModel):
    """User-facing options for the optional media-only separation step."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    required: bool = False
    provider: Literal["mdx", "command"] = "mdx"
    command_env: str = Field(default="CAPTIONER_VOICE_SEPARATION_COMMAND", min_length=1)


class VoiceSeparator(Protocol):
    """Separate vocals without changing the ASR or Worker contracts."""

    def separate(self, audio: AudioAsset, output_path: Path) -> AudioAsset:
        """Write a new vocal-only asset and return its media metadata."""
        ...


class CommandVoiceSeparator:
    """Run a configured synchronous separator command.

    The command receives the prepared audio path followed by the requested
    output path. The adapter intentionally has no knowledge of ASR or subtitle
    objects; an MDX-compatible wrapper can be supplied through the environment.
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        command_env: str = "CAPTIONER_VOICE_SEPARATION_COMMAND",
    ) -> None:
        if command is None:
            raw_command = os.getenv(command_env, "")
            command = tuple(shlex.split(raw_command))
        self._command = tuple(command)

    def separate(self, audio: AudioAsset, output_path: Path) -> AudioAsset:
        if not self._command:
            raise VoiceSeparationError(
                "voice-separation command is not configured; "
                "set CAPTIONER_VOICE_SEPARATION_COMMAND"
            )
        if output_path.resolve() == audio.path.resolve():
            raise VoiceSeparationError(
                "voice-separation output must not overwrite prepared audio"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f".tmp{output_path.suffix}")
        command = (*self._command, str(audio.path), str(temporary_path))
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise VoiceSeparationError(
                f"could not execute voice-separation command: {exc}"
            ) from exc
        if completed.returncode != 0 or not temporary_path.is_file():
            detail = completed.stderr.strip() or "unknown separator error"
            raise VoiceSeparationError(f"voice separation failed: {detail[-1_000:]}")
        try:
            temporary_path.replace(output_path)
        except OSError as exc:
            raise VoiceSeparationError(
                f"could not finalize separated audio: {output_path}"
            ) from exc
        return AudioAsset(
            source_path=audio.source_path,
            path=output_path,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
        )


__all__ = [
    "CommandVoiceSeparator",
    "VoiceSeparationError",
    "VoiceSeparationOptions",
    "VoiceSeparator",
]
