"""Faster Whisper provider configuration and Worker client."""

import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from captioner.shared.errors import ProviderUnavailableError
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TranscriptDocument
from captioner.transcription.providers.worker_client import NdjsonWorkerClient
from captioner.transcription.requests import TranscriptionRequest


class FasterWhisperVadConfig(BaseModel):
    """Provider-specific Silero VAD parameters accepted by faster-whisper."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0, le=1)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    speech_pad_ms: int = Field(default=200, ge=0)

    def as_transcribe_parameters(self) -> dict[str, float | int]:
        """Return only the fields understood by faster-whisper's VAD API."""

        return {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        }


class FasterWhisperConfig(BaseModel):
    """Faster Whisper settings kept outside the public ASR request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="large-v3", min_length=1)
    device: str = Field(default="cuda", min_length=1)
    compute_type: str = Field(default="float16", min_length=1)
    beam_size: int = Field(default=5, ge=1)
    vad: FasterWhisperVadConfig = Field(default_factory=FasterWhisperVadConfig)


class FasterWhisperWorkerClient:
    """Core-side client for the separate Faster Whisper Worker environment."""

    def __init__(
        self,
        environment_name: str = "captioner-asr-faster-whisper",
        command: tuple[str, ...] | None = None,
    ) -> None:
        self._environment_name = environment_name
        self._command = command
        self._client: NdjsonWorkerClient | None = None

    def start(self, config: FasterWhisperConfig) -> AsrCapabilities:
        client = self._get_client()
        self._client = client
        return client.start({"config": config.model_dump(mode="json")})

    def probe(self) -> AsrCapabilities:
        client = self._get_client()
        self._client = client
        return client.probe()

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        if self._client is None:
            raise ProviderUnavailableError("Faster Whisper worker is not started")
        return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()
        self._client = None

    def _get_client(self) -> NdjsonWorkerClient:
        if self._client is not None:
            raise ProviderUnavailableError(
                "Faster Whisper worker client is already started"
            )
        command = self._command or self._conda_command()
        return NdjsonWorkerClient(
            command=command,
            expected_provider_id="faster-whisper",
        )

    def _conda_command(self) -> tuple[str, ...]:
        conda = shutil.which("conda")
        if conda is None:
            raise ProviderUnavailableError("Conda is required for Faster Whisper")
        return (
            conda,
            "run",
            "--no-capture-output",
            "-n",
            self._environment_name,
            "python",
            "-m",
            "workers.faster_whisper",
        )


__all__ = [
    "FasterWhisperConfig",
    "FasterWhisperVadConfig",
    "FasterWhisperWorkerClient",
]
