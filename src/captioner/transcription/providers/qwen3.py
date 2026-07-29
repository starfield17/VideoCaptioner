"""Qwen3 ASR provider configuration, Worker client, and service adapter."""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from captioner.shared.errors import ProviderUnavailableError, TranscriptionError
from captioner.shared.runtimes import resolve_worker_command
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TranscriptDocument
from captioner.transcription.providers.worker_client import NdjsonWorkerClient
from captioner.transcription.requests import TimestampRequirement, TranscriptionRequest


class Qwen3Config(BaseModel):
    """Qwen3 settings kept out of the provider-independent ASR request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="Qwen/Qwen3-ASR-1.7B", min_length=1)
    model_path: Path | None = None
    device: str = Field(default="cuda:0", min_length=1)
    dtype: str = Field(default="bfloat16", min_length=1)
    forced_aligner_model: str | None = "Qwen/Qwen3-ForcedAligner-0.6B"
    forced_aligner_path: Path | None = None


class _Qwen3Worker(Protocol):
    def start(self, config: Qwen3Config) -> AsrCapabilities:
        """Start and load the Qwen3 Worker."""
        ...

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        """Transcribe one prepared audio asset."""
        ...

    def shutdown(self) -> None:
        """Stop the Qwen3 Worker."""
        ...


class Qwen3WorkerClient:
    """Core-side client for the separate Qwen3 Conda environment."""

    def __init__(
        self,
        environment_name: str = "captioner-asr-qwen3",
        command: tuple[str, ...] | None = None,
    ) -> None:
        self._environment_name = environment_name
        self._command = command
        self._client: NdjsonWorkerClient | None = None

    def start(self, config: Qwen3Config) -> AsrCapabilities:
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
            raise ProviderUnavailableError("Qwen3 ASR Worker is not started")
        return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()
        self._client = None

    def _get_client(self) -> NdjsonWorkerClient:
        if self._client is not None:
            raise ProviderUnavailableError("Qwen3 ASR Worker client is already started")
        command = self._command or resolve_worker_command(
            "qwen3-asr",
            environment_name=self._environment_name,
            worker_module="workers.qwen3",
        )
        return NdjsonWorkerClient(
            command=command,
            expected_provider_id="qwen3-asr",
        )


class Qwen3TranscriptionService:
    """Use one configured Qwen3 Worker for a serial run."""

    def __init__(
        self,
        config: Qwen3Config,
        timestamps: TimestampRequirement = TimestampRequirement.REQUIRED,
        client: _Qwen3Worker | None = None,
    ) -> None:
        self._config = config
        self._timestamps = timestamps
        self._client = client or Qwen3WorkerClient()
        self._started = False

    def start(self, model_name: str = "qwen3-asr") -> AsrCapabilities:
        del model_name
        if self._started:
            raise ProviderUnavailableError(
                "Qwen3 transcription service is already started"
            )
        if (
            self._timestamps is TimestampRequirement.REQUIRED
            and self._config.forced_aligner_model is None
        ):
            raise ProviderUnavailableError(
                "Qwen3 requires forced_aligner_model when word timestamps are required"
            )
        capabilities = self._client.start(self._config)
        if self._timestamps is TimestampRequirement.REQUIRED and not (
            capabilities.native_word_timestamps or capabilities.forced_alignment
        ):
            self._client.shutdown()
            raise ProviderUnavailableError(
                "Qwen3 Worker cannot provide required word timestamps"
            )
        self._started = True
        return capabilities

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        if not self._started:
            raise ProviderUnavailableError("Qwen3 transcription service is not started")
        if (
            request.timestamps is TimestampRequirement.REQUIRED
            and self._config.forced_aligner_model is None
        ):
            raise TranscriptionError(
                "Qwen3 cannot satisfy required word timestamps without a forced aligner"
            )
        return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        self._client.shutdown()
        self._started = False


__all__ = ["Qwen3Config", "Qwen3TranscriptionService", "Qwen3WorkerClient"]
