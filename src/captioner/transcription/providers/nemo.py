"""NVIDIA NeMo provider configuration, Worker client, and service adapter."""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from captioner.shared.errors import ProviderUnavailableError, TranscriptionError
from captioner.shared.runtimes import resolve_worker_command
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TranscriptDocument
from captioner.transcription.providers.worker_client import NdjsonWorkerClient
from captioner.transcription.requests import TranscriptionRequest


class NemoConfig(BaseModel):
    """NeMo settings kept outside the provider-independent ASR request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="nvidia/parakeet-tdt-0.6b-v3", min_length=1)
    model_path: Path | None = None
    device: str = Field(default="auto", min_length=1)
    batch_size: int = Field(default=1, ge=1)


class _NemoWorker(Protocol):
    def start(self, config: NemoConfig) -> AsrCapabilities:
        """Start and load the NeMo Worker."""
        ...

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        """Transcribe one prepared audio asset."""
        ...

    def shutdown(self) -> None:
        """Stop the NeMo Worker."""
        ...


class NemoWorkerClient:
    """Core-side client for the separate NeMo Conda environment."""

    def __init__(
        self,
        environment_name: str = "captioner-asr-nemo",
        command: tuple[str, ...] | None = None,
    ) -> None:
        self._environment_name = environment_name
        self._command = command
        self._client: NdjsonWorkerClient | None = None

    def start(self, config: NemoConfig) -> AsrCapabilities:
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
            raise ProviderUnavailableError("NeMo ASR Worker is not started")
        return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()
        self._client = None

    def _get_client(self) -> NdjsonWorkerClient:
        if self._client is not None:
            raise ProviderUnavailableError("NeMo ASR Worker client is already started")
        command = self._command or resolve_worker_command(
            "nemo-asr",
            environment_name=self._environment_name,
            worker_module="workers.nemo",
        )
        return NdjsonWorkerClient(
            command=command,
            expected_provider_id="nemo-asr",
        )


class NemoTranscriptionService:
    """Use one configured NeMo Worker for a serial run."""

    def __init__(
        self,
        config: NemoConfig,
        client: _NemoWorker | None = None,
    ) -> None:
        self._config = config
        self._client = client or NemoWorkerClient()
        self._started = False

    def start(self, model_name: str = "nemo-asr") -> AsrCapabilities:
        del model_name
        if self._started:
            raise ProviderUnavailableError(
                "NeMo transcription service is already started"
            )
        capabilities = self._client.start(self._config)
        if not capabilities.native_word_timestamps:
            self._client.shutdown()
            raise ProviderUnavailableError(
                "NeMo Worker does not provide native word timestamps"
            )
        self._started = True
        return capabilities

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        if not self._started:
            raise ProviderUnavailableError("NeMo transcription service is not started")
        if request.initial_prompt:
            raise TranscriptionError("NeMo Parakeet does not support initial_prompt")
        return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        self._client.shutdown()
        self._started = False


__all__ = ["NemoConfig", "NemoTranscriptionService", "NemoWorkerClient"]
