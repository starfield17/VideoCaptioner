"""Synchronous provider transcription services."""

import logging
from pathlib import Path
from typing import Protocol

from captioner.shared.errors import ProviderUnavailableError, TranscriptionError
from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import TranscriptDocument
from captioner.transcription.providers.faster_whisper import (
    FasterWhisperConfig,
    FasterWhisperWorkerClient,
)
from captioner.transcription.providers.worker_client import FakeWorkerClient
from captioner.transcription.requests import TranscriptionRequest


class _FasterWhisperWorker(Protocol):
    def start(self, config: FasterWhisperConfig) -> AsrCapabilities:
        """Start and load the provider Worker."""
        ...

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        """Transcribe one prepared audio asset."""
        ...

    def shutdown(self) -> None:
        """Stop the provider Worker."""
        ...


class FakeTranscriptionService:
    """Use one worker session for all files in a synchronous run."""

    def __init__(self, client: FakeWorkerClient | None = None) -> None:
        self._client = client or FakeWorkerClient()
        self._started = False

    def start(self, model_name: str = "fake-v1") -> AsrCapabilities:
        if self._started:
            raise ProviderUnavailableError(
                "Fake transcription service is already started"
            )
        capabilities = self._client.start(model_name)
        self._started = True
        return capabilities

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        if not self._started:
            raise ProviderUnavailableError("Fake transcription service is not started")
        return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        self._client.shutdown()
        self._started = False


class FasterWhisperTranscriptionService:
    """Use one configured Faster Whisper Worker for a serial run."""

    def __init__(
        self,
        config: FasterWhisperConfig,
        client: _FasterWhisperWorker | None = None,
    ) -> None:
        self._config = config
        self._client = client or FasterWhisperWorkerClient()
        self._started = False
        self._cpu_fallback_used = False

    def start(self, model_name: str = "faster-whisper") -> AsrCapabilities:
        del model_name
        if self._started:
            raise ProviderUnavailableError(
                "Faster Whisper transcription service is already started"
            )
        capabilities = self._client.start(self._config)
        if not capabilities.native_word_timestamps:
            self._client.shutdown()
            raise ProviderUnavailableError(
                "Faster Whisper Worker does not provide native word timestamps"
            )
        self._started = True
        return capabilities

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        if not self._started:
            raise ProviderUnavailableError(
                "Faster Whisper transcription service is not started"
            )
        if request.timestamps.value == "disabled":
            raise TranscriptionError(
                "Faster Whisper Phase 1 requires native word timestamps"
            )
        try:
            return self._client.transcribe(request, artifact_dir)
        except TranscriptionError as exc:
            if (
                self._config.device != "auto"
                or self._cpu_fallback_used
                or not _is_cuda_runtime_failure(str(exc))
            ):
                raise
            self._cpu_fallback_used = True
            logging.getLogger("captioner.transcription").warning(
                "CUDA ASR failed; retrying once on CPU",
                extra={"stage": "transcription", "fallback": "cpu"},
            )
            self._client.shutdown()
            self._config = self._config.model_copy(
                update={"device": "cpu", "compute_type": "int8"}
            )
            self._client.start(self._config)
            return self._client.transcribe(request, artifact_dir)

    def shutdown(self) -> None:
        self._client.shutdown()
        self._started = False
        self._cpu_fallback_used = False


def _is_cuda_runtime_failure(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            ".so",
            "shared librar",
            "cuda",
            "cudnn",
            "cublas",
            "out of memory",
        )
    )
