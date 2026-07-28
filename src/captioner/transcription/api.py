"""Public transcription API."""

from pathlib import Path
from typing import Protocol

from captioner.transcription.capabilities import AsrCapabilities
from captioner.transcription.models import (
    TimedWord,
    TimingOrigin,
    TranscriptDocument,
    TranscriptSegment,
)
from captioner.transcription.providers.faster_whisper import (
    FasterWhisperConfig,
    FasterWhisperVadConfig,
    FasterWhisperWorkerClient,
)
from captioner.transcription.providers.nemo import (
    NemoConfig,
    NemoTranscriptionService,
    NemoWorkerClient,
)
from captioner.transcription.providers.qwen3 import (
    Qwen3Config,
    Qwen3TranscriptionService,
    Qwen3WorkerClient,
)
from captioner.transcription.requests import TimestampRequirement, TranscriptionRequest
from captioner.transcription.service import (
    FakeTranscriptionService as _FakeTranscriptionService,
)
from captioner.transcription.service import (
    FasterWhisperTranscriptionService as _FasterWhisperTranscriptionService,
)

FakeTranscriptionService = _FakeTranscriptionService
FasterWhisperTranscriptionService = _FasterWhisperTranscriptionService


class TranscriptionService(Protocol):
    """Lifecycle and request contract for one synchronous worker session."""

    def start(self, model_name: str = "fake-v1") -> AsrCapabilities:
        """Start and load the worker exactly once for a run."""
        ...

    def transcribe(
        self, request: TranscriptionRequest, artifact_dir: Path
    ) -> TranscriptDocument:
        """Transcribe one prepared input into a stable document."""
        ...

    def shutdown(self) -> None:
        """Release the worker."""
        ...


__all__ = [
    "AsrCapabilities",
    "FasterWhisperConfig",
    "FasterWhisperTranscriptionService",
    "FasterWhisperVadConfig",
    "FasterWhisperWorkerClient",
    "NemoConfig",
    "NemoTranscriptionService",
    "NemoWorkerClient",
    "Qwen3Config",
    "Qwen3TranscriptionService",
    "Qwen3WorkerClient",
    "TimedWord",
    "TimestampRequirement",
    "TimingOrigin",
    "TranscriptionRequest",
    "TranscriptionService",
    "TranscriptDocument",
    "TranscriptSegment",
]
__all__.append("FakeTranscriptionService")
