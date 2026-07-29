"""Synchronous progress and cooperative cancellation contracts."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event

from captioner.shared.errors import OperationCancelled


class ProgressKind(StrEnum):
    RUN_STARTED = "run_started"
    FILE_STARTED = "file_started"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    FILE_COMPLETED = "file_completed"
    FILE_FAILED = "file_failed"
    RUN_COMPLETED = "run_completed"
    CANCELLED = "cancelled"


class ProgressStage(StrEnum):
    CONFIGURATION = "configuration"
    MODEL_DOWNLOAD = "model_download"
    RUNTIME_INSTALL = "runtime_install"
    PROVIDER = "provider"
    MEDIA = "media"
    VOICE_SEPARATION = "voice_separation"
    TRANSCRIPTION = "transcription"
    CONTEXT_ANALYSIS = "context_analysis"
    SEGMENTATION = "segmentation"
    CORRECTION = "correction"
    CLEANUP = "cleanup"
    TRANSLATION = "translation"
    QUALITY = "quality"
    REPAIR = "repair"
    EXPORT = "export"


@dataclass(frozen=True)
class ProgressEvent:
    """One stable application event consumed by CLI and GUI adapters."""

    kind: ProgressKind
    stage: ProgressStage | None = None
    input_path: Path | None = None
    file_index: int | None = None
    file_count: int | None = None
    message: str = ""


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled("operation cancelled by user")


ProgressObserver = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class ExecutionContext:
    """Optional observer and cancellation state for one synchronous operation."""

    cancellation: CancellationToken
    observer: ProgressObserver | None = None

    @classmethod
    def create(
        cls,
        *,
        cancellation: CancellationToken | None = None,
        observer: ProgressObserver | None = None,
    ) -> "ExecutionContext":
        return cls(cancellation or CancellationToken(), observer)

    def checkpoint(self) -> None:
        self.cancellation.raise_if_cancelled()

    def emit(self, event: ProgressEvent) -> None:
        if self.observer is not None:
            self.observer(event)


def execution_context(value: ExecutionContext | None) -> ExecutionContext:
    return value or ExecutionContext.create()


__all__ = [
    "CancellationToken",
    "ExecutionContext",
    "ProgressEvent",
    "ProgressKind",
    "ProgressObserver",
    "ProgressStage",
]
