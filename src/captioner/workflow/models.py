"""Workflow result models."""

from dataclasses import dataclass
from pathlib import Path

from captioner.subtitles.models import QualityReport, SubtitleDocument
from captioner.transcription.api import TranscriptDocument


@dataclass(frozen=True)
class ProcessingResult:
    """Successful result for one input file."""

    input_path: Path
    subtitle: SubtitleDocument
    quality_report: QualityReport
    output_paths: tuple[Path, ...]
    workdir: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileFailure:
    """A file-level failure in a batch run."""

    input_path: Path
    error_type: str
    message: str


@dataclass(frozen=True)
class RunResult:
    """Complete batch summary."""

    succeeded: tuple[ProcessingResult, ...]
    failed: tuple[FileFailure, ...]
    workdir: Path | None


@dataclass(frozen=True)
class TranscriptionResult:
    """Successful transcript-only result for one input file."""

    input_path: Path
    transcript: TranscriptDocument
    output_path: Path
    workdir: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranscriptionRunResult:
    """Complete transcript-only batch summary."""

    succeeded: tuple[TranscriptionResult, ...]
    failed: tuple[FileFailure, ...]
    workdir: Path | None
