"""Public workflow API."""

from captioner.media.voice_separation import VoiceSeparationOptions
from captioner.subtitles.glossary import Glossary, GlossaryEntry
from captioner.subtitles.quality import QualityOptions
from captioner.workflow.doctor import DoctorReport, run_doctor
from captioner.workflow.models import (
    FileFailure,
    ProcessingResult,
    RunResult,
    TranscriptionResult,
    TranscriptionRunResult,
)
from captioner.workflow.options import (
    AsrOptions,
    AudioOptions,
    CorrectionOptions,
    FakeAsrOptions,
    FasterWhisperAsrOptions,
    LlmOptions,
    OutputFormat,
    OutputOptions,
    PipelineOptions,
    Qwen3AsrOptions,
    RepairOptions,
    RunOptions,
    SegmentationOptions,
    TranslationOptions,
    load_options,
    with_keep_workdir,
)
from captioner.workflow.pipeline import (
    PipelineServices,
    build_fake_services,
    build_services,
    build_subtitle_service,
    discover_inputs,
    process_media,
    run_files,
    transcribe_files,
)
from captioner.workflow.refine import RefineResult, refine_srt

__all__ = [
    "AsrOptions",
    "AudioOptions",
    "VoiceSeparationOptions",
    "CorrectionOptions",
    "DoctorReport",
    "FileFailure",
    "FakeAsrOptions",
    "FasterWhisperAsrOptions",
    "Qwen3AsrOptions",
    "LlmOptions",
    "Glossary",
    "GlossaryEntry",
    "OutputFormat",
    "OutputOptions",
    "PipelineOptions",
    "PipelineServices",
    "QualityOptions",
    "ProcessingResult",
    "RunOptions",
    "RepairOptions",
    "RunResult",
    "RefineResult",
    "SegmentationOptions",
    "TranslationOptions",
    "build_fake_services",
    "build_services",
    "build_subtitle_service",
    "discover_inputs",
    "load_options",
    "process_media",
    "run_files",
    "refine_srt",
    "TranscriptionResult",
    "TranscriptionRunResult",
    "transcribe_files",
    "run_doctor",
    "with_keep_workdir",
]
