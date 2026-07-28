"""Synchronous provider pipeline composition."""

import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from captioner.llm.api import ParallelLlmExecutor
from captioner.llm.fake import FakeLlm
from captioner.llm.openai_adapter import OpenAICompatibleLlm
from captioner.media.api import FakeMediaService, FfmpegMediaService, MediaService
from captioner.media.voice_separation import (
    CommandVoiceSeparator,
    VoiceSeparationError,
    VoiceSeparator,
)
from captioner.shared.errors import (
    CaptionerError,
    ConfigurationError,
    OperationCancelled,
    ProviderUnavailableError,
    SubtitleValidationError,
)
from captioner.shared.logging import log_extra
from captioner.subtitles.api import (
    ContentContext,
    QualityReport,
    SubtitleDocument,
    SubtitleService,
)
from captioner.subtitles.segmentation import SegmentationConstraints
from captioner.transcription.api import (
    FakeTranscriptionService,
    FasterWhisperTranscriptionService,
    NemoTranscriptionService,
    Qwen3TranscriptionService,
    TranscriptDocument,
    TranscriptionRequest,
    TranscriptionService,
)
from captioner.workflow.models import (
    FileFailure,
    ProcessingResult,
    RunResult,
    TranscriptionResult,
    TranscriptionRunResult,
)
from captioner.workflow.options import (
    FasterWhisperAsrOptions,
    NemoAsrOptions,
    PipelineOptions,
    Qwen3AsrOptions,
)
from captioner.workflow.progress import (
    ExecutionContext,
    ProgressEvent,
    ProgressKind,
    ProgressStage,
    execution_context,
)
from captioner.workflow.workspace import RunWorkspace


@dataclass(frozen=True)
class PipelineServices:
    """Concrete service ports composed by the application boundary."""

    media: MediaService
    transcription: TranscriptionService
    subtitles: SubtitleService
    voice_separator: VoiceSeparator | None = None


def build_subtitle_service(
    options: PipelineOptions,
    context: ExecutionContext | None = None,
) -> SubtitleService:
    """Build the subtitle service without starting media or ASR."""

    llm = (
        OpenAICompatibleLlm(options.llm)
        if options.llm.provider == "openai-compatible"
        else FakeLlm()
    )
    selected_context = execution_context(context)
    return SubtitleService(
        llm,
        executor=ParallelLlmExecutor(
            cancellation_check=selected_context.checkpoint,
        ),
        glossary=options.glossary,
    )


def build_fake_services(
    options: PipelineOptions,
    context: ExecutionContext | None = None,
) -> PipelineServices:
    """Build an all-local deterministic composition for tests and demos."""

    return PipelineServices(
        media=FakeMediaService(
            sample_rate=options.audio.sample_rate,
            channels=options.audio.channels,
        ),
        transcription=FakeTranscriptionService(),
        subtitles=build_subtitle_service(options, context),
        voice_separator=(
            CommandVoiceSeparator(
                command_env=options.audio.voice_separation.command_env
            )
            if options.audio.voice_separation.enabled
            else None
        ),
    )


def build_services(
    options: PipelineOptions,
    context: ExecutionContext | None = None,
) -> PipelineServices:
    """Compose services for the selected static ASR provider."""

    voice_separator = (
        CommandVoiceSeparator(command_env=options.audio.voice_separation.command_env)
        if options.audio.voice_separation.enabled
        else None
    )
    subtitles = build_subtitle_service(options, context)

    if isinstance(options.asr, FasterWhisperAsrOptions):
        return PipelineServices(
            media=FfmpegMediaService(
                sample_rate=options.audio.sample_rate,
                channels=options.audio.channels,
            ),
            transcription=FasterWhisperTranscriptionService(options.asr.faster_whisper),
            subtitles=subtitles,
            voice_separator=voice_separator,
        )
    if isinstance(options.asr, Qwen3AsrOptions):
        return PipelineServices(
            media=FfmpegMediaService(
                sample_rate=options.audio.sample_rate,
                channels=options.audio.channels,
            ),
            transcription=Qwen3TranscriptionService(
                options.asr.qwen3,
                timestamps=options.asr.timestamps,
            ),
            subtitles=subtitles,
            voice_separator=voice_separator,
        )
    if isinstance(options.asr, NemoAsrOptions):
        return PipelineServices(
            media=FfmpegMediaService(
                sample_rate=options.audio.sample_rate,
                channels=options.audio.channels,
            ),
            transcription=NemoTranscriptionService(options.asr.nemo),
            subtitles=subtitles,
            voice_separator=voice_separator,
        )
    return PipelineServices(
        media=FakeMediaService(
            sample_rate=options.audio.sample_rate,
            channels=options.audio.channels,
        ),
        transcription=FakeTranscriptionService(),
        subtitles=subtitles,
        voice_separator=voice_separator,
    )


def process_media(
    input_path: Path,
    options: PipelineOptions,
    services: PipelineServices,
    output_dir: Path,
    context: ExecutionContext | None = None,
) -> ProcessingResult:
    """Process one input with an isolated temporary workspace."""

    result = run_files((input_path,), options, services, output_dir, context)
    if result.failed:
        failure = result.failed[0]
        workdir = result.workdir or "unavailable"
        raise ConfigurationError(
            f"{failure.error_type}: {failure.message}; workdir retained at {workdir}"
        )
    return result.succeeded[0]


def run_files(
    input_paths: tuple[Path, ...],
    options: PipelineOptions,
    services: PipelineServices,
    output_dir: Path,
    context: ExecutionContext | None = None,
    *,
    source_root: Path | None = None,
) -> RunResult:
    """Process files serially while sharing one ASR worker session."""

    selected_context = execution_context(context)
    selected_context.checkpoint()
    options = PipelineOptions.validate_for_phase0(options)
    if not input_paths:
        raise ConfigurationError("at least one input file is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = RunWorkspace(keep=options.run.keep_workdir)
    workspace.write_run_metadata(input_paths, options)
    selected_context.emit(
        ProgressEvent(
            ProgressKind.RUN_STARTED,
            file_count=len(input_paths),
        )
    )
    successes: list[ProcessingResult] = []
    failures: list[FileFailure] = []
    started = False
    try:
        _stage_started(selected_context, ProgressStage.PROVIDER)
        services.transcription.start()
        started = True
        _stage_completed(selected_context, ProgressStage.PROVIDER)
        for index, input_path in enumerate(input_paths, start=1):
            selected_context.checkpoint()
            selected_context.emit(
                ProgressEvent(
                    ProgressKind.FILE_STARTED,
                    input_path=input_path,
                    file_index=index,
                    file_count=len(input_paths),
                )
            )
            try:
                successes.append(
                    _process_one(
                        input_path,
                        index,
                        options,
                        services,
                        workspace,
                        _output_dir_for(input_path, output_dir, source_root),
                        selected_context,
                        len(input_paths),
                    )
                )
                selected_context.emit(
                    ProgressEvent(
                        ProgressKind.FILE_COMPLETED,
                        input_path=input_path,
                        file_index=index,
                        file_count=len(input_paths),
                    )
                )
            except OperationCancelled:
                raise
            except Exception as exc:
                selected_context.emit(
                    ProgressEvent(
                        ProgressKind.FILE_FAILED,
                        input_path=input_path,
                        file_index=index,
                        file_count=len(input_paths),
                        message=str(exc),
                    )
                )
                logging.getLogger("captioner.pipeline").error(
                    "file processing failed",
                    extra=log_extra(
                        stage="pipeline",
                        input=str(input_path),
                        error_type=type(exc).__name__,
                    ),
                    exc_info=True,
                )
                failures.append(
                    FileFailure(
                        input_path=input_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if not options.run.continue_on_error:
                    break
    except OperationCancelled as exc:
        selected_context.emit(ProgressEvent(ProgressKind.CANCELLED))
        raise OperationCancelled(
            f"{exc}; workdir retained at {workspace.root}"
        ) from exc
    except CaptionerError as exc:
        raise ProviderUnavailableError(
            f"{exc}; workdir retained at {workspace.root}"
        ) from exc
    finally:
        if started:
            services.transcription.shutdown()

    retained_workdir = workspace.root
    if not failures and not options.run.keep_workdir:
        workspace.cleanup()
        retained_workdir = None
        successes = [replace(result, workdir=None) for result in successes]
    selected_context.emit(
        ProgressEvent(
            ProgressKind.RUN_COMPLETED,
            file_count=len(input_paths),
        )
    )
    return RunResult(
        succeeded=tuple(successes),
        failed=tuple(failures),
        workdir=retained_workdir,
    )


def transcribe_files(
    input_paths: tuple[Path, ...],
    options: PipelineOptions,
    services: PipelineServices,
    output_dir: Path,
    context: ExecutionContext | None = None,
    *,
    source_root: Path | None = None,
) -> TranscriptionRunResult:
    """Prepare and transcribe files serially into Transcript JSON artifacts."""

    selected_context = execution_context(context)
    selected_context.checkpoint()
    options = PipelineOptions.validate_for_phase0(options)
    if not input_paths:
        raise ConfigurationError("at least one input file is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = RunWorkspace(keep=options.run.keep_workdir)
    workspace.write_run_metadata(input_paths, options)
    selected_context.emit(
        ProgressEvent(
            ProgressKind.RUN_STARTED,
            file_count=len(input_paths),
        )
    )
    successes: list[TranscriptionResult] = []
    failures: list[FileFailure] = []
    started = False
    try:
        _stage_started(selected_context, ProgressStage.PROVIDER)
        services.transcription.start()
        started = True
        _stage_completed(selected_context, ProgressStage.PROVIDER)
        for index, input_path in enumerate(input_paths, start=1):
            selected_context.checkpoint()
            selected_context.emit(
                ProgressEvent(
                    ProgressKind.FILE_STARTED,
                    input_path=input_path,
                    file_index=index,
                    file_count=len(input_paths),
                )
            )
            try:
                _, transcript, warnings = _transcribe_input(
                    input_path,
                    index,
                    options,
                    services,
                    workspace,
                    selected_context,
                    len(input_paths),
                )
                file_output_dir = _output_dir_for(input_path, output_dir, source_root)
                file_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = file_output_dir / f"{input_path.stem}.transcript.json"
                _write_transcript(output_path, transcript)
                successes.append(
                    TranscriptionResult(
                        input_path=input_path,
                        transcript=transcript,
                        output_path=output_path,
                        workdir=workspace.root,
                        warnings=warnings,
                    )
                )
                selected_context.emit(
                    ProgressEvent(
                        ProgressKind.FILE_COMPLETED,
                        input_path=input_path,
                        file_index=index,
                        file_count=len(input_paths),
                    )
                )
            except OperationCancelled:
                raise
            except Exception as exc:
                selected_context.emit(
                    ProgressEvent(
                        ProgressKind.FILE_FAILED,
                        input_path=input_path,
                        file_index=index,
                        file_count=len(input_paths),
                        message=str(exc),
                    )
                )
                logging.getLogger("captioner.pipeline").error(
                    "file transcription failed",
                    extra=log_extra(
                        stage="transcription",
                        input=str(input_path),
                        error_type=type(exc).__name__,
                    ),
                    exc_info=True,
                )
                failures.append(
                    FileFailure(
                        input_path=input_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if not options.run.continue_on_error:
                    break
    except OperationCancelled as exc:
        selected_context.emit(ProgressEvent(ProgressKind.CANCELLED))
        raise OperationCancelled(
            f"{exc}; workdir retained at {workspace.root}"
        ) from exc
    except CaptionerError as exc:
        raise ProviderUnavailableError(
            f"{exc}; workdir retained at {workspace.root}"
        ) from exc
    finally:
        if started:
            services.transcription.shutdown()

    retained_workdir = workspace.root
    if not failures and not options.run.keep_workdir:
        workspace.cleanup()
        retained_workdir = None
        successes = [replace(result, workdir=None) for result in successes]
    selected_context.emit(
        ProgressEvent(
            ProgressKind.RUN_COMPLETED,
            file_count=len(input_paths),
        )
    )
    return TranscriptionRunResult(
        succeeded=tuple(successes),
        failed=tuple(failures),
        workdir=retained_workdir,
    )


def discover_inputs(
    input_path: Path,
    provider: str = "fake",
    context: ExecutionContext | None = None,
) -> tuple[Path, ...]:
    """Resolve one input or supported recursive children of an input directory."""

    selected_context = execution_context(context)
    selected_context.checkpoint()
    if input_path.is_file():
        return (input_path,)
    if input_path.is_dir():
        suffixes = {".json"} if provider == "fake" else _MEDIA_SUFFIXES
        scan_errors: list[OSError] = []
        inputs: list[Path] = []
        for root, _directories, filenames in os.walk(
            input_path,
            followlinks=False,
            onerror=scan_errors.append,
        ):
            selected_context.checkpoint()
            inputs.extend(
                Path(root) / name
                for name in filenames
                if Path(name).suffix.lower() in suffixes
            )
        if scan_errors:
            raise ConfigurationError(
                f"could not scan input directory {input_path}: {scan_errors[0]}"
            )
        inputs.sort(
            key=lambda path: (
                path.relative_to(input_path).as_posix().casefold(),
                path.relative_to(input_path).as_posix(),
            )
        )
        if inputs:
            return tuple(inputs)
    raise ConfigurationError(f"no supported inputs found at {input_path}")


def _output_dir_for(
    input_path: Path,
    output_dir: Path,
    source_root: Path | None,
) -> Path:
    if source_root is None:
        return output_dir
    try:
        relative_parent = input_path.parent.relative_to(source_root)
    except ValueError as exc:
        raise ConfigurationError(
            f"input {input_path} is outside source root {source_root}"
        ) from exc
    return output_dir / relative_parent


def _process_one(
    input_path: Path,
    index: int,
    options: PipelineOptions,
    services: PipelineServices,
    workspace: RunWorkspace,
    output_dir: Path,
    execution: ExecutionContext,
    file_count: int,
) -> ProcessingResult:
    file_dir, transcript, warnings = _transcribe_input(
        input_path,
        index,
        options,
        services,
        workspace,
        execution,
        file_count,
    )
    mutable_warnings = list(warnings)
    context: ContentContext | None = None
    if options.context_analysis.enabled and (
        options.segmentation.enabled
        or options.correction.enabled
        or options.translation.enabled
    ):
        _stage_started(
            execution,
            ProgressStage.CONTEXT_ANALYSIS,
            input_path,
            index,
            file_count,
        )
        context_text = transcript.text
        if len(context_text) > options.context_analysis.max_characters:
            context_text = context_text[: options.context_analysis.max_characters]
            mutable_warnings.append("context_analysis_truncated")
        try:
            context = services.subtitles.analyze_context(context_text)
        except CaptionerError as exc:
            context = ContentContext()
            mutable_warnings.append(f"context_analysis_fallback:{type(exc).__name__}")
        _atomic_write(
            file_dir / "content_context.json",
            context.model_dump_json(indent=2),
        )
        _log_event(file_dir, "context_analysis", "completed")
        _stage_completed(
            execution,
            ProgressStage.CONTEXT_ANALYSIS,
            input_path,
            index,
            file_count,
        )
    _stage_started(
        execution,
        ProgressStage.SEGMENTATION,
        input_path,
        index,
        file_count,
    )
    subtitle = services.subtitles.segment(
        transcript,
        context=context,
        batch_tokens=options.segmentation.batch_tokens,
        overlap_tokens=options.segmentation.overlap_tokens,
        constraints=SegmentationConstraints(
            max_duration_ms=options.subtitle.max_duration_ms,
            max_chars_cjk=options.subtitle.max_chars_cjk,
            max_words_latin=options.subtitle.max_words_latin,
            max_cps=options.subtitle.max_cps,
            silence_boundary_ms=options.segmentation.silence_boundary_ms,
        ),
        parallelism=options.segmentation.parallelism,
    )
    _write_document(file_dir / "subtitle.segmented.json", subtitle)
    _log_event(file_dir, "segmentation", "completed")
    _stage_completed(
        execution,
        ProgressStage.SEGMENTATION,
        input_path,
        index,
        file_count,
    )

    if options.correction.enabled:
        _stage_started(
            execution,
            ProgressStage.CORRECTION,
            input_path,
            index,
            file_count,
        )
        subtitle = services.subtitles.correct(
            subtitle,
            context=context,
            batch_size=options.correction.batch_size,
            parallelism=options.correction.parallelism,
            max_change_ratio=options.correction.max_change_ratio,
        )
        _write_document(file_dir / "subtitle.corrected.json", subtitle)
        _log_event(file_dir, "correction", "completed")
        _stage_completed(
            execution,
            ProgressStage.CORRECTION,
            input_path,
            index,
            file_count,
        )

    if options.cleanup.enabled:
        _stage_started(
            execution,
            ProgressStage.CLEANUP,
            input_path,
            index,
            file_count,
        )
        subtitle = services.subtitles.cleanup(
            subtitle,
            fillers=options.cleanup.fillers,
            non_speech_markers=options.cleanup.non_speech_markers,
            collapse_repetitions=options.cleanup.collapse_repetitions,
        )
        _write_document(file_dir / "subtitle.cleaned.json", subtitle)
        _log_event(file_dir, "cleanup", "completed")
        _stage_completed(
            execution,
            ProgressStage.CLEANUP,
            input_path,
            index,
            file_count,
        )

    if options.translation.enabled:
        _stage_started(
            execution,
            ProgressStage.TRANSLATION,
            input_path,
            index,
            file_count,
        )
        subtitle = services.subtitles.translate(
            subtitle,
            target_language=options.translation.target_language,
            allow_partial=True,
            context=context,
            batch_size=options.translation.batch_size,
            parallelism=options.translation.parallelism,
        )
        _write_document(file_dir / "subtitle.translated.json", subtitle)
        _log_event(file_dir, "translation", "completed")
        _stage_completed(
            execution,
            ProgressStage.TRANSLATION,
            input_path,
            index,
            file_count,
        )

    _stage_started(
        execution,
        ProgressStage.QUALITY,
        input_path,
        index,
        file_count,
    )
    quality_report = services.subtitles.check_quality(
        subtitle, options=options.subtitle
    )
    _write_quality(file_dir / "quality.json", quality_report)
    _log_event(file_dir, "quality", "completed")
    _stage_completed(
        execution,
        ProgressStage.QUALITY,
        input_path,
        index,
        file_count,
    )
    if options.repair.enabled and quality_report.has_repairable_issues:
        _stage_started(
            execution,
            ProgressStage.REPAIR,
            input_path,
            index,
            file_count,
        )
        subtitle = services.subtitles.repair(
            subtitle,
            quality_report,
            context=context,
            batch_size=options.repair.batch_size,
            parallelism=options.repair.parallelism,
        )
        _write_document(file_dir / "subtitle.repaired.json", subtitle)
        quality_report = services.subtitles.check_quality(
            subtitle, options=options.subtitle
        )
        _write_quality(file_dir / "quality.json", quality_report)
        _log_event(file_dir, "repair", "completed")
        _stage_completed(
            execution,
            ProgressStage.REPAIR,
            input_path,
            index,
            file_count,
        )
    if (
        options.translation.enabled
        and not options.translation.allow_partial
        and any(cue.translated_text is None for cue in subtitle.cues)
    ):
        raise SubtitleValidationError(
            "translation remains incomplete after repair and partial output is disabled"
        )
    formats = tuple(output_format.value for output_format in options.output.formats)
    _stage_started(
        execution,
        ProgressStage.EXPORT,
        input_path,
        index,
        file_count,
    )
    output_paths = services.subtitles.export(
        subtitle,
        output_dir=output_dir,
        basename=input_path.stem,
        formats=formats,
        bilingual=options.output.bilingual,
        quality_options=options.subtitle,
        overwrite=options.output.overwrite == "replace",
    )
    _log_event(file_dir, "export", "completed")
    _stage_completed(
        execution,
        ProgressStage.EXPORT,
        input_path,
        index,
        file_count,
    )
    return ProcessingResult(
        input_path=input_path,
        subtitle=subtitle,
        quality_report=quality_report,
        output_paths=output_paths,
        workdir=workspace.root,
        warnings=tuple(mutable_warnings),
    )


def _transcribe_input(
    input_path: Path,
    index: int,
    options: PipelineOptions,
    services: PipelineServices,
    workspace: RunWorkspace,
    context: ExecutionContext,
    file_count: int,
) -> tuple[Path, TranscriptDocument, tuple[str, ...]]:
    file_dir = workspace.file_dir(index, input_path)
    _stage_started(
        context,
        ProgressStage.MEDIA,
        input_path,
        index,
        file_count,
    )
    audio = services.media.prepare_audio(input_path, file_dir)
    _log_event(file_dir, "media", "completed")
    _stage_completed(
        context,
        ProgressStage.MEDIA,
        input_path,
        index,
        file_count,
    )
    warnings: list[str] = []
    if options.audio.voice_separation.enabled:
        separator = services.voice_separator
        if separator is None:
            error = VoiceSeparationError("voice-separation service is not configured")
            if options.audio.voice_separation.required:
                raise error
            warnings.append(f"voice_separation_fallback:{type(error).__name__}")
        else:
            try:
                _stage_started(
                    context,
                    ProgressStage.VOICE_SEPARATION,
                    input_path,
                    index,
                    file_count,
                )
                audio = separator.separate(audio, file_dir / "vocals.wav")
                _log_event(file_dir, "voice_separation", "completed")
                _stage_completed(
                    context,
                    ProgressStage.VOICE_SEPARATION,
                    input_path,
                    index,
                    file_count,
                )
            except VoiceSeparationError as exc:
                if options.audio.voice_separation.required:
                    raise
                warnings.append(f"voice_separation_fallback:{type(exc).__name__}")
    request = TranscriptionRequest(
        audio_path=audio.path,
        language=None if options.asr.language == "auto" else options.asr.language,
        initial_prompt=options.asr.initial_prompt or None,
        timestamps=options.asr.timestamps,
    )
    _stage_started(
        context,
        ProgressStage.TRANSCRIPTION,
        input_path,
        index,
        file_count,
    )
    transcript = services.transcription.transcribe(request, file_dir)
    _log_event(file_dir, "transcription", "completed")
    _stage_completed(
        context,
        ProgressStage.TRANSCRIPTION,
        input_path,
        index,
        file_count,
    )
    return (file_dir, transcript, tuple(warnings))


def _write_document(path: Path, document: SubtitleDocument) -> None:
    content = document.model_dump_json(indent=2)
    _atomic_write(path, content)


def _write_transcript(path: Path, transcript: TranscriptDocument) -> None:
    _atomic_write(path, transcript.model_dump_json(indent=2))


def _write_quality(path: Path, report: QualityReport) -> None:
    content = report.model_dump_json(indent=2)
    _atomic_write(path, content)


def _atomic_write(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _log_event(file_dir: Path, stage: str, status: str) -> None:
    logging.getLogger("captioner.pipeline").info(
        "%s %s",
        stage,
        status,
        extra=log_extra(stage=stage, status=status),
    )
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": status,
    }
    with (file_dir / "processing.log").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _stage_started(
    context: ExecutionContext,
    stage: ProgressStage,
    input_path: Path | None = None,
    file_index: int | None = None,
    file_count: int | None = None,
) -> None:
    context.checkpoint()
    context.emit(
        ProgressEvent(
            ProgressKind.STAGE_STARTED,
            stage=stage,
            input_path=input_path,
            file_index=file_index,
            file_count=file_count,
        )
    )


def _stage_completed(
    context: ExecutionContext,
    stage: ProgressStage,
    input_path: Path | None = None,
    file_index: int | None = None,
    file_count: int | None = None,
) -> None:
    context.checkpoint()
    context.emit(
        ProgressEvent(
            ProgressKind.STAGE_COMPLETED,
            stage=stage,
            input_path=input_path,
            file_index=file_index,
            file_count=file_count,
        )
    )


_MEDIA_SUFFIXES = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}
