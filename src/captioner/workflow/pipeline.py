"""Synchronous provider pipeline composition."""

from dataclasses import dataclass, replace
from pathlib import Path

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
    ProviderUnavailableError,
)
from captioner.subtitles.api import QualityReport, SubtitleDocument, SubtitleService
from captioner.transcription.api import (
    FakeTranscriptionService,
    FasterWhisperTranscriptionService,
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
    PipelineOptions,
    Qwen3AsrOptions,
)
from captioner.workflow.workspace import RunWorkspace


@dataclass(frozen=True)
class PipelineServices:
    """Concrete service ports composed by the application boundary."""

    media: MediaService
    transcription: TranscriptionService
    subtitles: SubtitleService
    voice_separator: VoiceSeparator | None = None


def build_subtitle_service(options: PipelineOptions) -> SubtitleService:
    """Build the subtitle service without starting media or ASR."""

    llm = (
        OpenAICompatibleLlm(options.llm)
        if options.llm.provider == "openai-compatible"
        else FakeLlm()
    )
    return SubtitleService(llm, glossary=options.glossary)


def build_fake_services(options: PipelineOptions) -> PipelineServices:
    """Build an all-local deterministic composition for tests and demos."""

    return PipelineServices(
        media=FakeMediaService(
            sample_rate=options.audio.sample_rate,
            channels=options.audio.channels,
        ),
        transcription=FakeTranscriptionService(),
        subtitles=build_subtitle_service(options),
        voice_separator=(
            CommandVoiceSeparator(
                command_env=options.audio.voice_separation.command_env
            )
            if options.audio.voice_separation.enabled
            else None
        ),
    )


def build_services(options: PipelineOptions) -> PipelineServices:
    """Compose services for the selected static ASR provider."""

    voice_separator = (
        CommandVoiceSeparator(command_env=options.audio.voice_separation.command_env)
        if options.audio.voice_separation.enabled
        else None
    )
    subtitles = build_subtitle_service(options)

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
) -> ProcessingResult:
    """Process one input with an isolated temporary workspace."""

    result = run_files((input_path,), options, services, output_dir)
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
) -> RunResult:
    """Process files serially while sharing one ASR worker session."""

    options = PipelineOptions.validate_for_phase0(options)
    if not input_paths:
        raise ConfigurationError("at least one input file is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = RunWorkspace(keep=options.run.keep_workdir)
    workspace.write_run_metadata(input_paths, options)
    successes: list[ProcessingResult] = []
    failures: list[FileFailure] = []
    started = False
    try:
        services.transcription.start()
        started = True
        for index, input_path in enumerate(input_paths, start=1):
            try:
                successes.append(
                    _process_one(
                        input_path,
                        index,
                        options,
                        services,
                        workspace,
                        output_dir,
                    )
                )
            except Exception as exc:
                failures.append(
                    FileFailure(
                        input_path=input_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if not options.run.continue_on_error:
                    break
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
) -> TranscriptionRunResult:
    """Prepare and transcribe files serially into Transcript JSON artifacts."""

    options = PipelineOptions.validate_for_phase0(options)
    if not input_paths:
        raise ConfigurationError("at least one input file is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = RunWorkspace(keep=options.run.keep_workdir)
    workspace.write_run_metadata(input_paths, options)
    successes: list[TranscriptionResult] = []
    failures: list[FileFailure] = []
    started = False
    try:
        services.transcription.start()
        started = True
        for index, input_path in enumerate(input_paths, start=1):
            try:
                _, transcript, warnings = _transcribe_input(
                    input_path,
                    index,
                    options,
                    services,
                    workspace,
                )
                output_path = output_dir / f"{input_path.stem}.transcript.json"
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
            except Exception as exc:
                failures.append(
                    FileFailure(
                        input_path=input_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                if not options.run.continue_on_error:
                    break
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
    return TranscriptionRunResult(
        succeeded=tuple(successes),
        failed=tuple(failures),
        workdir=retained_workdir,
    )


def discover_inputs(input_path: Path, provider: str = "fake") -> tuple[Path, ...]:
    """Resolve one input or supported direct children of an input directory."""

    if input_path.is_file():
        return (input_path,)
    if input_path.is_dir():
        suffixes = {".json"} if provider == "fake" else _MEDIA_SUFFIXES
        inputs = tuple(
            sorted(
                path for path in input_path.iterdir() if path.suffix.lower() in suffixes
            )
        )
        if inputs:
            return inputs
    raise ConfigurationError(f"no supported inputs found at {input_path}")


def _process_one(
    input_path: Path,
    index: int,
    options: PipelineOptions,
    services: PipelineServices,
    workspace: RunWorkspace,
    output_dir: Path,
) -> ProcessingResult:
    file_dir, transcript, warnings = _transcribe_input(
        input_path,
        index,
        options,
        services,
        workspace,
    )
    subtitle = services.subtitles.segment(
        transcript,
        batch_tokens=options.segmentation.batch_tokens,
        parallelism=options.segmentation.parallelism,
    )
    _write_document(file_dir / "subtitle.segmented.json", subtitle)

    if options.correction.enabled:
        subtitle = services.subtitles.correct(
            subtitle,
            batch_size=options.correction.batch_size,
            parallelism=options.correction.parallelism,
        )
        _write_document(file_dir / "subtitle.corrected.json", subtitle)

    if options.translation.enabled:
        subtitle = services.subtitles.translate(
            subtitle,
            target_language=options.translation.target_language,
            allow_partial=options.translation.allow_partial,
            batch_size=options.translation.batch_size,
            parallelism=options.translation.parallelism,
        )
        _write_document(file_dir / "subtitle.translated.json", subtitle)

    quality_report = services.subtitles.check_quality(
        subtitle, options=options.subtitle
    )
    _write_quality(file_dir / "quality.json", quality_report)
    if options.repair.enabled and quality_report.has_repairable_issues:
        subtitle = services.subtitles.repair(
            subtitle,
            quality_report,
            batch_size=options.repair.batch_size,
            parallelism=options.repair.parallelism,
        )
        _write_document(file_dir / "subtitle.repaired.json", subtitle)
        quality_report = services.subtitles.check_quality(
            subtitle, options=options.subtitle
        )
        _write_quality(file_dir / "quality.json", quality_report)
    formats = tuple(output_format.value for output_format in options.output.formats)
    output_paths = services.subtitles.export(
        subtitle,
        output_dir=output_dir,
        basename=input_path.stem,
        formats=formats,
        bilingual=options.output.bilingual,
    )
    return ProcessingResult(
        input_path=input_path,
        subtitle=subtitle,
        quality_report=quality_report,
        output_paths=output_paths,
        workdir=workspace.root,
        warnings=warnings,
    )


def _transcribe_input(
    input_path: Path,
    index: int,
    options: PipelineOptions,
    services: PipelineServices,
    workspace: RunWorkspace,
) -> tuple[Path, TranscriptDocument, tuple[str, ...]]:
    file_dir = workspace.file_dir(index, input_path)
    audio = services.media.prepare_audio(input_path, file_dir)
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
                audio = separator.separate(audio, file_dir / "vocals.wav")
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
    return (
        file_dir,
        services.transcription.transcribe(request, file_dir),
        tuple(warnings),
    )


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
