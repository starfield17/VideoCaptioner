"""Application workflow for refining an existing subtitle file."""

from dataclasses import dataclass
from pathlib import Path

from captioner.llm.api import ContentContext
from captioner.shared.errors import (
    CaptionerError,
    ConfigurationError,
    SubtitleValidationError,
)
from captioner.subtitles.importers.json import read_json
from captioner.subtitles.importers.srt import read_srt
from captioner.subtitles.models import QualityReport, SubtitleDocument
from captioner.workflow.options import PipelineOptions
from captioner.workflow.pipeline import build_subtitle_service
from captioner.workflow.progress import (
    ExecutionContext,
    ProgressEvent,
    ProgressKind,
    ProgressStage,
    execution_context,
)


@dataclass(frozen=True)
class RefineResult:
    """Result of one synchronous existing-subtitle refinement."""

    input_path: Path
    subtitle: SubtitleDocument
    quality_report: QualityReport
    output_paths: tuple[Path, ...]


def refine_srt(
    input_path: Path,
    options: PipelineOptions,
    output_dir: Path,
    *,
    source_language: str = "und",
    input_bilingual: bool = False,
    context: ExecutionContext | None = None,
) -> RefineResult:
    """Run the configured subtitle stages without media or ASR work."""

    selected_context = execution_context(context)
    selected_context.checkpoint()
    selected_context.emit(
        ProgressEvent(ProgressKind.RUN_STARTED, input_path=input_path, file_count=1)
    )
    if input_path.suffix.lower() == ".srt":
        document = read_srt(
            input_path,
            source_language=source_language,
            bilingual=input_bilingual,
        )
    elif input_path.suffix.lower() == ".json":
        document = read_json(input_path)
    else:
        raise ConfigurationError("refine accepts an SRT or subtitle JSON file")
    service = build_subtitle_service(options, selected_context)
    subtitle_context = None
    if options.context_analysis.enabled:
        _stage(selected_context, ProgressStage.CONTEXT_ANALYSIS, True, input_path)
        try:
            subtitle_context = service.analyze_context(
                " ".join(
                    cue.corrected_text or cue.source_text for cue in document.cues
                )[: options.context_analysis.max_characters]
            )
        except CaptionerError:
            subtitle_context = ContentContext()
        _stage(selected_context, ProgressStage.CONTEXT_ANALYSIS, False, input_path)
    if options.correction.enabled:
        _stage(selected_context, ProgressStage.CORRECTION, True, input_path)
        document = service.correct(
            document,
            context=subtitle_context,
            batch_size=options.correction.batch_size,
            parallelism=options.correction.parallelism,
            max_change_ratio=options.correction.max_change_ratio,
        )
        _stage(selected_context, ProgressStage.CORRECTION, False, input_path)
    if options.cleanup.enabled:
        _stage(selected_context, ProgressStage.CLEANUP, True, input_path)
        document = service.cleanup(
            document,
            fillers=options.cleanup.fillers,
            non_speech_markers=options.cleanup.non_speech_markers,
            collapse_repetitions=options.cleanup.collapse_repetitions,
        )
        _stage(selected_context, ProgressStage.CLEANUP, False, input_path)
    if options.translation.enabled:
        _stage(selected_context, ProgressStage.TRANSLATION, True, input_path)
        document = service.translate(
            document,
            target_language=options.translation.target_language,
            allow_partial=True,
            context=subtitle_context,
            batch_size=options.translation.batch_size,
            parallelism=options.translation.parallelism,
        )
        _stage(selected_context, ProgressStage.TRANSLATION, False, input_path)
    _stage(selected_context, ProgressStage.QUALITY, True, input_path)
    report = service.check_quality(document, options=options.subtitle)
    _stage(selected_context, ProgressStage.QUALITY, False, input_path)
    if options.repair.enabled and report.has_repairable_issues:
        _stage(selected_context, ProgressStage.REPAIR, True, input_path)
        document = service.repair(
            document,
            report,
            context=subtitle_context,
            batch_size=options.repair.batch_size,
            parallelism=options.repair.parallelism,
        )
        report = service.check_quality(document, options=options.subtitle)
        _stage(selected_context, ProgressStage.REPAIR, False, input_path)
    if (
        options.translation.enabled
        and not options.translation.allow_partial
        and any(cue.translated_text is None for cue in document.cues)
    ):
        raise SubtitleValidationError(
            "translation remains incomplete after repair and partial output is disabled"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(output_format.value for output_format in options.output.formats)
    _stage(selected_context, ProgressStage.EXPORT, True, input_path)
    output_paths = service.export(
        document,
        output_dir=output_dir,
        basename=f"{input_path.stem}.refined",
        formats=formats,
        bilingual=options.output.bilingual,
        quality_options=options.subtitle,
        overwrite=options.output.overwrite == "replace",
    )
    _stage(selected_context, ProgressStage.EXPORT, False, input_path)
    selected_context.emit(
        ProgressEvent(
            ProgressKind.FILE_COMPLETED,
            input_path=input_path,
            file_index=1,
            file_count=1,
        )
    )
    selected_context.emit(ProgressEvent(ProgressKind.RUN_COMPLETED, file_count=1))
    return RefineResult(
        input_path=input_path,
        subtitle=document,
        quality_report=report,
        output_paths=output_paths,
    )


def _stage(
    context: ExecutionContext,
    stage: ProgressStage,
    started: bool,
    input_path: Path,
) -> None:
    context.checkpoint()
    context.emit(
        ProgressEvent(
            ProgressKind.STAGE_STARTED if started else ProgressKind.STAGE_COMPLETED,
            stage=stage,
            input_path=input_path,
            file_index=1,
            file_count=1,
        )
    )


__all__ = ["RefineResult", "refine_srt"]
