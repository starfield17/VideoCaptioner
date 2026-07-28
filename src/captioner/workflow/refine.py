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
) -> RefineResult:
    """Run the configured subtitle stages without media or ASR work."""

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
    service = build_subtitle_service(options)
    context = None
    if options.context_analysis.enabled:
        try:
            context = service.analyze_context(
                " ".join(
                    cue.corrected_text or cue.source_text for cue in document.cues
                )[: options.context_analysis.max_characters]
            )
        except CaptionerError:
            context = ContentContext()
    if options.correction.enabled:
        document = service.correct(
            document,
            context=context,
            batch_size=options.correction.batch_size,
            parallelism=options.correction.parallelism,
            max_change_ratio=options.correction.max_change_ratio,
        )
    if options.cleanup.enabled:
        document = service.cleanup(
            document,
            fillers=options.cleanup.fillers,
            non_speech_markers=options.cleanup.non_speech_markers,
            collapse_repetitions=options.cleanup.collapse_repetitions,
        )
    if options.translation.enabled:
        document = service.translate(
            document,
            target_language=options.translation.target_language,
            allow_partial=True,
            context=context,
            batch_size=options.translation.batch_size,
            parallelism=options.translation.parallelism,
        )
    report = service.check_quality(document, options=options.subtitle)
    if options.repair.enabled and report.has_repairable_issues:
        document = service.repair(
            document,
            report,
            context=context,
            batch_size=options.repair.batch_size,
            parallelism=options.repair.parallelism,
        )
        report = service.check_quality(document, options=options.subtitle)
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
    output_paths = service.export(
        document,
        output_dir=output_dir,
        basename=f"{input_path.stem}.refined",
        formats=formats,
        bilingual=options.output.bilingual,
        quality_options=options.subtitle,
        overwrite=options.output.overwrite == "replace",
    )
    return RefineResult(
        input_path=input_path,
        subtitle=document,
        quality_report=report,
        output_paths=output_paths,
    )


__all__ = ["RefineResult", "refine_srt"]
