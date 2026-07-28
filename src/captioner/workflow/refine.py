"""Application workflow for refining an existing subtitle file."""

from dataclasses import dataclass
from pathlib import Path

from captioner.shared.errors import ConfigurationError
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

    if input_path.suffix.lower() != ".srt":
        raise ConfigurationError("refine currently accepts an SRT file")
    document = read_srt(
        input_path,
        source_language=source_language,
        bilingual=input_bilingual,
    )
    service = build_subtitle_service(options)
    if options.correction.enabled:
        document = service.correct(
            document,
            batch_size=options.correction.batch_size,
            parallelism=options.correction.parallelism,
        )
    if options.translation.enabled:
        document = service.translate(
            document,
            target_language=options.translation.target_language,
            allow_partial=options.translation.allow_partial,
            batch_size=options.translation.batch_size,
            parallelism=options.translation.parallelism,
        )
    report = service.check_quality(document, options=options.subtitle)
    if options.repair.enabled and report.has_repairable_issues:
        document = service.repair(
            document,
            report,
            batch_size=options.repair.batch_size,
            parallelism=options.repair.parallelism,
        )
        report = service.check_quality(document, options=options.subtitle)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(output_format.value for output_format in options.output.formats)
    output_paths = service.export(
        document,
        output_dir=output_dir,
        basename=f"{input_path.stem}.refined",
        formats=formats,
        bilingual=options.output.bilingual,
    )
    return RefineResult(
        input_path=input_path,
        subtitle=document,
        quality_report=report,
        output_paths=output_paths,
    )


__all__ = ["RefineResult", "refine_srt"]
