"""Synchronous subtitle stages with bounded LLM batch parallelism."""

from pathlib import Path
from typing import Literal

from captioner.llm.api import (
    BatchResult,
    BoundarySelection,
    CloudLlm,
    LlmTextItem,
    LlmToken,
    ParallelLlmExecutor,
    TextUpdateBatch,
    split_batches,
)
from captioner.shared.errors import SubtitleValidationError
from captioner.subtitles.exporters.json_export import write_json
from captioner.subtitles.exporters.srt import write_srt
from captioner.subtitles.exporters.vtt import write_vtt
from captioner.subtitles.glossary import Glossary
from captioner.subtitles.models import QualityReport, SubtitleCue, SubtitleDocument
from captioner.subtitles.quality import QualityOptions, check_quality
from captioner.subtitles.segmentation import (
    SegmentationPiece,
    build_subtitle_document,
    make_segmentation_pieces,
    rule_boundaries,
)
from captioner.transcription.api import TranscriptDocument


class SubtitleService:
    """Own immutable stage transitions and the LLM stage barriers."""

    def __init__(
        self,
        llm: CloudLlm,
        executor: ParallelLlmExecutor | None = None,
        glossary: Glossary | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor or ParallelLlmExecutor()
        self._glossary = glossary or Glossary()

    def segment(
        self,
        transcript: TranscriptDocument,
        *,
        batch_tokens: int = 800,
        parallelism: int = 1,
    ) -> SubtitleDocument:
        pieces = make_segmentation_pieces(transcript)
        tokens = tuple(
            LlmToken(
                id=piece.id,
                text=piece.text,
                gap_after_ms=_gap_after(pieces, index),
            )
            for index, piece in enumerate(pieces)
        )
        batches = split_batches(tokens, batch_tokens)
        outcomes = self._executor.map(
            batches,
            self._llm.choose_boundaries,
            max_workers=parallelism,
        )
        boundary_ids: list[str] = []
        used_fallback = False
        for outcome in outcomes:
            selection, fallback = _segmentation_result(outcome)
            boundary_ids.extend(selection.break_after)
            used_fallback = used_fallback or fallback
        selection = BoundarySelection(break_after=tuple(dict.fromkeys(boundary_ids)))
        document = build_subtitle_document(transcript, pieces, selection)
        if used_fallback:
            document = _add_warning(document, "segmentation_fallback")
        return document

    def correct(
        self,
        document: SubtitleDocument,
        *,
        batch_size: int = 30,
        parallelism: int = 1,
    ) -> SubtitleDocument:
        requests = split_batches(
            tuple(
                LlmTextItem(id=cue.id, text=cue.source_text) for cue in document.cues
            ),
            batch_size,
        )
        outcomes = self._executor.map(
            requests,
            self._llm.correct,
            max_workers=parallelism,
        )
        values, warnings = _collect_updates(outcomes, "correction")
        return _apply_text_stage(document, values, warnings, "corrected_text")

    def translate(
        self,
        document: SubtitleDocument,
        target_language: str,
        allow_partial: bool = True,
        *,
        batch_size: int = 30,
        parallelism: int = 1,
    ) -> SubtitleDocument:
        requests = split_batches(
            tuple(
                LlmTextItem(
                    id=cue.id,
                    text=cue.corrected_text or cue.source_text,
                )
                for cue in document.cues
            ),
            batch_size,
        )
        outcomes = self._executor.map(
            requests,
            lambda items: self._llm.translate(items, target_language),
            max_workers=parallelism,
        )
        values, warnings = _collect_updates(outcomes, "translation")
        values = {
            item_id: self._glossary.apply(value) for item_id, value in values.items()
        }
        if not allow_partial and warnings:
            raise SubtitleValidationError(
                "translation failed and partial output is disabled"
            )
        translated = _apply_text_stage(
            document,
            values,
            warnings,
            "translated_text",
        )
        return _replace_cues(translated, translated.cues, target_language)

    def repair(
        self,
        document: SubtitleDocument,
        report: QualityReport | None = None,
        *,
        batch_size: int = 20,
        parallelism: int = 1,
    ) -> SubtitleDocument:
        if document.target_language is None:
            return document
        repairable_ids = {
            issue.cue_id
            for issue in (report.issues if report is not None else ())
            if issue.code in {"translation_missing", "empty_translation"}
            and issue.cue_id is not None
        }
        if report is None:
            repairable_ids = {
                cue.id for cue in document.cues if cue.translated_text is None
            }
        requests = split_batches(
            tuple(
                LlmTextItem(
                    id=cue.id,
                    text=cue.corrected_text or cue.source_text,
                )
                for cue in document.cues
                if cue.id in repairable_ids
            ),
            batch_size,
        )
        if not requests:
            return document
        outcomes = self._executor.map(
            requests,
            lambda items: self._llm.repair(items, document.target_language or ""),
            max_workers=parallelism,
        )
        values, warnings = _collect_updates(outcomes, "repair")
        values = {
            item_id: self._glossary.apply(value) for item_id, value in values.items()
        }
        return _apply_text_stage(document, values, warnings, "translated_text")

    def check_quality(
        self,
        document: SubtitleDocument,
        options: QualityOptions | None = None,
    ) -> QualityReport:
        return check_quality(document, options)

    def export(
        self,
        document: SubtitleDocument,
        output_dir: Path,
        basename: str,
        formats: tuple[str, ...],
        bilingual: bool,
    ) -> tuple[Path, ...]:
        outputs: list[Path] = []
        for output_format in formats:
            if output_format == "srt":
                outputs.append(
                    write_srt(
                        document,
                        output_dir / f"{basename}.srt",
                        bilingual=bilingual,
                    )
                )
            elif output_format == "json":
                outputs.append(
                    write_json(document, output_dir / f"{basename}.subtitle.json")
                )
            elif output_format == "vtt":
                outputs.append(
                    write_vtt(
                        document,
                        output_dir / f"{basename}.vtt",
                        bilingual=bilingual,
                    )
                )
            elif output_format == "bilingual_srt":
                outputs.append(
                    write_srt(
                        document,
                        output_dir / f"{basename}.bilingual.srt",
                        bilingual=True,
                    )
                )
            else:
                raise SubtitleValidationError(
                    f"unsupported output format: {output_format}"
                )
        return tuple(outputs)


def _segmentation_result(
    outcome: BatchResult[tuple[LlmToken, ...], BoundarySelection],
) -> tuple[BoundarySelection, bool]:
    if outcome.error is not None or outcome.value is None:
        return rule_boundaries(outcome.request), True
    try:
        return _validate_selection(outcome.value, outcome.request), False
    except (SubtitleValidationError, TypeError, ValueError):
        return rule_boundaries(outcome.request), True


def _validate_selection(
    selection: BoundarySelection,
    tokens: tuple[LlmToken, ...],
) -> BoundarySelection:
    validated = BoundarySelection.model_validate(selection.model_dump(mode="json"))
    token_ids = {token.id for token in tokens}
    unknown_ids = set(validated.break_after) - token_ids
    if unknown_ids:
        raise SubtitleValidationError(
            f"segmentation returned unknown boundary IDs: {sorted(unknown_ids)}"
        )
    if not validated.break_after:
        raise SubtitleValidationError("segmentation returned no boundary IDs")
    return validated


def _collect_updates(
    outcomes: tuple[BatchResult[tuple[LlmTextItem, ...], TextUpdateBatch], ...],
    stage: str,
) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    warnings: dict[str, str] = {}
    for outcome in outcomes:
        expected_ids = tuple(item.id for item in outcome.request)
        if outcome.error is not None or outcome.value is None:
            _mark_batch_warning(warnings, expected_ids, stage, outcome.error)
            continue
        try:
            values.update(_validated_updates(outcome.value, expected_ids))
        except (SubtitleValidationError, TypeError, ValueError) as exc:
            _mark_batch_warning(warnings, expected_ids, stage, exc)
    return values, warnings


def _validated_updates(
    batch: TextUpdateBatch,
    expected_ids: tuple[str, ...],
) -> dict[str, str]:
    validated = TextUpdateBatch.model_validate(batch.model_dump(mode="json"))
    expected = set(expected_ids)
    values: dict[str, str] = {}
    for update in validated.items:
        if update.id not in expected:
            raise SubtitleValidationError(f"LLM returned unknown ID: {update.id}")
        if update.id in values:
            raise SubtitleValidationError(f"LLM returned duplicate ID: {update.id}")
        if not update.text.strip():
            raise SubtitleValidationError(f"LLM returned blank text for {update.id}")
        values[update.id] = update.text
    if set(values) != expected:
        raise SubtitleValidationError("LLM update IDs do not match batch IDs")
    return values


def _mark_batch_warning(
    warnings: dict[str, str],
    ids: tuple[str, ...],
    stage: str,
    error: Exception | None,
) -> None:
    suffix = type(error).__name__ if error is not None else "UnknownError"
    warning = f"{stage}_failed:{suffix}"
    for item_id in ids:
        warnings[item_id] = warning


def _apply_text_stage(
    document: SubtitleDocument,
    values: dict[str, str],
    warnings: dict[str, str],
    field_name: Literal["corrected_text", "translated_text"],
) -> SubtitleDocument:
    cues: list[SubtitleCue] = []
    for cue in document.cues:
        update: dict[str, object] = {}
        if cue.id in values:
            update[field_name] = values[cue.id]
        if cue.id in warnings:
            update["warnings"] = (*cue.warnings, warnings[cue.id])
        cues.append(cue.model_copy(update=update))
    return _replace_cues(document, tuple(cues))


def _replace_cues(
    document: SubtitleDocument,
    cues: tuple[SubtitleCue, ...],
    target_language: str | None = None,
) -> SubtitleDocument:
    return SubtitleDocument(
        source_language=document.source_language,
        target_language=(
            target_language if target_language is not None else document.target_language
        ),
        cues=cues,
    )


def _add_warning(document: SubtitleDocument, warning: str) -> SubtitleDocument:
    return _replace_cues(
        document,
        tuple(
            cue.model_copy(update={"warnings": (*cue.warnings, warning)})
            for cue in document.cues
        ),
    )


def _gap_after(pieces: tuple[SegmentationPiece, ...], index: int) -> int:
    if index == len(pieces) - 1:
        return 0
    return max(
        0,
        pieces[index + 1].start_ms - pieces[index].end_ms,
    )


__all__ = ["SubtitleService"]
