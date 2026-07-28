"""Synchronous subtitle stages with bounded LLM batch parallelism."""

import re
from pathlib import Path
from typing import Literal

from captioner.llm.api import (
    BatchResult,
    BoundarySelection,
    CloudLlm,
    ContentContext,
    LlmGlossaryEntry,
    LlmStageContext,
    LlmTextBatch,
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
    SegmentationConstraints,
    SegmentationPiece,
    SegmentationWindow,
    build_subtitle_document,
    constrain_boundaries,
    make_segmentation_pieces,
    make_segmentation_windows,
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

    def analyze_context(self, text: str) -> ContentContext:
        """Analyze one bounded transcript exactly once at the workflow boundary."""

        return self._llm.analyze_context(text)

    def segment(
        self,
        transcript: TranscriptDocument,
        *,
        context: ContentContext | None = None,
        batch_tokens: int = 800,
        overlap_tokens: int = 80,
        constraints: SegmentationConstraints | None = None,
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
        windows = make_segmentation_windows(tokens, batch_tokens, overlap_tokens)
        outcomes = self._executor.map(
            windows,
            lambda window: self._llm.choose_boundaries(
                window.tokens,
                context=_stage_context(context, self._glossary),
            ),
            max_workers=parallelism,
        )
        boundary_ids: list[str] = []
        used_fallback = False
        for outcome in outcomes:
            selection, fallback = _segmentation_result(outcome)
            boundary_ids.extend(selection.break_after)
            used_fallback = used_fallback or fallback
        selection = constrain_boundaries(
            pieces,
            tuple(dict.fromkeys(boundary_ids)),
            constraints or SegmentationConstraints(),
        )
        policy = constraints or SegmentationConstraints()
        document = build_subtitle_document(transcript, pieces, selection)
        if used_fallback:
            document = _add_warning(document, "segmentation_fallback")
        if not transcript.words and any(
            piece.end_ms - piece.start_ms > policy.max_duration_ms for piece in pieces
        ):
            document = _add_warning(document, "atomic_segment_too_long")
        return document

    def correct(
        self,
        document: SubtitleDocument,
        *,
        context: ContentContext | None = None,
        batch_size: int = 30,
        parallelism: int = 1,
        max_change_ratio: float = 0.5,
    ) -> SubtitleDocument:
        requests = _text_batches(
            tuple(
                LlmTextItem(id=cue.id, text=cue.source_text) for cue in document.cues
            ),
            batch_size,
            context,
            self._glossary,
        )
        outcomes = self._executor.map(
            requests,
            lambda request: self._llm.correct(
                request.items,
                context=request.context,
            ),
            max_workers=parallelism,
        )
        values, warnings = _collect_updates(
            outcomes,
            "correction",
            max_change_ratio=max_change_ratio,
        )
        return _apply_text_stage(document, values, warnings, "corrected_text")

    def cleanup(
        self,
        document: SubtitleDocument,
        *,
        fillers: tuple[str, ...] = (),
        non_speech_markers: tuple[str, ...] = (),
        collapse_repetitions: bool = False,
    ) -> SubtitleDocument:
        """Apply explicitly configured deterministic text cleanup."""

        cues: list[SubtitleCue] = []
        removable = tuple(
            item for item in (*fillers, *non_speech_markers) if item.strip()
        )
        for cue in document.cues:
            text = cue.corrected_text or cue.source_text
            for item in removable:
                text = text.replace(item, "")
            if collapse_repetitions:
                text = re.sub(r"\b(\w+)(?:\s+\1){1,}\b", r"\1", text)
            text = " ".join(text.split()).strip()
            if not text:
                cues.append(
                    cue.model_copy(
                        update={"warnings": (*cue.warnings, "cleanup_skipped:blank")}
                    )
                )
                continue
            cues.append(cue.model_copy(update={"corrected_text": text}))
        return _replace_cues(document, tuple(cues))

    def translate(
        self,
        document: SubtitleDocument,
        target_language: str,
        allow_partial: bool = True,
        *,
        context: ContentContext | None = None,
        batch_size: int = 30,
        parallelism: int = 1,
    ) -> SubtitleDocument:
        requests = _text_batches(
            tuple(
                LlmTextItem(
                    id=cue.id,
                    text=cue.corrected_text or cue.source_text,
                )
                for cue in document.cues
            ),
            batch_size,
            context,
            self._glossary,
        )
        outcomes = self._executor.map(
            requests,
            lambda request: self._llm.translate(
                request.items,
                target_language,
                context=request.context,
            ),
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
        context: ContentContext | None = None,
        batch_size: int = 20,
        parallelism: int = 1,
    ) -> SubtitleDocument:
        if document.target_language is None:
            return document
        repairable_ids = set(report.repairable_cue_ids if report is not None else ())
        if report is None:
            repairable_ids = {
                cue.id for cue in document.cues if cue.translated_text is None
            }
        requests = _text_batches(
            tuple(
                LlmTextItem(
                    id=cue.id,
                    text=cue.corrected_text or cue.source_text,
                )
                for cue in document.cues
                if cue.id in repairable_ids
            ),
            batch_size,
            context,
            self._glossary,
        )
        if not requests:
            return document
        outcomes = self._executor.map(
            requests,
            lambda request: self._llm.repair(
                request.items,
                document.target_language or "",
                context=request.context,
            ),
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
        return check_quality(document, options, glossary=self._glossary)

    def export(
        self,
        document: SubtitleDocument,
        output_dir: Path,
        basename: str,
        formats: tuple[str, ...],
        bilingual: bool,
        *,
        quality_options: QualityOptions | None = None,
        overwrite: bool = False,
    ) -> tuple[Path, ...]:
        formatting = quality_options or QualityOptions()
        outputs: list[Path] = []
        for output_format in formats:
            if output_format == "srt":
                outputs.append(
                    write_srt(
                        document,
                        output_dir / f"{basename}.srt",
                        bilingual=bilingual,
                        max_line_chars=formatting.max_line_chars,
                        max_lines=formatting.max_lines,
                        overwrite=overwrite,
                    )
                )
            elif output_format == "json":
                outputs.append(
                    write_json(
                        document,
                        output_dir / f"{basename}.subtitle.json",
                        overwrite=overwrite,
                    )
                )
            elif output_format == "vtt":
                outputs.append(
                    write_vtt(
                        document,
                        output_dir / f"{basename}.vtt",
                        bilingual=bilingual,
                        max_line_chars=formatting.max_line_chars,
                        max_lines=formatting.max_lines,
                        overwrite=overwrite,
                    )
                )
            elif output_format == "bilingual_srt":
                outputs.append(
                    write_srt(
                        document,
                        output_dir / f"{basename}.bilingual.srt",
                        bilingual=True,
                        max_line_chars=formatting.max_line_chars,
                        max_lines=formatting.max_lines,
                        overwrite=overwrite,
                    )
                )
            else:
                raise SubtitleValidationError(
                    f"unsupported output format: {output_format}"
                )
        return tuple(outputs)


def _segmentation_result(
    outcome: BatchResult[SegmentationWindow, BoundarySelection],
) -> tuple[BoundarySelection, bool]:
    if outcome.error is not None or outcome.value is None:
        fallback = rule_boundaries(
            tuple(
                token
                for token in outcome.request.tokens
                if token.id in set(outcome.request.owner_ids)
            )
        )
        return fallback, True
    try:
        validated = _validate_selection(outcome.value, outcome.request.tokens)
        owner_ids = set(outcome.request.owner_ids)
        selected = tuple(
            item_id for item_id in validated.break_after if item_id in owner_ids
        )
        if not selected and outcome.request.owner_ids:
            selected = (outcome.request.owner_ids[-1],)
        return BoundarySelection(break_after=selected), False
    except (SubtitleValidationError, TypeError, ValueError):
        fallback = rule_boundaries(
            tuple(
                token
                for token in outcome.request.tokens
                if token.id in set(outcome.request.owner_ids)
            )
        )
        return fallback, True


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
    outcomes: tuple[BatchResult[LlmTextBatch, TextUpdateBatch], ...],
    stage: str,
    *,
    max_change_ratio: float | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    warnings: dict[str, str] = {}
    for outcome in outcomes:
        expected_ids = tuple(item.id for item in outcome.request.items)
        if outcome.error is not None or outcome.value is None:
            _mark_batch_warning(warnings, expected_ids, stage, outcome.error)
            continue
        try:
            values.update(
                _validated_updates(
                    outcome.value,
                    outcome.request.items,
                    max_change_ratio=max_change_ratio,
                )
            )
        except (SubtitleValidationError, TypeError, ValueError) as exc:
            _mark_batch_warning(warnings, expected_ids, stage, exc)
    return values, warnings


def _text_batches(
    items: tuple[LlmTextItem, ...],
    batch_size: int,
    content: ContentContext | None,
    glossary: Glossary,
) -> tuple[LlmTextBatch, ...]:
    batches = split_batches(items, batch_size)
    glossary_entries = tuple(
        LlmGlossaryEntry(
            source=entry.source,
            target=entry.target,
            note=entry.note,
        )
        for entry in glossary.entries
    )
    requests: list[LlmTextBatch] = []
    offset = 0
    for batch in batches:
        before = items[max(0, offset - 2) : offset]
        after_start = offset + len(batch)
        after = items[after_start : after_start + 2]
        requests.append(
            LlmTextBatch(
                items=batch,
                context=LlmStageContext(
                    content=content or ContentContext(),
                    glossary=glossary_entries,
                    before=before,
                    after=after,
                    style_rules=("faithful", "concise", "subtitle-readable"),
                ),
            )
        )
        offset = after_start
    return tuple(requests)


def _stage_context(
    content: ContentContext | None,
    glossary: Glossary,
) -> LlmStageContext:
    return LlmStageContext(
        content=content or ContentContext(),
        glossary=tuple(
            LlmGlossaryEntry(
                source=entry.source,
                target=entry.target,
                note=entry.note,
            )
            for entry in glossary.entries
        ),
    )


def _validated_updates(
    batch: TextUpdateBatch,
    source_items: tuple[LlmTextItem, ...],
    *,
    max_change_ratio: float | None = None,
) -> dict[str, str]:
    validated = TextUpdateBatch.model_validate(batch.model_dump(mode="json"))
    expected_ids = tuple(item.id for item in source_items)
    expected = set(expected_ids)
    sources = {item.id: item.text for item in source_items}
    values: dict[str, str] = {}
    for update in validated.items:
        if update.id not in expected:
            raise SubtitleValidationError(f"LLM returned unknown ID: {update.id}")
        if update.id in values:
            raise SubtitleValidationError(f"LLM returned duplicate ID: {update.id}")
        if not update.text.strip():
            raise SubtitleValidationError(f"LLM returned blank text for {update.id}")
        if max_change_ratio is not None:
            _validate_correction(
                sources[update.id],
                update.text,
                update.id,
                max_change_ratio,
            )
        values[update.id] = update.text
    if set(values) != expected:
        raise SubtitleValidationError("LLM update IDs do not match batch IDs")
    return values


def _validate_correction(
    source: str,
    corrected: str,
    item_id: str,
    max_change_ratio: float,
) -> None:
    for protected in _protected_values(source):
        if protected not in corrected:
            raise SubtitleValidationError(
                f"correction removed protected content from {item_id}: {protected}"
            )
    if _contains_cjk(source) and not _contains_cjk(corrected):
        raise SubtitleValidationError(
            f"correction changed the source script for {item_id}"
        )
    if _contains_latin(source) and not _contains_latin(corrected):
        raise SubtitleValidationError(
            f"correction changed the source script for {item_id}"
        )
    normalized_source = _normalized_for_distance(source)
    normalized_corrected = _normalized_for_distance(corrected)
    denominator = max(len(normalized_source), len(normalized_corrected), 1)
    ratio = _edit_distance(normalized_source, normalized_corrected) / denominator
    if ratio > max_change_ratio:
        raise SubtitleValidationError(
            f"correction change ratio {ratio:.3f} exceeds "
            f"{max_change_ratio:.3f} for {item_id}"
        )


def _protected_values(text: str) -> tuple[str, ...]:
    patterns = (
        r"https?://[^\s]+",
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        r"`[^`]+`",
        r"\b\d+(?:[.,:/-]\d+)*%?\b",
    )
    return tuple(
        match.group(0) for pattern in patterns for match in re.finditer(pattern, text)
    )


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in text)


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _normalized_for_distance(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


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
