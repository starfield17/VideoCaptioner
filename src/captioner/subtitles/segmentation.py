"""Deterministic cue construction from transcript timing."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from captioner.llm.api import BoundarySelection, CloudLlm, LlmToken
from captioner.shared.errors import LlmPermanentError, SubtitleValidationError
from captioner.shared.ids import make_cue_id
from captioner.subtitles.models import SubtitleCue, SubtitleDocument
from captioner.transcription.api import TranscriptDocument


@dataclass(frozen=True)
class _Piece:
    id: str
    text: str
    start_ms: int
    end_ms: int
    source_word_ids: tuple[str, ...]


SegmentationPiece = _Piece


@dataclass(frozen=True)
class SegmentationWindow:
    """One owner range with read-only overlap tokens."""

    tokens: tuple[LlmToken, ...]
    owner_ids: tuple[str, ...]


@dataclass(frozen=True)
class SegmentationConstraints:
    max_duration_ms: int = 7_000
    max_chars_cjk: int = 24
    max_words_latin: int = 14
    max_cps: float = 17.0
    silence_boundary_ms: int = 700


def segment_transcript(
    transcript: TranscriptDocument,
    llm: CloudLlm,
) -> SubtitleDocument:
    """Use boundary IDs to build cues without changing transcript timing."""

    pieces = _pieces(transcript)
    tokens = tuple(
        LlmToken(
            id=piece.id,
            text=piece.text,
            gap_after_ms=_gap_after(pieces, index),
        )
        for index, piece in enumerate(pieces)
    )
    try:
        selection = llm.choose_boundaries(tokens)
        _validate_selection(selection, pieces)
    except (LlmPermanentError, SubtitleValidationError, ValueError):
        selection = _rule_boundaries(tokens)
    return _build_document(transcript, pieces, selection)


def make_segmentation_pieces(
    transcript: TranscriptDocument,
) -> tuple[SegmentationPiece, ...]:
    """Expose immutable owner pieces to the batched subtitle service."""

    return _pieces(transcript)


def make_segmentation_windows(
    tokens: tuple[LlmToken, ...],
    owner_size: int,
    overlap: int,
) -> tuple[SegmentationWindow, ...]:
    windows: list[SegmentationWindow] = []
    for owner_start in range(0, len(tokens), owner_size):
        owner_end = min(len(tokens), owner_start + owner_size)
        request_start = max(0, owner_start - overlap)
        request_end = min(len(tokens), owner_end + overlap)
        windows.append(
            SegmentationWindow(
                tokens=tokens[request_start:request_end],
                owner_ids=tuple(token.id for token in tokens[owner_start:owner_end]),
            )
        )
    return tuple(windows)


def constrain_boundaries(
    pieces: tuple[SegmentationPiece, ...],
    proposed_ids: Iterable[str],
    constraints: SegmentationConstraints,
) -> BoundarySelection:
    """Merge semantic choices with deterministic readability boundaries."""

    proposed = set(proposed_ids)
    selected: list[str] = []
    cue_start = 0
    candidate: int | None = None
    for index, piece in enumerate(pieces):
        gap = _gap_after(pieces, index)
        if _has_terminal_punctuation(piece.text) or (
            gap >= constraints.silence_boundary_ms
        ):
            candidate = index
        text = _join_text(value.text for value in pieces[cue_start : index + 1])
        duration_ms = piece.end_ms - pieces[cue_start].start_ms
        cjk_count = sum(1 for character in text if _looks_cjk(character))
        latin_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
        cps = len(text.replace(" ", "")) / max(duration_ms / 1_000, 0.001)
        exceeded = (
            duration_ms > constraints.max_duration_ms
            or cjk_count > constraints.max_chars_cjk
            or latin_words > constraints.max_words_latin
            or (
                duration_ms >= constraints.max_duration_ms // 2
                and cps > constraints.max_cps
            )
        )
        cut: int | None = None
        if piece.id in proposed:
            cut = index
        elif exceeded:
            cut = (
                candidate
                if candidate is not None and candidate >= cue_start
                else max(cue_start, index - 1)
            )
        if cut is not None:
            boundary_id = pieces[cut].id
            if not selected or selected[-1] != boundary_id:
                selected.append(boundary_id)
            cue_start = cut + 1
            candidate = None
    if pieces and (not selected or selected[-1] != pieces[-1].id):
        selected.append(pieces[-1].id)
    return BoundarySelection(break_after=tuple(selected))


def _has_terminal_punctuation(text: str) -> bool:
    return text.rstrip().endswith(("。", "！", "？", ".", "!", "?", ";", "；"))


def _pieces(transcript: TranscriptDocument) -> tuple[_Piece, ...]:
    if transcript.words:
        return tuple(
            _Piece(
                id=word.id,
                text=word.text,
                start_ms=word.start_ms,
                end_ms=word.end_ms,
                source_word_ids=(word.id,),
            )
            for word in transcript.words
        )
    return tuple(
        _Piece(
            id=segment.id,
            text=segment.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            source_word_ids=(),
        )
        for segment in transcript.segments
    )


def _gap_after(pieces: tuple[_Piece, ...], index: int) -> int:
    if index == len(pieces) - 1:
        return 0
    return max(0, pieces[index + 1].start_ms - pieces[index].end_ms)


def _validate_selection(
    selection: BoundarySelection, pieces: tuple[_Piece, ...]
) -> None:
    piece_ids = {piece.id for piece in pieces}
    unknown_ids = set(selection.break_after) - piece_ids
    if unknown_ids:
        raise SubtitleValidationError(
            f"segmentation returned unknown boundary IDs: {sorted(unknown_ids)}"
        )
    if not selection.break_after:
        raise SubtitleValidationError("segmentation returned no boundary IDs")


def _rule_boundaries(tokens: tuple[LlmToken, ...]) -> BoundarySelection:
    """Fallback that always selects punctuation or the final token."""

    selected = [
        token.id
        for token in tokens
        if _has_terminal_punctuation(token.text) or token.gap_after_ms >= 700
    ]
    if tokens and tokens[-1].id not in selected:
        selected.append(tokens[-1].id)
    return BoundarySelection(break_after=tuple(selected))


def rule_boundaries(tokens: tuple[LlmToken, ...]) -> BoundarySelection:
    """Return deterministic boundaries for one failed LLM window."""

    return _rule_boundaries(tokens)


def _build_document(
    transcript: TranscriptDocument,
    pieces: tuple[_Piece, ...],
    selection: BoundarySelection,
) -> SubtitleDocument:
    boundary_ids = set(selection.break_after)
    cues: list[SubtitleCue] = []
    current: list[_Piece] = []
    for piece in pieces:
        current.append(piece)
        if piece.id in boundary_ids:
            cues.append(_make_cue(len(cues) + 1, current))
            current = []
    if current:
        cues.append(_make_cue(len(cues) + 1, current))
    return SubtitleDocument(
        source_language=transcript.language,
        target_language=None,
        cues=tuple(cues),
    )


def build_subtitle_document(
    transcript: TranscriptDocument,
    pieces: tuple[SegmentationPiece, ...],
    selection: BoundarySelection,
) -> SubtitleDocument:
    """Build a new immutable document after main-thread boundary merging."""

    return _build_document(transcript, pieces, selection)


def _make_cue(index: int, pieces: list[_Piece]) -> SubtitleCue:
    return SubtitleCue(
        id=make_cue_id(index),
        start_ms=pieces[0].start_ms,
        end_ms=pieces[-1].end_ms,
        source_text=_join_text(piece.text for piece in pieces),
        source_word_ids=tuple(
            word_id for piece in pieces for word_id in piece.source_word_ids
        ),
    )


def _join_text(parts: Iterable[str]) -> str:
    values = tuple(parts)
    result = ""
    punctuation = "，。！？；：、,.!?;:)]}」』"
    for value in values:
        if not result:
            result = value
        elif (
            value[:1] in punctuation
            or result[-1:].isspace()
            or value[:1].isspace()
            or result[-1:] in "([{「『"
            or _looks_cjk(result[-1])
            or _looks_cjk(value[0])
        ):
            result += value
        else:
            result += " " + value
    return result


def _looks_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3000 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


__all__ = [
    "SegmentationPiece",
    "SegmentationConstraints",
    "SegmentationWindow",
    "build_subtitle_document",
    "constrain_boundaries",
    "make_segmentation_pieces",
    "make_segmentation_windows",
    "rule_boundaries",
    "segment_transcript",
]
