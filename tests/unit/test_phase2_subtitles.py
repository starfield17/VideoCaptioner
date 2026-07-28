from dataclasses import dataclass

from captioner.llm.api import BoundarySelection, LlmTextItem, LlmToken, TextUpdate
from captioner.llm.errors import LlmRetryableError
from captioner.llm.models import TextUpdateBatch
from captioner.shared.errors import LlmPermanentError
from captioner.subtitles.api import SubtitleService
from captioner.subtitles.models import (
    QualityIssue,
    QualityReport,
    QualitySeverity,
    SubtitleCue,
    SubtitleDocument,
)
from captioner.transcription.models import (
    TimedWord,
    TimingOrigin,
    TranscriptDocument,
    TranscriptSegment,
)


def _transcript() -> TranscriptDocument:
    words = tuple(
        TimedWord(
            id=f"w{index}",
            text=f"word{index}",
            start_ms=index * 500,
            end_ms=index * 500 + 300,
        )
        for index in range(6)
    )
    return TranscriptDocument(
        language="en",
        text=" ".join(word.text for word in words),
        timing_origin=TimingOrigin.ASR_NATIVE,
        words=words,
        segments=(
            TranscriptSegment(
                id="seg1",
                text=" ".join(word.text for word in words),
                start_ms=0,
                end_ms=2_800,
                word_ids=tuple(word.id for word in words),
            ),
        ),
        provider="fake",
        model_name="fake-v1",
    )


def _document() -> SubtitleDocument:
    return SubtitleDocument(
        source_language="en",
        cues=tuple(
            SubtitleCue(
                id=f"cue{index}",
                start_ms=index * 1_000,
                end_ms=index * 1_000 + 500,
                source_text=f"source {index}",
            )
            for index in range(6)
        ),
    )


@dataclass
class _StageLlm:
    failed_correction_id: str | None = None
    failed_translation_id: str | None = None
    fail_repair: bool = False

    def choose_boundaries(self, tokens: tuple[LlmToken, ...]) -> BoundarySelection:
        return BoundarySelection(break_after=(tokens[-1].id,))

    def correct(self, items: tuple[LlmTextItem, ...]) -> TextUpdateBatch:
        if self.failed_correction_id in {item.id for item in items}:
            raise LlmPermanentError("correction batch failed")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=f"corrected {item.text}") for item in items
            )
        )

    def translate(
        self, items: tuple[LlmTextItem, ...], target_language: str
    ) -> TextUpdateBatch:
        if self.failed_translation_id in {item.id for item in items}:
            raise LlmRetryableError("translation batch failed")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=f"{target_language}:{item.text}")
                for item in items
            )
        )

    def repair(
        self, items: tuple[LlmTextItem, ...], target_language: str
    ) -> TextUpdateBatch:
        if self.fail_repair:
            raise LlmPermanentError("repair batch failed")
        return TextUpdateBatch(
            items=tuple(
                TextUpdate(id=item.id, text=f"{target_language}:repaired")
                for item in items
            )
        )


def test_segmentation_batches_merge_boundary_ids_in_input_order() -> None:
    service = SubtitleService(_StageLlm())

    document = service.segment(_transcript(), batch_tokens=2, parallelism=8)

    assert [cue.source_word_ids for cue in document.cues] == [
        ("w0", "w1"),
        ("w2", "w3"),
        ("w4", "w5"),
    ]
    assert [cue.start_ms for cue in document.cues] == [0, 1_000, 2_000]


def test_serial_and_parallel_stage_results_are_identical() -> None:
    def run(parallelism: int) -> SubtitleDocument:
        service = SubtitleService(_StageLlm())
        document = service.segment(
            _transcript(), batch_tokens=2, parallelism=parallelism
        )
        document = service.correct(document, batch_size=2, parallelism=parallelism)
        return service.translate(
            document,
            "fr",
            batch_size=2,
            parallelism=parallelism,
        )

    assert run(1) == run(8)


def test_failed_correction_batch_falls_back_without_losing_other_batches() -> None:
    service = SubtitleService(_StageLlm(failed_correction_id="cue2"))
    document = service.correct(_document(), batch_size=2, parallelism=8)

    assert document.cues[0].corrected_text == "corrected source 0"
    assert document.cues[1].corrected_text == "corrected source 1"
    assert document.cues[2].corrected_text is None
    assert document.cues[2].warnings == ("correction_failed:LlmPermanentError",)
    assert document.cues[4].corrected_text == "corrected source 4"


def test_translation_waits_for_complete_correction_and_repair_runs_once() -> None:
    llm = _StageLlm(failed_translation_id="cue2")
    service = SubtitleService(llm)
    corrected = service.correct(_document(), batch_size=2, parallelism=8)
    translated = service.translate(
        corrected,
        "fr",
        batch_size=2,
        parallelism=8,
        allow_partial=True,
    )
    report = service.check_quality(translated)
    repaired = service.repair(report=report, document=translated, batch_size=2)
    final_report = service.check_quality(repaired)

    assert translated.target_language == "fr"
    assert translated.cues[0].translated_text == "fr:corrected source 0"
    assert translated.cues[2].translated_text is None
    assert report.has_repairable_issues
    assert repaired.cues[2].translated_text == "fr:repaired"
    assert not final_report.has_repairable_issues


def test_repair_failure_preserves_pre_repair_document_with_warning() -> None:
    service = SubtitleService(_StageLlm(fail_repair=True))
    document = SubtitleDocument(
        source_language="en",
        target_language="fr",
        cues=(SubtitleCue(id="cue1", start_ms=0, end_ms=500, source_text="hello"),),
    )
    report = QualityReport(
        issues=(
            QualityIssue(
                code="translation_missing",
                severity=QualitySeverity.WARNING,
                cue_id="cue1",
                message="translation is missing",
            ),
        )
    )

    repaired = service.repair(document, report, batch_size=1)

    assert repaired.cues[0].translated_text is None
    assert repaired.cues[0].warnings == ("repair_failed:LlmPermanentError",)
