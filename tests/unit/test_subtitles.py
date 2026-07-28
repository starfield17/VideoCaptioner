from captioner.llm.fake import FakeLlm
from captioner.subtitles.api import QualitySeverity, SubtitleService
from captioner.transcription.models import (
    TimedWord,
    TimingOrigin,
    TranscriptDocument,
    TranscriptSegment,
)


def _document() -> TranscriptDocument:
    words = (
        TimedWord(id="w1", text="错误", start_ms=0, end_ms=300),
        TimedWord(id="w2", text="句子。", start_ms=320, end_ms=800),
    )
    return TranscriptDocument(
        language="zh",
        text="错误句子。",
        timing_origin=TimingOrigin.ASR_NATIVE,
        words=words,
        segments=(
            TranscriptSegment(
                id="seg1",
                text="错误句子。",
                start_ms=0,
                end_ms=800,
                word_ids=("w1", "w2"),
            ),
        ),
        provider="fake",
        model_name="fake-v1",
    )


def test_fake_stages_preserve_source_and_timing() -> None:
    service = SubtitleService(FakeLlm())
    segmented = service.segment(_document())
    corrected = service.correct(segmented)
    translated = service.translate(corrected, "en")

    cue = translated.cues[0]
    assert cue.source_text == "错误句子。"
    assert cue.corrected_text == "正确句子。"
    assert cue.translated_text == "[en] 正确句子。"
    assert (cue.start_ms, cue.end_ms) == (0, 800)


def test_translation_failure_is_a_warning_when_partial_is_allowed() -> None:
    service = SubtitleService(FakeLlm(fail_stage="translation"))
    document = service.correct(service.segment(_document()))
    translated = service.translate(document, "en", allow_partial=True)
    report = service.check_quality(translated)
    assert translated.cues[0].translated_text is None
    assert report.issues[0].severity is QualitySeverity.WARNING
