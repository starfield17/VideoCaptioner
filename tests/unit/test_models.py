import pytest
from pydantic import ValidationError

from captioner.subtitles.models import SubtitleCue, SubtitleDocument
from captioner.transcription.models import (
    TimedWord,
    TimingOrigin,
    TranscriptDocument,
    TranscriptSegment,
)


def _transcript() -> TranscriptDocument:
    words = (
        TimedWord(id="w1", text="hello", start_ms=0, end_ms=400),
        TimedWord(id="w2", text=".", start_ms=450, end_ms=600),
    )
    return TranscriptDocument(
        language="en",
        text="hello.",
        timing_origin=TimingOrigin.ASR_NATIVE,
        words=words,
        segments=(
            TranscriptSegment(
                id="s1",
                text="hello.",
                start_ms=0,
                end_ms=600,
                word_ids=("w1", "w2"),
            ),
        ),
        provider="fake",
        model_name="fake-v1",
    )


def test_transcript_rejects_segment_native_words() -> None:
    with pytest.raises(ValidationError):
        TranscriptDocument(
            language="en",
            text="hello",
            timing_origin=TimingOrigin.SEGMENT_NATIVE,
            words=(TimedWord(id="w1", text="hello", start_ms=0, end_ms=100),),
            segments=(),
            provider="fake",
            model_name="fake-v1",
        )


def test_transcript_rejects_duplicate_ids_and_bad_coverage() -> None:
    transcript = _transcript()
    with pytest.raises(ValidationError):
        TranscriptDocument(
            language=transcript.language,
            text=transcript.text,
            timing_origin=transcript.timing_origin,
            words=transcript.words,
            segments=(
                TranscriptSegment(
                    id="s1",
                    text="hello.",
                    start_ms=0,
                    end_ms=500,
                    word_ids=("w1", "w2"),
                ),
                TranscriptSegment(
                    id="s1",
                    text="hello.",
                    start_ms=600,
                    end_ms=700,
                    word_ids=(),
                ),
            ),
            provider="fake",
            model_name="fake-v1",
        )


def test_domain_documents_are_frozen_and_cues_do_not_overlap() -> None:
    with pytest.raises(ValidationError):
        TimedWord(id="w1", text="hello", start_ms=100, end_ms=100)

    cue = SubtitleCue(id="cue1", start_ms=0, end_ms=500, source_text="hello")
    document = SubtitleDocument(source_language="en", cues=(cue,))
    with pytest.raises(ValidationError):
        cue.source_text = "changed"
    assert document.cues[0].source_text == "hello"

    with pytest.raises(ValidationError):
        SubtitleDocument(
            source_language="en",
            cues=(
                cue,
                SubtitleCue(id="cue2", start_ms=400, end_ms=700, source_text="overlap"),
            ),
        )
