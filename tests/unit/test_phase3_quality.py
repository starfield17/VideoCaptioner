from captioner.llm.fake import FakeLlm
from captioner.subtitles.glossary import Glossary, GlossaryEntry
from captioner.subtitles.models import SubtitleCue, SubtitleDocument
from captioner.subtitles.quality import QualityOptions, check_quality
from captioner.subtitles.service import SubtitleService


def test_quality_reports_readability_timing_and_overlap_issues_deterministically() -> (
    None
):
    document = SubtitleDocument.model_construct(
        source_language="en",
        target_language="zh",
        cues=(
            SubtitleCue.model_construct(
                id="cue1",
                start_ms=0,
                end_ms=100,
                source_text="x" * 50,
                corrected_text=None,
                translated_text="y" * 50,
            ),
            SubtitleCue.model_construct(
                id="cue2",
                start_ms=50,
                end_ms=50,
                source_text="",
                corrected_text=None,
                translated_text=None,
            ),
        ),
    )
    options = QualityOptions(
        min_duration_ms=700,
        max_duration_ms=1_000,
        max_cps=10,
        max_line_chars=20,
        max_lines=2,
    )

    first = check_quality(document, options)
    second = check_quality(document, options)

    assert first == second
    codes = {issue.code for issue in first.issues}
    assert {
        "empty_source",
        "invalid_duration",
        "overlap",
        "duration_too_short",
        "line_too_long",
        "cps_exceeded",
        "translation_missing",
    }.issubset(codes)


def test_glossary_replaces_translation_terms_in_configured_order() -> None:
    document = SubtitleDocument(
        source_language="en",
        cues=(
            SubtitleCue(
                id="cue1",
                start_ms=0,
                end_ms=1_000,
                source_text="OpenAI model",
            ),
        ),
    )
    service = SubtitleService(
        FakeLlm(),
        glossary=Glossary(
            entries=(
                GlossaryEntry(source="[fr]", target="FR"),
                GlossaryEntry(source="model", target="modèle"),
            )
        ),
    )

    translated = service.translate(document, "fr")

    assert translated.cues[0].translated_text == "FR OpenAI modèle"
