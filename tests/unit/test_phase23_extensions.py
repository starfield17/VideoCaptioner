import json
from pathlib import Path

import pytest

from captioner.llm.api import LlmStageContext, LlmTextItem, TextUpdate
from captioner.llm.fake import FakeLlm
from captioner.llm.models import TextUpdateBatch
from captioner.shared.errors import ExportError
from captioner.subtitles.exporters.srt import render_srt, write_srt
from captioner.subtitles.formatting import format_text
from captioner.subtitles.glossary import Glossary, GlossaryEntry
from captioner.subtitles.models import SubtitleCue, SubtitleDocument
from captioner.subtitles.service import SubtitleService
from captioner.workflow.api import (
    PipelineOptions,
    build_fake_services,
    refine_srt,
    run_files,
)

ROOT = Path(__file__).resolve().parents[2]


class _UnsafeCorrectionLlm(FakeLlm):
    def correct(
        self,
        items: tuple[LlmTextItem, ...],
        *,
        context: LlmStageContext | None = None,
    ) -> TextUpdateBatch:
        del context
        return TextUpdateBatch(
            items=tuple(TextUpdate(id=item.id, text="rewritten") for item in items)
        )


def _document(text: str = "hello") -> SubtitleDocument:
    return SubtitleDocument(
        source_language="en",
        cues=(
            SubtitleCue(
                id="cue000001",
                start_ms=0,
                end_ms=1_000,
                source_text=text,
            ),
        ),
    )


def test_correction_rejects_loss_of_protected_content() -> None:
    service = SubtitleService(_UnsafeCorrectionLlm())

    corrected = service.correct(
        _document("visit https://example.test with 42"),
        max_change_ratio=1.0,
    )

    assert corrected.cues[0].corrected_text is None
    assert corrected.cues[0].warnings == ("correction_failed:SubtitleValidationError",)


def test_cleanup_is_explicit_and_deterministic() -> None:
    service = SubtitleService(FakeLlm())

    cleaned = service.cleanup(
        _document("um hello hello [noise]"),
        fillers=("um ",),
        non_speech_markers=("[noise]",),
        collapse_repetitions=True,
    )

    assert cleaned.cues[0].source_text == "um hello hello [noise]"
    assert cleaned.cues[0].corrected_text == "hello"


def test_translation_qc_marks_repairable_deterministic_failures() -> None:
    service = SubtitleService(
        FakeLlm(),
        glossary=Glossary(entries=(GlossaryEntry(source="API", target="接口"),)),
    )
    document = SubtitleDocument(
        source_language="en",
        target_language="zh",
        cues=(
            SubtitleCue(
                id="cue000001",
                start_ms=0,
                end_ms=2_000,
                source_text="API version 42 at https://example.test",
                translated_text="API version",
            ),
        ),
    )

    report = service.check_quality(document)
    codes = {issue.code for issue in report.issues}

    assert {"number_missing", "protected_content_missing", "glossary_missing"} <= codes
    assert report.has_repairable_issues


def test_formatting_is_output_only_and_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    document = _document("one two three four")

    assert format_text(document.cues[0].source_text, 8, 2) == (
        "one two",
        "three four",
    )
    assert "\none two\nthree four\n" in render_srt(
        document,
        max_line_chars=8,
        max_lines=2,
    )
    output = write_srt(document, tmp_path / "one.srt")
    with pytest.raises(ExportError, match="already exists"):
        write_srt(document, output)
    assert document.cues[0].source_text == "one two three four"


def test_context_failure_continues_and_processing_log_is_retained(
    tmp_path: Path,
) -> None:
    options = PipelineOptions.model_validate(
        {
            "run": {"keep_workdir": True},
            "output": {"formats": ["json"]},
        }
    )
    services = build_fake_services(options)
    services = services.__class__(
        media=services.media,
        transcription=services.transcription,
        subtitles=SubtitleService(FakeLlm(fail_stage="context")),
    )

    result = run_files(
        (ROOT / "tests/fixtures/fake_input.json",),
        options,
        services,
        tmp_path,
    )

    assert not result.failed
    assert result.succeeded[0].warnings == (
        "context_analysis_fallback:LlmPermanentError",
    )
    assert result.workdir is not None
    file_dir = next(path for path in result.workdir.iterdir() if path.is_dir())
    context = json.loads((file_dir / "content_context.json").read_text("utf-8"))
    events = [
        json.loads(line)
        for line in (file_dir / "processing.log").read_text("utf-8").splitlines()
    ]
    assert context["summary"] == ""
    assert {event["stage"] for event in events} >= {
        "context_analysis",
        "transcription",
        "translation",
        "export",
    }


def test_refine_accepts_canonical_subtitle_json(tmp_path: Path) -> None:
    input_path = tmp_path / "existing.json"
    input_path.write_text(_document().model_dump_json(), encoding="utf-8")
    options = PipelineOptions.model_validate(
        {
            "context_analysis": {"enabled": False},
            "correction": {"enabled": False},
            "translation": {"enabled": False},
            "repair": {"enabled": False},
            "output": {"formats": ["json"]},
        }
    )

    result = refine_srt(input_path, options, tmp_path / "out")

    assert result.subtitle == _document()
    assert result.output_paths[0].is_file()
