from pathlib import Path

from captioner.llm.fake import FakeLlm
from captioner.subtitles.exporters.srt import render_srt
from captioner.subtitles.exporters.vtt import render_vtt
from captioner.subtitles.models import SubtitleCue, SubtitleDocument
from captioner.subtitles.service import SubtitleService


def _document() -> SubtitleDocument:
    return SubtitleDocument(
        source_language="en",
        target_language="zh",
        cues=(
            SubtitleCue(
                id="cue1",
                start_ms=0,
                end_ms=1_001,
                source_text="hello\nworld",
                translated_text="你好\n世界",
            ),
            SubtitleCue(
                id="cue2",
                start_ms=1_200,
                end_ms=2_000,
                source_text="second",
                translated_text="第二",
            ),
        ),
    )


def test_bilingual_srt_has_stable_source_then_translation_lines() -> None:
    assert render_srt(_document(), bilingual=True) == (
        "1\n00:00:00,000 --> 00:00:01,001\n"
        "hello\nworld\n你好\n世界\n\n"
        "2\n00:00:01,200 --> 00:00:02,000\nsecond\n第二\n"
    )


def test_vtt_has_stable_header_timestamps_and_line_order() -> None:
    assert render_vtt(_document(), bilingual=True) == (
        "WEBVTT\n\n"
        "1\n00:00:00.000 --> 00:00:01.001\n"
        "hello\nworld\n你好\n世界\n\n"
        "2\n00:00:01.200 --> 00:00:02.000\nsecond\n第二\n"
    )


def test_service_exports_vtt_and_explicit_bilingual_srt(tmp_path: Path) -> None:
    outputs = SubtitleService(FakeLlm()).export(
        _document(),
        tmp_path,
        "clip",
        ("vtt", "bilingual_srt"),
        bilingual=False,
    )

    assert outputs == (tmp_path / "clip.vtt", tmp_path / "clip.bilingual.srt")
    assert "你好" not in (tmp_path / "clip.vtt").read_text(encoding="utf-8")
    assert "你好" in (tmp_path / "clip.bilingual.srt").read_text(encoding="utf-8")
