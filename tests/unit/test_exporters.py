from pathlib import Path

from captioner.subtitles.exporters.json_export import write_json
from captioner.subtitles.exporters.srt import render_srt, write_srt
from captioner.subtitles.models import SubtitleCue, SubtitleDocument


def test_srt_and_json_exporters_write_expected_boundaries(tmp_path: Path) -> None:
    document = SubtitleDocument(
        source_language="en",
        target_language="zh",
        cues=(
            SubtitleCue(
                id="cue000001",
                start_ms=0,
                end_ms=1_001,
                source_text="hello",
                translated_text="你好",
            ),
        ),
    )
    expected_srt = "1\n00:00:00,000 --> 00:00:01,001\nhello\n你好\n"
    assert render_srt(document) == expected_srt
    srt_path = write_srt(document, tmp_path / "one.srt")
    json_path = write_json(document, tmp_path / "one.subtitle.json")
    assert srt_path.read_text(encoding="utf-8") == expected_srt
    assert json_path.read_text(encoding="utf-8").startswith('{\n  "schema_version"')
