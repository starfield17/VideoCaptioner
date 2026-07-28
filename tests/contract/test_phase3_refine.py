import json
from pathlib import Path

import pytest

from captioner.cli.main import main


def test_refine_command_reads_existing_srt_and_writes_refined_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "existing.srt"
    input_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,200\n错误句子。\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "refine",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--source-language",
                "zh",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert (output_dir / "existing.refined.srt").is_file()
    assert (output_dir / "existing.refined.subtitle.json").is_file()
    assert payload["quality_issues"] == []
    refined_srt = (output_dir / "existing.refined.srt").read_text(encoding="utf-8")
    assert "正确句子。" in refined_srt
    assert "00:00:00,000 --> 00:00:01,200" in refined_srt
