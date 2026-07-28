import json
import os
from pathlib import Path

import pytest

from captioner.workflow.api import PipelineOptions, build_services, run_files


def test_faster_whisper_real_vertical_loop(tmp_path: Path) -> None:
    model = os.environ.get("CAPTIONER_FASTER_WHISPER_MODEL")
    input_value = os.environ.get("CAPTIONER_FASTER_WHISPER_INPUT")
    if not model or not input_value:
        pytest.skip(
            "set CAPTIONER_FASTER_WHISPER_MODEL and "
            "CAPTIONER_FASTER_WHISPER_INPUT to run the real-model integration test"
        )
    input_path = Path(input_value)
    if not input_path.is_file():
        pytest.skip(f"integration input does not exist: {input_path}")

    device = os.environ.get("CAPTIONER_FASTER_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get(
        "CAPTIONER_FASTER_WHISPER_COMPUTE_TYPE",
        "int8" if device == "cpu" else "float16",
    )
    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "faster-whisper",
                "language": os.environ.get("CAPTIONER_FASTER_WHISPER_LANGUAGE", "auto"),
                "faster_whisper": {
                    "model": model,
                    "device": device,
                    "compute_type": compute_type,
                },
            },
            "correction": {"enabled": False},
            "translation": {"enabled": False},
            "run": {"keep_workdir": True},
        }
    )

    result = run_files(
        (input_path,), options, build_services(options), tmp_path / "outputs"
    )

    assert not result.failed, result.failed
    assert len(result.succeeded) == 1
    assert result.succeeded[0].subtitle.cues
    assert result.succeeded[0].workdir is not None
    raw_path = next(result.succeeded[0].workdir.glob("001-*/transcript.raw.json"))
    raw_document = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_document["timing_origin"] == "asr_native"
    assert raw_document["provider"] == "faster-whisper"
    assert raw_document["words"]
    srt_path = tmp_path / "outputs" / f"{input_path.stem}.srt"
    srt_text = srt_path.read_text(encoding="utf-8")
    assert " --> " in srt_text
    assert srt_text.strip()
