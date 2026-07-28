import json
import os
from pathlib import Path

import pytest

from captioner.workflow.api import PipelineOptions, build_services, run_files


def test_qwen3_real_vertical_loop(tmp_path: Path) -> None:
    model = os.environ.get("CAPTIONER_QWEN3_MODEL")
    input_value = os.environ.get("CAPTIONER_QWEN3_INPUT")
    if not model or not input_value:
        pytest.skip(
            "set CAPTIONER_QWEN3_MODEL and CAPTIONER_QWEN3_INPUT to run the "
            "real-model integration test"
        )
    input_path = Path(input_value)
    if not input_path.is_file():
        pytest.skip(f"integration input does not exist: {input_path}")

    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "qwen3-asr",
                "language": os.environ.get("CAPTIONER_QWEN3_LANGUAGE", "auto"),
                "qwen3": {
                    "model": model,
                    "device": os.environ.get("CAPTIONER_QWEN3_DEVICE", "cuda:0"),
                    "dtype": os.environ.get("CAPTIONER_QWEN3_DTYPE", "bfloat16"),
                    "forced_aligner_model": os.environ.get(
                        "CAPTIONER_QWEN3_ALIGNER",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                    ),
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
    assert raw_document["timing_origin"] == "forced_alignment"
    assert raw_document["provider"] == "qwen3-asr"
    assert raw_document["words"]
