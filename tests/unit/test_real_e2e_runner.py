# pyright: reportPrivateUsage=false

import logging
from pathlib import Path

import pytest
from scripts.run_real_e2e import (
    _assert_secret_absent,
    _cpu_options,
    _FallbackAuditHandler,
    _is_cuda_failure,
    _require_empty_run_record,
)

from captioner.workflow.options import PipelineOptions


def test_cuda_linker_failure_switches_provider_to_cpu() -> None:
    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "faster-whisper",
                "faster_whisper": {
                    "device": "cuda",
                    "compute_type": "float16",
                },
            }
        }
    )

    cpu_options = _cpu_options(options)

    assert cpu_options.asr.provider == "faster-whisper"
    assert cpu_options.asr.faster_whisper.device == "cpu"
    assert cpu_options.asr.faster_whisper.compute_type == "int8"
    assert _is_cuda_failure("libcudnn.so: cannot open shared object file")
    assert not _is_cuda_failure("LLM returned invalid JSON")


def test_nemo_cpu_fallback_only_changes_provider_device() -> None:
    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "nemo-asr",
                "nemo": {
                    "model": "nvidia/parakeet-tdt-0.6b-v3",
                    "device": "cuda",
                    "batch_size": 2,
                },
            }
        }
    )

    cpu_options = _cpu_options(options)

    assert cpu_options.asr.provider == "nemo-asr"
    assert cpu_options.asr.nemo.device == "cpu"
    assert cpu_options.asr.nemo.batch_size == 2


def test_record_secret_scan_ignores_only_local_configs(tmp_path: Path) -> None:
    options = PipelineOptions.model_validate(
        {
            "llm": {
                "provider": "openai-compatible",
                "api_key": "local-secret",
                "model": "test-model",
            }
        }
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "local.toml").write_text("local-secret", encoding="utf-8")
    _assert_secret_absent(options, tmp_path)

    (tmp_path / "run.log").write_text("local-secret", encoding="utf-8")
    with pytest.raises(RuntimeError, match="credential leaked"):
        _assert_secret_absent(options, tmp_path)


def test_fallback_audit_handler_retains_structured_event() -> None:
    events: list[dict[str, str]] = []
    handler = _FallbackAuditHandler(events)
    record = logging.LogRecord(
        "captioner.transcription",
        logging.WARNING,
        __file__,
        1,
        "CUDA ASR failed; retrying once on CPU",
        (),
        None,
    )
    record.__dict__["fallback"] = "cpu"

    handler.emit(record)

    assert events == [
        {
            "level": "WARNING",
            "message": "CUDA ASR failed; retrying once on CPU",
            "fallback": "cpu",
        }
    ]


def test_existing_record_is_rejected_before_a_rerun(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "existing.srt").write_text("existing", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already contains run results"):
        _require_empty_run_record(tmp_path)
