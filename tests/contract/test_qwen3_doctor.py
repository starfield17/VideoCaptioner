import pytest

from captioner.shared.errors import ProviderUnavailableError
from captioner.transcription.capabilities import AsrCapabilities
from captioner.workflow import doctor as doctor_module
from captioner.workflow.api import PipelineOptions
from captioner.workflow.doctor import run_doctor


def test_qwen3_doctor_records_python313_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableQwenClient:
        def __init__(self) -> None:
            pass

        def probe(self) -> AsrCapabilities:
            raise ProviderUnavailableError(
                "qwen-asr import failed under Python 3.13: simulated upstream failure"
            )

    monkeypatch.setattr(doctor_module, "Qwen3WorkerClient", UnavailableQwenClient)
    report = run_doctor(
        PipelineOptions.model_validate({"asr": {"provider": "qwen3-asr"}}),
        provider="qwen3-asr",
    )

    assert not report.checks["provider_environment"]
    assert "Python 3.13" in report.details["provider_environment"]
    assert "simulated upstream failure" in report.details["provider_environment"]


def test_qwen3_doctor_accepts_a_compatible_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AvailableQwenClient:
        def __init__(self) -> None:
            pass

        def probe(self) -> AsrCapabilities:
            return AsrCapabilities(
                native_word_timestamps=False,
                forced_alignment=True,
                language_detection=True,
                initial_prompt=True,
                internal_vad=False,
                supported_languages=None,
            )

    monkeypatch.setattr(doctor_module, "Qwen3WorkerClient", AvailableQwenClient)
    report = run_doctor(
        PipelineOptions.model_validate({"asr": {"provider": "qwen3-asr"}}),
        provider="qwen3-asr",
    )

    assert report.checks["provider_environment"]
    assert "forced_alignment=True" in report.details["provider_environment"]
