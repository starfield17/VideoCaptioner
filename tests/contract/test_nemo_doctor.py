import pytest

from captioner.shared.errors import ProviderUnavailableError
from captioner.transcription.capabilities import AsrCapabilities
from captioner.workflow import doctor as doctor_module
from captioner.workflow.api import PipelineOptions
from captioner.workflow.doctor import run_doctor


def test_nemo_doctor_records_python313_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableNemoClient:
        def probe(self) -> AsrCapabilities:
            raise ProviderUnavailableError(
                "nemo-toolkit import failed under Python 3.13: simulated failure"
            )

    monkeypatch.setattr(doctor_module, "NemoWorkerClient", UnavailableNemoClient)
    report = run_doctor(
        PipelineOptions.model_validate({"asr": {"provider": "nemo-asr"}}),
        provider="nemo-asr",
    )

    assert not report.checks["provider_environment"]
    assert "Python 3.13" in report.details["provider_environment"]
    assert "simulated failure" in report.details["provider_environment"]


def test_nemo_doctor_accepts_a_compatible_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AvailableNemoClient:
        def probe(self) -> AsrCapabilities:
            return AsrCapabilities(
                native_word_timestamps=True,
                forced_alignment=False,
                language_detection=True,
                initial_prompt=False,
                internal_vad=False,
                supported_languages=("en",),
            )

    monkeypatch.setattr(doctor_module, "NemoWorkerClient", AvailableNemoClient)
    report = run_doctor(
        PipelineOptions.model_validate({"asr": {"provider": "nemo-asr"}}),
        provider="nemo-asr",
    )

    assert report.checks["provider_environment"]
    assert "native_word_timestamps=True" in report.details["provider_environment"]
