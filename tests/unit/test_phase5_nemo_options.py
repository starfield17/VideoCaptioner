import pytest
from pydantic import ValidationError

from captioner.workflow.api import NemoAsrOptions, PipelineOptions


def test_nemo_options_are_discriminated_and_strict() -> None:
    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "nemo-asr",
                "language": "en",
                "nemo": {
                    "model": "nvidia/parakeet-tdt-0.6b-v3",
                    "device": "cpu",
                    "batch_size": 2,
                },
            }
        }
    )

    assert isinstance(options.asr, NemoAsrOptions)
    assert options.asr.nemo.device == "cpu"
    assert options.asr.nemo.batch_size == 2

    with pytest.raises(ValidationError):
        PipelineOptions.model_validate(
            {
                "asr": {
                    "provider": "nemo-asr",
                    "nemo": {"unknown": True},
                }
            }
        )


def test_nemo_defaults_to_parakeet_v3() -> None:
    options = PipelineOptions.model_validate({"asr": {"provider": "nemo-asr"}})

    assert isinstance(options.asr, NemoAsrOptions)
    assert options.asr.nemo.model == "nvidia/parakeet-tdt-0.6b-v3"
    assert options.asr.nemo.device == "auto"
