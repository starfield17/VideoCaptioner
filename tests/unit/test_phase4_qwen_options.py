import pytest
from pydantic import ValidationError

from captioner.transcription.providers.qwen3 import (
    Qwen3Config,
    Qwen3TranscriptionService,
)
from captioner.workflow.api import PipelineOptions, build_services
from captioner.workflow.options import Qwen3AsrOptions


def test_qwen3_provider_options_are_discriminated_and_strict() -> None:
    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "qwen3-asr",
                "language": "zh",
                "qwen3": {
                    "model": "fixture-asr",
                    "device": "cpu",
                    "dtype": "float32",
                    "forced_aligner_model": "fixture-aligner",
                },
            }
        }
    )

    assert isinstance(options.asr, Qwen3AsrOptions)
    assert options.asr.provider == "qwen3-asr"
    assert options.asr.qwen3.forced_aligner_model == "fixture-aligner"
    assert isinstance(options.asr.qwen3, Qwen3Config)

    with pytest.raises(ValidationError):
        PipelineOptions.model_validate(
            {
                "asr": {
                    "provider": "qwen3-asr",
                    "qwen3": {"unexpected_provider_setting": True},
                }
            }
        )


def test_workflow_composes_qwen3_without_changing_pipeline_stages() -> None:
    options = PipelineOptions.model_validate({"asr": {"provider": "qwen3-asr"}})

    services = build_services(options)

    assert isinstance(services.transcription, Qwen3TranscriptionService)
