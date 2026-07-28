from pathlib import Path

import pytest
from pydantic import ValidationError

from captioner.shared.errors import ConfigurationError
from captioner.transcription.requests import TimestampRequirement
from captioner.workflow.options import PipelineOptions, load_options


def test_defaults_are_strongly_typed() -> None:
    options = PipelineOptions()
    assert options.asr.provider == "fake"
    assert options.asr.timestamps is TimestampRequirement.REQUIRED
    assert options.output.formats[0].value == "srt"


def test_direct_llm_secret_is_masked_in_serialized_options() -> None:
    options = PipelineOptions.model_validate(
        {
            "llm": {
                "provider": "openai-compatible",
                "api_key": "local-secret",
                "base_url": "https://api.example.com",
                "model": "test-model",
                "structured_output_mode": "json_object",
            }
        }
    )

    dumped = options.model_dump(mode="json")

    assert dumped["llm"]["api_key"] == "**********"
    assert "local-secret" not in options.model_dump_json()


def test_unknown_configuration_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineOptions.model_validate({"run": {"unexpected": True}})


def test_disabled_timestamps_are_rejected_by_phase0() -> None:
    options = PipelineOptions.model_validate({"asr": {"timestamps": "disabled"}})
    with pytest.raises(ConfigurationError):
        PipelineOptions.validate_for_phase0(options)


def test_toml_configuration_is_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "captioner.toml"
    config_path.write_text(
        "[translation]\nenabled = false\n[output]\nformats = ['json']\n",
        encoding="utf-8",
    )
    options = load_options(config_path)
    assert not options.translation.enabled
    assert options.output.formats[0].value == "json"


def test_phase2_parallelism_has_a_hard_one_to_hundred_bound() -> None:
    options = PipelineOptions.model_validate(
        {
            "segmentation": {"parallelism": 100},
            "correction": {"parallelism": 8, "batch_size": 30},
            "translation": {"parallelism": 16, "batch_size": 30},
            "repair": {"parallelism": 8, "batch_size": 20},
            "llm": {"provider": "openai-compatible", "max_attempts": 3},
        }
    )
    assert options.segmentation.parallelism == 100
    assert options.llm.provider == "openai-compatible"

    for field_name in ("segmentation", "correction", "translation", "repair"):
        with pytest.raises(ValidationError):
            PipelineOptions.model_validate({field_name: {"parallelism": 101}})


def test_config_file_parallelism_101_fails_during_preflight(tmp_path: Path) -> None:
    config_path = tmp_path / "captioner.toml"
    config_path.write_text("[translation]\nparallelism = 101\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_options(config_path)


def test_phase3_options_are_strictly_typed() -> None:
    options = PipelineOptions.model_validate(
        {
            "audio": {
                "voice_separation": {
                    "enabled": True,
                    "required": False,
                    "provider": "mdx",
                }
            },
            "subtitle": {
                "max_cps": 17.0,
                "max_line_chars": 42,
            },
            "glossary": {
                "entries": [
                    {"source": "OpenAI", "target": "OpenAI-zh"},
                ]
            },
            "output": {"formats": ["vtt", "bilingual_srt"]},
        }
    )

    assert options.audio.voice_separation.enabled
    assert options.subtitle.max_cps == 17
    assert options.glossary.entries[0].target == "OpenAI-zh"
    assert options.output.formats[0].value == "vtt"
