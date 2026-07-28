import json
from pathlib import Path

from pytest import CaptureFixture

from captioner.cli.main import main


def test_run_dry_run_is_versioned_and_has_no_side_effects(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    output = tmp_path / "output"

    result = main(
        (
            "run",
            "tests/fixtures/fake_input.json",
            "--asr-profile",
            "fake",
            "--output-dir",
            str(output),
            "--dry-run",
        )
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "cli.v1"
    assert payload["command"] == "run"
    assert not output.exists()


def test_directory_dry_run_lists_recursive_inputs(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "first.json"
    second = nested / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    assert (
        main(
            (
                "run",
                str(tmp_path),
                "--asr-profile",
                "fake",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            )
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["input_root"] == str(tmp_path)
    assert payload["inputs"] == [str(first), str(second)]


def test_config_show_masks_secret(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[llm]\n"
        'provider = "openai-compatible"\n'
        'api_key = "local-secret"\n'
        'base_url = "https://api.example.com"\n',
        encoding="utf-8",
    )

    assert main(("config", "show", "--config", str(config))) == 0
    output = capsys.readouterr().out
    assert "local-secret" not in output
    assert "**********" in output


def test_doctor_rejects_conflicting_provider_and_profile(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[logging]\nfile = false\nconsole = false\n",
        encoding="utf-8",
    )

    result = main(
        (
            "doctor",
            "--config",
            str(config),
            "--provider",
            "faster-whisper",
            "--asr-profile",
            "qwen3-1.7b",
        )
    )

    assert result == 2


def test_models_path_uses_versioned_json(capsys: CaptureFixture[str]) -> None:
    assert main(("models", "path")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "cli.v1"
    assert payload["command"] == "models"
