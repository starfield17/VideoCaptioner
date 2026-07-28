import json
import shutil
from pathlib import Path

import pytest

from captioner.shared.errors import OperationCancelled
from captioner.workflow.api import (
    CancellationToken,
    ExecutionContext,
    PipelineOptions,
    ProgressEvent,
    ProgressKind,
    build_fake_services,
    execute_run,
    load_options,
    plan_operation,
    render_options_toml,
    run_files,
    save_options,
)


def test_canonical_config_round_trip_includes_secret_but_no_runtime_paths(
    tmp_path: Path,
) -> None:
    options = PipelineOptions.model_validate(
        {
            "asr": {
                "provider": "faster-whisper",
                "faster_whisper": {
                    "model": "turbo",
                    "model_path": "/temporary/model",
                },
            },
            "llm": {
                "provider": "openai-compatible",
                "api_key": "local-secret",
                "base_url": "https://api.example.com",
            },
        }
    )

    rendered = render_options_toml(options)
    path = save_options(options, tmp_path / "config.toml")
    loaded = load_options(path)

    assert 'api_key = "local-secret"' in rendered
    assert "model_path" not in rendered
    assert loaded.llm.api_key is not None
    assert loaded.llm.api_key.get_secret_value() == "local-secret"
    assert loaded.asr.provider == "faster-whisper"


def test_plan_operation_has_no_output_side_effect(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fake_input.json")
    output = tmp_path / "not-created"

    plan = plan_operation(
        "run",
        fixture,
        output,
        PipelineOptions.model_validate({"asr": {"provider": "fake"}}),
    )

    assert plan.inputs == (fixture,)
    assert plan.input_root is None
    assert not output.exists()


def test_directory_plan_exposes_recursive_input_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    fixture = nested / "fixture.json"
    fixture.write_text("{}", encoding="utf-8")

    plan = plan_operation(
        "run",
        tmp_path,
        tmp_path / "output",
        PipelineOptions.model_validate({"asr": {"provider": "fake"}}),
    )

    assert plan.inputs == (fixture,)
    assert plan.input_root == tmp_path


def test_execute_run_preserves_recursive_output_structure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    first_dir = source / "first"
    second_dir = source / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    fixture = Path("tests/fixtures/fake_input.json")
    shutil.copyfile(fixture, first_dir / "episode.json")
    shutil.copyfile(fixture, second_dir / "episode.json")
    options = PipelineOptions.model_validate(
        {
            "asr": {"provider": "fake"},
            "logging": {"file": False, "console": False},
        }
    )

    result = execute_run(source, options, tmp_path / "output")

    assert not result.failed
    assert (tmp_path / "output/first/episode.srt").is_file()
    assert (tmp_path / "output/second/episode.srt").is_file()


def test_cancellation_retains_workdir_and_emits_cancelled(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fake_input.json")
    options = PipelineOptions.model_validate(
        {
            "asr": {"provider": "fake"},
            "logging": {"file": False, "console": False},
        }
    )
    token = CancellationToken()
    events: list[ProgressEvent] = []

    def observe(event: ProgressEvent) -> None:
        events.append(event)
        if (
            event.kind is ProgressKind.STAGE_COMPLETED
            and event.stage is not None
            and event.stage.value == "media"
        ):
            token.cancel()

    context = ExecutionContext.create(cancellation=token, observer=observe)

    with pytest.raises(OperationCancelled, match="workdir retained at") as raised:
        run_files(
            (fixture,),
            options,
            build_fake_services(options, context),
            tmp_path / "outputs",
            context,
        )

    workdir = Path(str(raised.value).rsplit(" ", maxsplit=1)[-1])
    assert workdir.is_dir()
    assert events[0].kind is ProgressKind.RUN_STARTED
    assert events[-1].kind is ProgressKind.CANCELLED
    assert not (tmp_path / "outputs" / "fake_input.srt").exists()


def test_progress_events_are_json_serializable() -> None:
    event = ProgressEvent(ProgressKind.RUN_STARTED, file_count=2)

    assert json.dumps(event.__dict__, default=str)
