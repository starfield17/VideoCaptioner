from pathlib import Path

from captioner.workflow.api import (
    PipelineOptions,
    build_fake_services,
    run_files,
)

ROOT = Path(__file__).resolve().parents[2]


def test_fake_pipeline_matches_golden_srt_and_json(tmp_path: Path) -> None:
    options = PipelineOptions.model_validate({"run": {"keep_workdir": True}})
    result = run_files(
        (ROOT / "tests/fixtures/fake_input.json",),
        options,
        build_fake_services(options),
        tmp_path,
    )
    assert not result.failed
    assert result.workdir is not None
    assert (tmp_path / "fake_input.srt").read_text(encoding="utf-8") == (
        ROOT / "tests/golden/expected/fake_input.srt"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "fake_input.subtitle.json").read_text(encoding="utf-8") == (
        ROOT / "tests/golden/expected/fake_input.subtitle.json"
    ).read_text(encoding="utf-8")
    assert (result.workdir / "001-fake_input" / "quality.json").is_file()
