from pathlib import Path

from captioner.workflow.options import PipelineOptions
from captioner.workflow.workspace import RunWorkspace


def test_workspace_writes_metadata_and_can_be_cleaned(tmp_path: Path) -> None:
    workspace = RunWorkspace(keep=False)
    workspace.write_run_metadata((tmp_path / "input.json",), PipelineOptions())
    file_dir = workspace.file_dir(1, Path("input.json"))
    assert (workspace.root / "run.json").is_file()
    assert file_dir.name == "001-input"
    root = workspace.root
    workspace.cleanup()
    assert not root.exists()
