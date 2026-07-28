"""Temporary run workspace lifecycle."""

import json
import shutil
import tempfile
from pathlib import Path

from captioner.shared.errors import ExportError
from captioner.workflow.options import PipelineOptions


class RunWorkspace:
    """Own one isolated temporary directory for a complete run."""

    def __init__(self, keep: bool) -> None:
        self.keep = keep
        self.root = Path(tempfile.mkdtemp(prefix="captioner-run-"))

    def write_run_metadata(
        self, input_paths: tuple[Path, ...], options: PipelineOptions
    ) -> None:
        payload = {
            "inputs": [str(path) for path in input_paths],
            "options": options.model_dump(mode="json"),
        }
        self._atomic_write(self.root / "run.json", json.dumps(payload, indent=2))

    def file_dir(self, index: int, input_path: Path) -> Path:
        safe_stem = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in input_path.stem
        ).strip("_")
        directory = self.root / f"{index:03d}-{safe_stem or 'input'}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _atomic_write(self, path: Path, content: str) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary_path.write_text(content + "\n", encoding="utf-8")
            temporary_path.replace(path)
        except OSError as exc:
            raise ExportError(f"could not write workspace metadata: {path}") from exc
