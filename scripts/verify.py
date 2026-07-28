"""Full architecture and acceptance verification command."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = (
        ["ruff", "format", "--check", "src", "workers", "scripts", "tests"],
        ["ruff", "check", "src", "workers", "scripts", "tests"],
        ["pyright"],
        [sys.executable, "-m", "pytest", "tests/unit"],
        [sys.executable, "-m", "pytest", "tests/contract"],
        [sys.executable, "-m", "pytest", "tests/integration"],
        [sys.executable, "-m", "pytest", "tests/architecture"],
        [sys.executable, "-m", "pytest", "tests/golden"],
    )
    for command in commands:
        _run(command)
    _run_cli_smoke()
    print("Full verify: PASS")
    return 0


def _run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _run_cli_smoke() -> None:
    print("\n$ Fake CLI smoke test", flush=True)
    fixture = ROOT / "tests/fixtures/fake_input.json"
    with tempfile.TemporaryDirectory(prefix="captioner-cli-smoke-") as directory:
        output_dir = Path(directory)
        environment = os.environ.copy()
        source_root = str(ROOT / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            (source_root, str(ROOT), environment.get("PYTHONPATH", ""))
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "captioner",
                "run",
                str(fixture),
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(completed.stdout)
        assert len(summary["succeeded"]) == 1
        assert summary["failed"] == []
        assert (output_dir / "fake_input.srt").is_file()
        assert (output_dir / "fake_input.subtitle.json").is_file()


if __name__ == "__main__":
    raise SystemExit(main())
