"""Smoke-test a frozen release directory without system Python or Conda."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.directory.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    cli = directory / f"captioner{suffix}"
    gui = directory / f"VideoCaptioner{suffix}"
    ffmpeg = directory / "_internal/runtime" / f"ffmpeg{suffix}"
    for executable in (cli, gui, ffmpeg):
        if not executable.is_file():
            raise SystemExit(f"missing packaged executable: {executable}")

    with tempfile.TemporaryDirectory(prefix="captioner-package-smoke-") as temp:
        temporary = Path(temp)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PATH"] = os.pathsep.join((str(directory), os.defpath))
        environment["QT_QPA_PLATFORM"] = "offscreen"
        environment["XDG_CONFIG_HOME"] = str(temporary / "config")
        environment["XDG_CACHE_HOME"] = str(temporary / "cache")
        environment["XDG_DATA_HOME"] = str(temporary / "data")
        environment["XDG_STATE_HOME"] = str(temporary / "state")
        environment["LOCALAPPDATA"] = str(temporary / "local")

        _run((str(cli), "--version"), environment)
        runtimes = _run((str(cli), "runtimes", "list"), environment)
        payload = json.loads(runtimes.stdout)
        if len(payload["runtimes"]) != 3:
            raise SystemExit("packaged runtime catalog is incomplete")
        _run((str(ffmpeg), "-version"), environment)
        output = temporary / "output"
        _run(
            (
                str(cli),
                "run",
                str(ROOT / "tests/fixtures/fake_input.json"),
                "--asr-profile",
                "fake",
                "--output-dir",
                str(output),
            ),
            environment,
        )
        if not (output / "fake_input.srt").is_file():
            raise SystemExit("packaged fake pipeline did not produce SRT")
        process = subprocess.Popen(
            (str(gui),),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        if process.returncode not in {0, -15, 1}:
            _stdout, stderr = process.communicate()
            raise SystemExit(f"packaged GUI failed: {stderr}")
    print("Packaged smoke: PASS")
    return 0


def _run(
    command: tuple[str, ...], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


if __name__ == "__main__":
    raise SystemExit(main())
