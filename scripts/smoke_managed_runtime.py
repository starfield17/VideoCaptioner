"""Install and exercise the stable Faster Whisper managed runtime."""

import argparse
import hashlib
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

SAMPLE_URL = (
    "https://raw.githubusercontent.com/ggerganov/whisper.cpp/"
    "a630b35c6fc02c8879f751ec3f39a61327f01dc7/samples/jfk.wav"
)
SAMPLE_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.directory.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    cli = directory / f"captioner{suffix}"
    with tempfile.TemporaryDirectory(prefix="captioner-runtime-smoke-") as temp:
        temporary = Path(temp)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PATH"] = os.pathsep.join((str(directory), os.defpath))
        environment["XDG_CONFIG_HOME"] = str(temporary / "config")
        environment["XDG_CACHE_HOME"] = str(temporary / "cache")
        environment["XDG_DATA_HOME"] = str(temporary / "data")
        environment["XDG_STATE_HOME"] = str(temporary / "state")
        environment["LOCALAPPDATA"] = str(temporary / "local")
        _run((str(cli), "runtimes", "install", "faster-whisper"), environment, 1200)

        sample = temporary / "jfk.wav"
        _download_sample(sample)
        config = temporary / "tiny.toml"
        config.write_text(
            "[asr]\n"
            'provider = "faster-whisper"\n'
            'language = "en"\n'
            "[asr.faster_whisper]\n"
            'model = "tiny"\n'
            'device = "cpu"\n'
            'compute_type = "int8"\n'
            "[correction]\n"
            "enabled = false\n"
            "[translation]\n"
            "enabled = false\n"
            "[logging]\n"
            "file = false\n"
            "console = true\n",
            encoding="utf-8",
        )
        output = temporary / "output"
        _run(
            (
                str(cli),
                "run",
                str(sample),
                "--config",
                str(config),
                "--output-dir",
                str(output),
            ),
            environment,
            1200,
        )
        srt = output / "jfk.srt"
        if not srt.is_file() or "-->" not in srt.read_text(encoding="utf-8"):
            raise SystemExit("managed Faster Whisper smoke produced no subtitles")
    print("Managed Faster Whisper smoke: PASS")
    return 0


def _run(
    command: tuple[str, ...],
    environment: dict[str, str],
    timeout: int,
) -> None:
    subprocess.run(command, env=environment, check=True, timeout=timeout)


def _download_sample(path: Path) -> None:
    request = urllib.request.Request(
        SAMPLE_URL, headers={"User-Agent": "VideoCaptioner-CI"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        path.open("wb") as destination,
    ):
        destination.write(response.read())
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SAMPLE_SHA256:
        raise SystemExit(f"sample checksum mismatch: {actual}")


if __name__ == "__main__":
    raise SystemExit(main())
