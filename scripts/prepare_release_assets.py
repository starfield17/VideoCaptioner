"""Download verified bootstrap binaries for a native release build."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "packaging/bootstrap-assets.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--platform")
    arguments = parser.parse_args()
    platform_key = arguments.platform or _platform_key()
    catalog = cast(
        dict[str, object],
        json.loads(CATALOG_PATH.read_text(encoding="utf-8")),
    )
    micromamba_catalog = cast(dict[str, dict[str, str]], catalog["micromamba"])
    ffmpeg_catalog = cast(dict[str, dict[str, str]], catalog["ffmpeg"])
    try:
        micromamba = micromamba_catalog[platform_key]
        ffmpeg = ffmpeg_catalog[platform_key]
    except KeyError as exc:
        raise SystemExit(f"unsupported release platform: {platform_key}") from exc

    output = arguments.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    temporary = output.parent / ".release-downloads"
    temporary.mkdir(parents=True, exist_ok=True)

    suffix = ".exe" if platform_key.startswith("windows-") else ""
    micromamba_path = output / f"micromamba{suffix}"
    micromamba_url = (
        "https://github.com/mamba-org/micromamba-releases/releases/download/"
        f"{catalog['micromamba_version']}/{micromamba['asset']}"
    )
    _download(micromamba_url, micromamba_path, micromamba["sha256"])

    wheel_path = temporary / Path(ffmpeg["url"]).name
    _download(ffmpeg["url"], wheel_path, ffmpeg["sha256"])
    ffmpeg_name = f"ffmpeg{suffix}"
    with zipfile.ZipFile(wheel_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if "/binaries/ffmpeg-" in name and not name.endswith("/")
        ]
        if len(candidates) != 1:
            raise SystemExit("FFmpeg wheel did not contain one binary")
        with (
            archive.open(candidates[0]) as source,
            (output / ffmpeg_name).open("wb") as destination,
        ):
            shutil.copyfileobj(source, destination)

    project_wheel = arguments.wheel.resolve()
    if not project_wheel.is_file():
        raise SystemExit(f"project wheel does not exist: {project_wheel}")
    shutil.copy2(project_wheel, output / project_wheel.name)
    if os.name != "nt":
        executable = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        micromamba_path.chmod(executable)
        (output / ffmpeg_name).chmod(executable)

    manifest = {
        "schema_version": "captioner-release-resources.v1",
        "platform": platform_key,
        "micromamba_version": catalog["micromamba_version"],
        "ffmpeg_version": catalog["ffmpeg_version"],
        "files": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _platform_key() -> str:
    os_name = {
        "linux": "linux",
        "darwin": "macos",
        "windows": "windows",
    }.get(platform.system().lower())
    architecture = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(platform.machine().lower())
    if os_name is None or architecture is None:
        raise SystemExit(
            f"unsupported release host: {platform.system()}/{platform.machine()}"
        )
    return f"{os_name}-{architecture}"


def _download(url: str, target: Path, expected_sha256: str) -> None:
    if target.is_file() and _sha256(target) == expected_sha256:
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "VideoCaptioner-CI"})
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        temporary.open("wb") as destination,
    ):
        shutil.copyfileobj(response, destination)
    actual = _sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
