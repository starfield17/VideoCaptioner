"""Build the wheel, verified resources, icons, and frozen application."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    wheel_dir = ROOT / "dist/wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run(
        (
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        )
    )
    wheels = tuple(wheel_dir.glob("captioner-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one project wheel, found {len(wheels)}")
    _run(
        (
            sys.executable,
            str(ROOT / "scripts/prepare_release_assets.py"),
            "--output",
            str(ROOT / "build/release-resources/runtime"),
            "--wheel",
            str(wheels[0]),
        )
    )
    _run(
        (
            sys.executable,
            str(ROOT / "scripts/build_icons.py"),
            "--output",
            str(ROOT / "build/release-resources/icons"),
        )
    )
    _run(
        (
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(ROOT / "packaging/VideoCaptioner.spec"),
        )
    )
    collection = ROOT / "dist/VideoCaptioner"
    if not collection.is_dir():
        raise SystemExit("PyInstaller collection is missing")
    for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / notice, collection / notice)
    app_resources = ROOT / "dist/VideoCaptioner.app/Contents/Resources"
    if app_resources.is_dir():
        for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(ROOT / notice, app_resources / notice)
    _run(
        (
            sys.executable,
            str(ROOT / "scripts/smoke_packaged.py"),
            "--directory",
            str(collection),
        )
    )
    return 0


def _run(command: tuple[str, ...]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
