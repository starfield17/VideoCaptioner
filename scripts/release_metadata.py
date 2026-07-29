"""Validate release versions and generate checksums/manifests."""

import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_ASSET_BYTES = 2 * 1024 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version")
    version.add_argument("--ref", default="")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("directory", type=Path)
    finalize.add_argument("--version", required=True)
    arguments = parser.parse_args()
    if arguments.command == "version":
        value = project_version()
        ref = arguments.ref
        if ref.startswith("refs/tags/"):
            tag = ref.removeprefix("refs/tags/")
            if tag != f"v{value}" and not tag.startswith(f"v{value}-rc."):
                raise SystemExit(
                    f"tag {tag!r} does not match project version {value!r}"
                )
        print(value)
        return 0
    finalize_assets(arguments.directory, arguments.version)
    return 0


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as source:
        document = tomllib.load(source)
    return str(document["project"]["version"])


def finalize_assets(directory: Path, version: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    assets = tuple(
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS", "release-manifest.json"}
    )
    if not assets:
        raise SystemExit("no release assets were found")
    manifest: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    for asset in assets:
        size = asset.stat().st_size
        if size >= MAX_ASSET_BYTES:
            raise SystemExit(f"release asset exceeds GitHub's 2 GiB limit: {asset}")
        relative = asset.relative_to(directory).as_posix()
        checksum = _sha256(asset)
        checksum_lines.append(f"{checksum}  {relative}")
        manifest.append(
            {
                "name": relative,
                "size": size,
                "sha256": checksum,
                "unsigned": "-unsigned" in asset.name,
            }
        )
    (directory / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    (directory / "release-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "captioner-release.v1",
                "version": version,
                "commit": os.environ.get("GITHUB_SHA"),
                "assets": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
