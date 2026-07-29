# Native packaging and releases

## Local build

Use Python 3.13 and install the GUI plus packaging tools:

```text
python -m pip install -e ".[gui]" build setuptools PyInstaller==6.21.0 Pillow
python scripts/build_release.py
```

The build verifies pinned SHA-256 hashes for micromamba and FFmpeg, freezes GUI,
CLI, and Fake Worker executables, then runs the packaged fake pipeline and an
offscreen GUI smoke test. Build output is under `dist/VideoCaptioner`.

`python scripts/smoke_managed_runtime.py --directory dist/VideoCaptioner`
performs the slower real Faster Whisper test: it creates a fresh managed
environment, downloads a checksum-pinned speech sample and tiny model, and
produces an SRT without using system Python or Conda.

## Release matrix

| Host | GitHub runner | Artifact |
| --- | --- | --- |
| Linux x86_64 | `ubuntu-24.04` | `.tar.zst` |
| Linux aarch64 | `ubuntu-24.04-arm` | `.tar.zst` |
| macOS Intel | `macos-15-intel` | `.app.zip`, `.dmg` |
| macOS Apple Silicon | `macos-15` | `.app.zip`, `.dmg` |
| Windows x86_64 | `windows-2025` | `.exe` installer |
| Windows ARM64 | `windows-11-arm` | `.exe` installer |

Push a tag matching the version in `pyproject.toml`, for example `v0.1.0`.
`v0.1.0-rc.1` creates a prerelease. Manual workflow runs build artifacts
without publishing a GitHub Release.

## Optional signing

Unsigned artifacts are intentionally allowed and include `-unsigned` in the
filename. Configure all secrets for a platform to enable signing:

- Apple: `APPLE_CERTIFICATE_P12`, `APPLE_CERTIFICATE_PASSWORD`,
  `APPLE_SIGNING_IDENTITY`, `APPLE_NOTARY_KEY`, `APPLE_NOTARY_KEY_ID`,
  `APPLE_NOTARY_ISSUER_ID`.
- Windows: `WINDOWS_CERTIFICATE_PFX`, `WINDOWS_CERTIFICATE_PASSWORD`.

Certificate and notary key values are base64-encoded file contents. The
workflow never prints them. Signed macOS builds are notarized and stapled;
Windows program and installer executables are SHA-256 signed and timestamped.

## Runtime behavior

Managed installations use an immutable provider/platform directory and an
atomically replaced `current.json` pointer. A failed repair leaves the previous
runtime selected. Cancelling an install terminates its subprocess and removes
only the incomplete directory.

Windows ARM64 uses a native application and micromamba bootstrap. Faster
Whisper and Qwen use x64 Worker environments through Windows emulation until
their upstream frameworks provide native ARM64 wheels. NeMo has no macOS or
Windows recipe. Runtime and model downloads are separate so uninstalling or
repairing a runtime does not delete model weights.
