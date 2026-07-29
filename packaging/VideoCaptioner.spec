# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

ROOT = Path(SPECPATH).parent
SOURCE = ROOT / "src"
RUNTIME = ROOT / "build" / "release-resources" / "runtime"
ICONS = ROOT / "build" / "release-resources" / "icons"

if not RUNTIME.is_dir():
    raise SystemExit("release runtime resources are missing")

common = {
    "pathex": [str(ROOT), str(SOURCE)],
    "datas": [(str(RUNTIME), "runtime")],
    "hiddenimports": [],
    "hookspath": [],
    "hooksconfig": {},
    "runtime_hooks": [],
    "excludes": ["tkinter"],
    "noarchive": False,
    "optimize": 1,
}

gui_analysis = Analysis([str(SOURCE / "captioner/gui/main.py")], **common)
gui_pyz = PYZ(gui_analysis.pure)
gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="VideoCaptioner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(
        ICONS
        / ("VideoCaptioner.ico" if sys.platform == "win32" else "VideoCaptioner.icns")
    ),
)

cli_analysis = Analysis([str(SOURCE / "captioner/__main__.py")], **common)
cli_pyz = PYZ(cli_analysis.pure)
cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="captioner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

fake_analysis = Analysis([str(ROOT / "workers/fake/__main__.py")], **common)
fake_pyz = PYZ(fake_analysis.pure)
fake_exe = EXE(
    fake_pyz,
    fake_analysis.scripts,
    [],
    exclude_binaries=True,
    name="captioner-worker-fake",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

collection = COLLECT(
    gui_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    cli_exe,
    cli_analysis.binaries,
    cli_analysis.datas,
    fake_exe,
    fake_analysis.binaries,
    fake_analysis.datas,
    name="VideoCaptioner",
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name="VideoCaptioner.app",
        icon=str(ICONS / "VideoCaptioner.icns"),
        bundle_identifier="io.github.starfield17.VideoCaptioner",
        info_plist={
            "CFBundleDisplayName": "VideoCaptioner",
            "LSMinimumSystemVersion": "13.0",
            "NSHighResolutionCapable": True,
        },
    )
