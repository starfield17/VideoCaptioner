# Third-party notices

VideoCaptioner release archives contain third-party components. Their upstream
copyright notices and license files remain authoritative.

| Component | Purpose | License/source |
| --- | --- | --- |
| CPython | Frozen application runtime | Python-2.0, https://www.python.org |
| PySide6 / Qt | Desktop user interface | LGPL-3.0-only or commercial/GPL options, https://doc.qt.io/qtforpython-6/ |
| PyInstaller | Freezing and bootloader | GPL-2.0-or-later with bootloader exception, https://pyinstaller.org |
| micromamba | Managed provider environments | BSD-3-Clause, https://github.com/mamba-org/micromamba-releases |
| FFmpeg from imageio-ffmpeg | Media decoding and conversion | GPL build; exact version and hash are in `runtime/manifest.json`, https://ffmpeg.org |
| imageio-ffmpeg | FFmpeg binary distribution | BSD-2-Clause, https://github.com/imageio/imageio-ffmpeg |
| Pydantic | Data validation | MIT, https://github.com/pydantic/pydantic |
| huggingface_hub | Model downloads | Apache-2.0, https://github.com/huggingface/huggingface_hub |
| platformdirs | Platform-native paths | MIT, https://github.com/tox-dev/platformdirs |

ASR frameworks and model weights are downloaded separately at the user's
request and are not part of the desktop archive. The Runtime Manager displays
whether a recipe is stable, experimental, or unavailable. Model repositories
retain their own licenses and acceptable-use terms.
