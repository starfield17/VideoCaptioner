# Desktop GUI

The PySide6 desktop application is a thin adapter over
`captioner.workflow.api`. It does not call FFmpeg, ASR workers, model SDKs, or
LLM providers directly.

## Install and start

Create the source-based Conda environment:

```bash
conda env create -f conda/gui.yml
conda run -n captioner-gui captioner-gui
```

Alternatively, install the optional dependencies in an existing Python 3.13
environment:

```bash
python -m pip install -e ".[gui]"
captioner-gui
```

English and the Light theme are the defaults for a new installation. Open
**Settings** to switch between English and Simplified Chinese or between Light
and Dark themes. Existing language and theme choices are retained through Qt
settings. The application selects common CJK fonts on Windows, macOS, and
Linux. A minimal Linux desktop without a CJK font should install its
distribution's Noto CJK package (for example, `fonts-noto-cjk` on
Debian/Ubuntu).

## Workspace

The top toolbar provides **Add file**, **Add folder**, **Scan**, **Start**,
**Cancel**, **Activity log**, **Models**, **Doctor**, and **Settings** actions.
The main workspace contains:

- **Source setup** for run mode, input, output, and ASR profile.
- Parameter tabs for ASR/languages, enabled pipeline stages, output formats,
  and a quick run summary.
- **Jobs** metrics, determinate per-file progress, the pipeline-stage ribbon,
  and output results.

Settings edits the shared LLM, stage batch/worker, model source, offline, and
logging options. Save writes the platform-native `config.toml`. The API key is
intentionally stored as plaintext in that local file and is masked by CLI
inspection and run metadata. Models and Doctor open focused utility windows;
the Activity Log window supports All/Process/Error filters and local export.

## Drag folders and recursive batches

Drag exactly one local media file or folder from Explorer, Finder, or a Linux
file manager anywhere onto the window. Folder selection switches subtitle
refine mode back to the full pipeline, then scans in a worker thread and shows
a preview. Review the supported-file count, extension totals, and first 100
relative paths, choose **Use this input**, then press **Start**. Dropping never
starts an expensive run automatically.

Directory discovery is recursive for both GUI and CLI. Symlink directories are
not followed. Outputs preserve the input-relative directory structure beneath
the selected output root, so files such as `season-1/episode.mp4` and
`season-2/episode.mp4` cannot overwrite each other's subtitles.

Before a run, the GUI shows its file count, mode, profile, stages, formats, and
output path. This confirmation can be disabled there and restored in Settings.
Completion, model download, diagnostics, validation errors, and active-task
window closing use focused dialogs.

ASR profiles contain their default model, device, and quantization choices.
Advanced provider-specific overrides remain in `config.toml`; this keeps the
GUI and CLI on one canonical configuration instead of creating a second
settings format.

## Test

Install `.[gui-test]`, then run:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/gui
python scripts/verify.py
```

The GUI tests cover English/Light defaults, retained language/theme choices,
drag-and-drop, recursive background scanning, worker progress/results,
cancellation, and byte-identical Subtitle JSON from GUI and CLI adapters using
one configuration.
