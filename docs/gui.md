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

The language selector in the lower-left rail switches between Simplified
Chinese and English immediately. The choice is retained through Qt settings.
The application selects common CJK fonts on Windows, macOS, and Linux. A
minimal Linux desktop without a CJK font should install its distribution's
Noto CJK package (for example, `fonts-noto-cjk` on Debian/Ubuntu).

## Pages

- **Run** selects a complete pipeline, transcription-only run, or subtitle
  refinement. It displays stage events, output files, and a cooperative Cancel
  action. Directory input is supported for ASR runs.
- **Settings** edits the shared LLM, stage batch/worker, model source, offline,
  and logging settings. Save writes the platform-native `config.toml`. The API
  key is intentionally stored as plaintext in that local file and is masked by
  CLI inspection and run metadata.
- **Models** shows the application model directory and downloads catalog models
  with their declared dependencies.
- **Diagnostics** runs Doctor through the shared application API and opens the
  platform-native log directory.

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

The GUI tests cover startup, runtime translation, worker progress/results, and
byte-identical Subtitle JSON from GUI and CLI adapters using one configuration.
