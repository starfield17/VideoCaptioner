# VideoCaptioner

The built-in ASR default is Faster Whisper Turbo with automatic
GPU `int8_float16` / CPU `int8` selection. Run `captioner config path`,
`captioner config show`, and `captioner models list` to inspect platform-native
paths and effective settings. See
[model defaults and quantization](docs/model-defaults-and-quantization.md) for
the selection evidence and acceptance rule.

No file is written when the default config is absent. `captioner config init`
creates it explicitly. Platform-native defaults are:

| Platform | Config | Models | Logs |
| --- | --- | --- | --- |
| Linux | `~/.config/VideoCaptioner/config.toml` | `~/.cache/VideoCaptioner/models` | `~/.local/state/VideoCaptioner/log` |
| macOS | `~/Library/Application Support/VideoCaptioner/config.toml` | `~/Library/Caches/VideoCaptioner/models` | `~/Library/Logs/VideoCaptioner` |
| Windows | `%LOCALAPPDATA%\\VideoCaptioner\\config.toml` | `%LOCALAPPDATA%\\VideoCaptioner\\Cache\\models` | `%LOCALAPPDATA%\\VideoCaptioner\\Logs` |

The built-in LLM stage defaults are correction 30 items / 8 workers,
translation 30 / 16, and repair 20 / 8. Rotating JSONL logs default to `INFO`,
10 MiB per file, and five backups; `DEBUG`, `WARNING`, `ERROR`, `ALL`, and
`OFF` are also supported.

The clean-room Video Captioner architecture: a synchronous ASR/LLM pipeline
with Fake, Faster Whisper, and Qwen3 ASR Workers. It turns a JSON fixture or
media input into SRT, VTT, and Subtitle JSON, with optional voice separation
and an existing-SRT `refine` command.

## Quick start

```bash
conda env create -f conda/core.yml
conda run -n captioner-core python scripts/verify.py
conda run -n captioner-core python -m captioner doctor
conda run -n captioner-core python -m captioner run tests/fixtures/fake_input.json --asr-profile fake --output-dir ./out
```

For the bilingual desktop console:

```bash
conda env create -f conda/gui.yml
conda run -n captioner-gui captioner-gui
```

The English-first GUI provides full-pipeline, transcribe-only, and
existing-subtitle refine runs; recursive file/folder drag-and-drop; Light and
Dark themes; shared settings; model inspection/downloads; Doctor diagnostics;
progress and cooperative cancellation. It uses the same Workflow/Application
API and configuration as the CLI. See [GUI usage and testing](docs/gui.md).

The fake input format is documented by `tests/fixtures/fake_input.json`. The
Qwen3 environment and its Python 3.13 Doctor checks are documented in
`docs/qwen3-python313-compatibility.md`.

## Real E2E audit

`scripts/run_real_e2e.py` runs one configured real ASR provider over a media
directory and retains sanitized logs, intermediate artifacts, outputs, checksums,
and a machine-readable summary. Local ignored TOML files may provide an LLM
`api_key`, `base_url`, and `structured_output_mode`; direct keys are represented
as `SecretStr` and are masked in run metadata. If a CUDA or shared-library error
occurs, the runner records that single GPU failure and retries once on CPU.
CUDA-to-CPU fallback events are retained in `runtime_events`, and a record
directory containing prior outputs is rejected before another expensive run.
