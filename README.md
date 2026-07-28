# VideoCaptioner

The clean-room Video Captioner architecture: a synchronous ASR/LLM pipeline
with Fake, Faster Whisper, and Qwen3 ASR Workers. It turns a JSON fixture or
media input into SRT, VTT, and Subtitle JSON, with optional voice separation
and an existing-SRT `refine` command.

## Quick start

```bash
conda env create -f conda/core.yml
conda run -n captioner-core python scripts/verify.py
conda run -n captioner-core python -m captioner doctor
conda run -n captioner-core python -m captioner run tests/fixtures/fake_input.json --output-dir ./out
```

The fake input format is documented by `tests/fixtures/fake_input.json`. The
Qwen3 environment and its Python 3.13 Doctor checks are documented in
`docs/qwen3-python313-compatibility.md`.
