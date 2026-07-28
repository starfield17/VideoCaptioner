# Qwen3 Python 3.13 compatibility evidence

The repository policy requires first-party code, tests, and tools to run on
Python 3.13. `conda/asr-qwen3.yml` therefore declares Python 3.13 explicitly;
it does not silently create the upstream README's Python 3.12 environment.

The upstream Qwen3-ASR README currently recommends a fresh Python 3.12
environment. That recommendation is treated as a compatibility risk, not as
permission to downgrade this project. The provider Worker imports `qwen_asr`
inside the Qwen environment during `hello`. An import failure includes the
actual Python version and the original exception, and Doctor reports
`provider_environment=false`. A model-load failure is reported separately;
there is no fallback to another ASR provider.

Run the reproducible checks after creating the environment:

```bash
conda env create -f conda/asr-qwen3.yml
conda run -n captioner-asr-qwen3 python -c \
  "import sys, qwen_asr; print(sys.version); print(qwen_asr.__file__)"
conda run -n captioner-core python -m captioner doctor --provider qwen3-asr
```

The contract tests also verify the Python 3.13 declaration and the explicit
failure message used when the provider import is incompatible.

Verified in this workspace on 2026-07-27 without loading model weights:

```text
Python 3.13.14
qwen-asr 0.0.6
Doctor: ok=true, forced_alignment=True, native_word_timestamps=False
```
