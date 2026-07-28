# Model defaults and quantization policy

Verified on 2026-07-28. The runtime only selects publisher checkpoints or
quantization implemented by the model's existing SDK. Community GGUF, MLX,
CoreML, and ONNX conversions are listed here for context but are not loaded.

## Default

The no-config profile is Faster Whisper `turbo`, source language `auto`, device
`auto`, and `auto-int8` compute:

- NVIDIA: try CTranslate2 `int8_float16`.
- CPU and the single automatic CUDA fallback: CTranslate2 `int8`.
- An automatic run gets at most one CPU fallback for CUDA, cuDNN, cuBLAS,
  missing `.so`, or out-of-memory load failures.

OpenAI describes Turbo as Large v3 pruned from 32 decoder layers to 4, with a
minor quality degradation and much higher speed. Faster Whisper's published
13-minute benchmark reports Large v2 INT8 at 59 seconds and 2,926 MB VRAM,
versus FP16 at 63 seconds and 4,525 MB. Its CPU benchmark reports Small INT8 at
102 seconds and 1,477 MB RAM, versus FP32 at 157 seconds and 2,257 MB.

Sources:

- <https://huggingface.co/openai/whisper-large-v3-turbo>
- <https://github.com/SYSTRAN/faster-whisper/blob/master/README.md>
- <https://opennmt.net/CTranslate2/quantization.html>

## Available precision variants

Faster Whisper/CTranslate2 supports eight stored precision choices:
`int8`, `int8_float32`, `int8_float16`, `int8_bfloat16`, `int16`, `float16`,
`bfloat16`, and `float32`. Runtime compute additionally supports `auto` and
platform-specific fallback conversion. The project uses only the two INT8
paths above.

Qwen publishes 0.6B and 1.7B ASR checkpoints, but no publisher quantized ASR
checkpoint. Community 4-bit, 8-bit, GGUF, MLX, ONNX, and CoreML conversions
exist and change frequently; they are deliberately excluded. Qwen's official
offline benchmark gives average WER 3.48 for 0.6B and 2.69 for 1.7B. The 0.6B
model is about 29% worse relatively, so it is opt-in rather than the default.

Source: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/README.md>

NVIDIA publishes Parakeet v3 and the 110M English model in full precision, with
no publisher quantized checkpoint. Hugging Face currently groups 57 community
quantizations under Parakeet v3; these use several incompatible runtimes and
lack one publisher-controlled accuracy comparison, so they are excluded. The
110M model is a speed-oriented opt-in: its published mean WER is 7.49. Parakeet
v3 is retained for multilingual use and native timestamps.

Sources:

- <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
- <https://huggingface.co/nvidia/parakeet-tdt_ctc-110m>

## Acceptance rule

A new quantized default must increase WER by no more than 1 percentage point
and no more than 10% relative on the selected validation set. It must also not
be more than 5% slower than the corresponding full-precision run. A model with
no reproducible publisher or local comparison remains opt-in.
