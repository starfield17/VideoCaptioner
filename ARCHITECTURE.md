# Video Captioner Architecture

This clean-room repository is a synchronous modular monolith with explicit
boundaries:

```text
CLI
  -> Workflow
      -> Media       (audio preparation and optional voice separation)
      -> Transcription (Fake, Faster Whisper, or Qwen3 blocking Workers)
      -> Subtitles   (segmentation, correction, translation, QC, refine, export)
          -> LLM contract (Fake or OpenAI-compatible adapter)
      -> Shared
```

## Components

- `src/captioner/shared`: immutable cross-cutting value objects and errors.
- `src/captioner/media`: fake/FFmpeg preparation and optional voice separation.
- `src/captioner/transcription`: provider-independent transcript contracts and
  static Worker adapters.
- `src/captioner/subtitles`: immutable subtitle documents and synchronous stages.
- `src/captioner/llm`: typed LLM contract and deterministic fake implementation.
- `src/captioner/workflow`: options, temporary workspaces, and the pipeline.
- `src/captioner/cli`: thin `doctor`/`run`/`transcribe`/`refine` adapters.
- `workers/common`, `workers/fake`, `workers/faster_whisper`, and `workers/qwen3`:
  blocking NDJSON Workers.

## Dependency rules

Only the Workflow layer composes domain services. Media, Transcription, and
LLM do not import Workflow. Subtitles imports only the public LLM contract;
it does not know about a provider or worker. The CLI imports only Workflow's
public API. Worker code never imports Workflow.

LLM batch parallelism is isolated in `src/captioner/llm/concurrency.py`.
There is no async code, database, file-level parallelism, plugin loader, or
general-purpose scheduler. Voice separation, ASR, subtitle stages, QC, and
export remain synchronous and file-serial.

## Artifact and lifecycle rules

Each run gets a unique temporary workspace. The selected ASR Worker is started
once, loaded once, and shut down once per run. Intermediate JSON artifacts are
written into the per-input workspace directory. Successful workspaces are
removed unless `keep_workdir` is enabled; failed runs retain their path.

The public domain documents store time in integer milliseconds. A provider-native
word result is marked `ASR_NATIVE`; Qwen3 Forced Aligner items are marked
`FORCED_ALIGNMENT`; a segment-only result is marked `SEGMENT_NATIVE`. No stage
after transcription can create or alter timing.
