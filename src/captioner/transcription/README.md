# Transcription

Transcription exposes the provider-independent `TranscriptionRequest` and
`TranscriptDocument` contracts. Provider-specific configuration stays in a
discriminated Workflow option and is passed to one blocking Worker session.

The static providers are Fake, Faster Whisper, Qwen3 ASR, and NVIDIA NeMo.
Qwen3 can use the Qwen3 Forced Aligner; aligned words are marked
`FORCED_ALIGNMENT`. NeMo Parakeet words use provider-native timestamps and are
marked `ASR_NATIVE`. A provider result without real word or segment timing is
rejected rather than being given interpolated timestamps.
