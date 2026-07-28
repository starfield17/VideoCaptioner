# Media

`FakeMediaService` copies JSON fixtures, while `FfmpegMediaService` prepares
mono 16 kHz audio for ASR. Phase 3 adds an optional synchronous
`VoiceSeparator` port. It writes `vocals.wav` beside the prepared asset and
never changes the original input or the ASR Worker Protocol.

Set `CAPTIONER_VOICE_SEPARATION_COMMAND` to an executable that accepts
`<prepared-input> <vocals-output>` when separation is enabled. A non-required
failure keeps the prepared input; a required failure marks the current file as
failed.
