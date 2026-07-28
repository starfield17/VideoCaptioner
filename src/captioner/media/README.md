# Media

Voice separation is an optional command adapter and is disabled by default.
When enabled, set the configured `command_env` variable to a command that
accepts the prepared input path and requested output path as its final two
arguments. The project does not bundle an MDX model. `required = false`
preserves the prepared audio and records a fallback warning when the command
is unavailable or fails.

`FakeMediaService` copies JSON fixtures, while `FfmpegMediaService` prepares
mono 16 kHz audio for ASR. Phase 3 adds an optional synchronous
`VoiceSeparator` port. It writes `vocals.wav` beside the prepared asset and
never changes the original input or the ASR Worker Protocol.

Set `CAPTIONER_VOICE_SEPARATION_COMMAND` to an executable that accepts
`<prepared-input> <vocals-output>` when separation is enabled. A non-required
failure keeps the prepared input; a required failure marks the current file as
failed.
