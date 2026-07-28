# Domain glossary

- **TranscriptDocument**: the immutable, provider-independent ASR result.
- **TimedWord**: a word with a real source timestamp in integer milliseconds.
- **TimingOrigin**: the declared source of transcript timing: ASR-native,
  Forced Alignment, or native Segment timing.
- **Forced Aligner**: a provider-side component that aligns ASR text to audio;
  it is the only source allowed to produce `FORCED_ALIGNMENT` words.
- **SubtitleDocument**: the immutable sequence of user-facing subtitle cues.
- **Cue**: a time range and its source/corrected/translated text variants.
- **Glossary**: an ordered immutable set of source-to-target terminology rules
  applied at the translation boundary.
- **VoiceSeparator**: a synchronous Media-boundary adapter that writes a new
  vocal asset without changing the original input or ASR protocol.
- **QualityReport**: deterministic subtitle readability, timing, overlap, and
  completeness findings.
- **Fake LLM**: a deterministic local test double; it never calls a network API.
- **Workspace**: the temporary directory containing run and intermediate files.
