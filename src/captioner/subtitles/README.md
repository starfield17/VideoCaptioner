# Subtitles

The fixed sequence is `segment -> correct -> translate -> QC -> one repair ->
final QC -> export`. Independent LLM batches run through the single bounded
executor, while immutable `SubtitleDocument` values are validated and merged
on the main thread. Timing remains transcript-owned and is never an LLM output.

Phase 3 adds ordered Glossary replacement at the translation boundary,
deterministic CPS/line-count/line-length/duration/overlap checks, WebVTT output,
and stable source-then-translation bilingual SRT output. Existing SRT files
are imported by the synchronous `refine` workflow without invoking media or
ASR.
