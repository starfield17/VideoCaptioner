# CLI

The CLI exposes `doctor`, `run`, `transcribe`, and `refine`. `refine` reads an
existing SRT and runs the configured correction/translation/QC/export stages
without media or ASR. The adapter parses a small set of options, calls the
public Workflow API, writes machine-readable results to stdout, and keeps
diagnostics on stderr.
