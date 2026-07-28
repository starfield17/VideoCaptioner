# Workflow

The workflow is synchronous and file-serial. It creates one temporary
workspace per run, starts one provider ASR worker session, applies the fixed
subtitle stages for `run` or writes Transcript JSON for `transcribe`, and
removes successful workspaces unless explicitly retained.
