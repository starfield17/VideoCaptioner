# AGENTS.md

These repository guidelines reduce common coding mistakes.

## Think before coding

- State assumptions when they matter and ask when an ambiguity changes scope.
- Prefer the simplest implementation that satisfies the requested behavior.
- Define observable success criteria before implementing multi-step changes.

## Surgical changes

- Touch only files required by the current goal.
- Match the existing style and do not refactor unrelated code.
- Remove only imports, variables, or functions made unused by the current change.

## Goal-driven execution

- Turn requested behavior into an automated check where practical.
- Run the relevant checks after implementation and report their results.
- Do not claim completion without evidence from the current worktree.

## Phase boundary

This repository is currently implementing Phase 0 from
`reffer/video-captioner-architecture-implementation-manual-v1.md`. Do not add
real ASR models, real LLM calls, SQLite, asyncio, GUI code, dynamic plugins,
generic DAGs, or concurrency unless a later phase explicitly authorizes them.
