"""Stable identifier helpers."""


def make_cue_id(index: int) -> str:
    """Return a deterministic document-local cue identifier."""

    if index < 1:
        raise ValueError("cue indexes start at one")
    return f"cue{index:06d}"
