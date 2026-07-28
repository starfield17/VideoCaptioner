"""Output-only subtitle line formatting."""

import re


def format_text(text: str, max_line_chars: int, max_lines: int) -> tuple[str, ...]:
    """Wrap text without dropping or changing its non-whitespace content."""

    normalized = " ".join(text.split())
    if not normalized:
        return ("",)
    lines: list[str] = []
    remaining = normalized
    while remaining and len(lines) < max_lines - 1:
        split_at = _split_position(remaining, max_line_chars)
        if split_at >= len(remaining):
            break
        lines.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    lines.append(remaining)
    return tuple(lines)


def _split_position(text: str, limit: int) -> int:
    if len(text) <= limit:
        return len(text)
    candidates = [
        match.end()
        for match in re.finditer(r"[\s,.;:!?，。；：！？、]", text[: limit + 1])
    ]
    return candidates[-1] if candidates else limit


__all__ = ["format_text"]
