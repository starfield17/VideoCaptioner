"""Conversions at the subtitle-format boundary."""


def format_srt_timestamp(milliseconds: int) -> str:
    """Format non-negative integer milliseconds as an SRT timestamp."""

    if milliseconds < 0:
        raise ValueError("SRT timestamps cannot be negative")

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
