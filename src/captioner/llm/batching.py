"""Deterministic immutable batch splitting."""

from collections.abc import Sequence


def split_batches[ItemT](
    items: Sequence[ItemT], batch_size: int
) -> tuple[tuple[ItemT, ...], ...]:
    """Split ordered input into non-overlapping, owner-stable batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return tuple(
        tuple(items[start : start + batch_size])
        for start in range(0, len(items), batch_size)
    )


__all__ = ["split_batches"]
