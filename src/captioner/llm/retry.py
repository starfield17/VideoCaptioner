"""Bounded, per-batch retry policy for transient LLM failures."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry one operation in place with exponential backoff and jitter."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)
    random_source: Callable[[], float] = field(default=random.random, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds cannot be negative")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds cannot be negative")

    def run(
        self,
        operation: Callable[[], ResultT],
        should_retry: Callable[[Exception], bool],
    ) -> ResultT:
        """Run one batch operation; never schedules a second copy of it."""

        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                if not should_retry(exc) or attempt == self.max_attempts:
                    raise
                exponent = 2 ** (attempt - 1)
                delay = min(
                    self.max_delay_seconds,
                    self.base_delay_seconds * exponent,
                )
                jitter = 0.5 + self.random_source()
                self.sleeper(delay * jitter)
        raise AssertionError("retry loop did not return or raise")


__all__ = ["RetryPolicy"]
