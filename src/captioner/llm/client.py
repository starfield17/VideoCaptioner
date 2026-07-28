"""Thread-local third-party client storage."""

from collections.abc import Callable
from threading import local
from typing import cast


class ThreadLocalClient[ClientT]:
    """Create exactly one SDK client per worker thread and reuse it."""

    def __init__(self, factory: Callable[[], ClientT]) -> None:
        self._factory = factory
        self._local = local()

    def get(self) -> ClientT:
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._factory()
            self._local.client = client
        return cast(ClientT, client)


__all__ = ["ThreadLocalClient"]
