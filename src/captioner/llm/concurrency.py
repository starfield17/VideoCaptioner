"""The single bounded executor used for all LLM batch parallelism."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchResult[RequestT, ResultT]:
    """An independent result collected and ordered by the main thread."""

    index: int
    request: RequestT
    value: ResultT | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.value is not None


class ParallelLlmExecutor:
    """Run independent LLM batches with a hard 1–100 worker limit."""

    MAX_WORKERS = 100

    def __init__(self, max_workers: int = 1) -> None:
        self._validate_workers(max_workers)
        self._max_workers = max_workers

    def map[RequestT, ResultT](
        self,
        requests: Sequence[RequestT],
        operation: Callable[[RequestT], ResultT],
        max_workers: int | None = None,
    ) -> tuple[BatchResult[RequestT, ResultT], ...]:
        """Execute requests and return stable input order, including failures."""

        configured_workers = self._max_workers if max_workers is None else max_workers
        self._validate_workers(configured_workers)
        ordered_requests = tuple(requests)
        if not ordered_requests:
            return ()
        actual_workers = min(
            configured_workers,
            len(ordered_requests),
            self.MAX_WORKERS,
        )
        if actual_workers == 1:
            return tuple(
                self._run_one(index, request, operation)
                for index, request in enumerate(ordered_requests)
            )

        results: list[BatchResult[RequestT, ResultT]] = []
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(self._run_one, index, request, operation): index
                for index, request in enumerate(ordered_requests)
            }
            for future in as_completed(futures):
                results.append(future.result())
        return tuple(sorted(results, key=lambda result: result.index))

    @classmethod
    def _validate_workers(cls, max_workers: int) -> None:
        if isinstance(max_workers, bool) or not 1 <= max_workers <= cls.MAX_WORKERS:
            raise ValueError("LLM parallelism must be between 1 and 100")

    @staticmethod
    def _run_one[RequestT, ResultT](
        index: int,
        request: RequestT,
        operation: Callable[[RequestT], ResultT],
    ) -> BatchResult[RequestT, ResultT]:
        try:
            return BatchResult(
                index=index,
                request=request,
                value=operation(request),
            )
        except Exception as exc:
            return BatchResult(index=index, request=request, error=exc)


__all__ = ["BatchResult", "ParallelLlmExecutor"]
