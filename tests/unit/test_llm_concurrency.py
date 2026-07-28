import threading
import time

import pytest

from captioner.llm.client import ThreadLocalClient
from captioner.llm.concurrency import ParallelLlmExecutor


def test_parallel_results_are_sorted_by_input_index() -> None:
    executor = ParallelLlmExecutor(max_workers=8)

    def operation(value: int) -> int:
        time.sleep((7 - value) * 0.002)
        return value * 10

    results = executor.map(tuple(range(8)), operation, max_workers=8)

    assert [result.index for result in results] == list(range(8))
    assert [result.value for result in results] == [value * 10 for value in range(8)]


def test_failed_batch_does_not_discard_other_results() -> None:
    def operation(value: int) -> int:
        if value == 2:
            raise RuntimeError("one batch failed")
        return value

    results = ParallelLlmExecutor(max_workers=4).map(
        (0, 1, 2, 3), operation, max_workers=4
    )

    assert [result.value for result in results] == [0, 1, None, 3]
    assert results[2].error is not None


def test_parallelism_100_is_valid_and_101_is_rejected() -> None:
    results = ParallelLlmExecutor(max_workers=100).map(
        tuple(range(100)), lambda value: value, max_workers=100
    )
    assert [result.value for result in results] == list(range(100))

    with pytest.raises(ValueError, match="between 1 and 100"):
        ParallelLlmExecutor(max_workers=101)
    with pytest.raises(ValueError, match="between 1 and 100"):
        ParallelLlmExecutor().map((1,), lambda value: value, max_workers=101)


def test_thread_local_client_is_reused_only_inside_each_thread() -> None:
    created: list[int] = []
    created_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def factory() -> object:
        thread_id = threading.get_ident()
        with created_lock:
            created.append(thread_id)
        return object()

    client = ThreadLocalClient(factory)
    identities: list[tuple[int, int]] = []

    def worker() -> None:
        barrier.wait()
        first = id(client.get())
        second = id(client.get())
        with created_lock:
            identities.append((first, second))

    threads = (threading.Thread(target=worker), threading.Thread(target=worker))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 2
    assert len({thread_id for thread_id in created}) == 2
    assert all(first == second for first, second in identities)
