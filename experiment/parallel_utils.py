"""Deterministic concurrency helpers for independent API-backed clients."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar


T = TypeVar("T")
R = TypeVar("R")

DEFAULT_NUM_WORKERS = 16


def worker_count(value: int) -> int:
    """Validate and normalize a CLI worker count."""
    count = int(value)
    if count < 1:
        raise ValueError("num_workers must be at least 1")
    return count


def parallel_map_ordered(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[R]:
    """Run independent API work concurrently while preserving input order."""
    values = list(items)
    workers = min(worker_count(num_workers), len(values)) if values else 1
    if workers == 1:
        return [function(item) for item in values]
    pool = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="task-client",
    )
    # Keep only one wave admitted to the executor. Submitting the entire
    # dataset up front means a shared provider/billing failure can begin every
    # queued API call before FIRST_EXCEPTION is observed.
    results: List[Any] = [None] * len(values)
    pending: Dict[Any, int] = {}
    next_index = 0

    def submit_until_full() -> None:
        nonlocal next_index
        while next_index < len(values) and len(pending) < workers:
            future = pool.submit(function, values[next_index])
            pending[future] = next_index
            next_index += 1

    submit_until_full()
    try:
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            failure = next(
                (future.exception() for future in done if future.exception()),
                None,
            )
            if failure is not None:
                for future in pending:
                    future.cancel()
                raise failure
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
            submit_until_full()
        pool.shutdown(wait=True)
        return results
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise


def process_map_ordered(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Tuple[Any, ...] = (),
) -> List[R]:
    """Run picklable work in isolated processes while preserving input order.

    A fresh ``spawn`` context is intentional: API clients and dataset file
    descriptors from the coordinator must not leak into workers.  Only one
    wave is submitted at a time so a provider-wide failure cannot enqueue the
    entire dataset before it is observed.
    """
    values = list(items)
    workers = min(worker_count(num_workers), len(values)) if values else 1
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        return [function(item) for item in values]

    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
    results: List[Any] = [None] * len(values)
    pending: Dict[Any, int] = {}
    next_index = 0

    def submit_until_full() -> None:
        nonlocal next_index
        while next_index < len(values) and len(pending) < workers:
            future = pool.submit(function, values[next_index])
            pending[future] = next_index
            next_index += 1

    submit_until_full()
    try:
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            failure = next(
                (future.exception() for future in done if future.exception()),
                None,
            )
            if failure is not None:
                for future in pending:
                    future.cancel()
                raise failure
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
            submit_until_full()
        pool.shutdown(wait=True)
        return results
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise


def add_num_workers_argument(parser) -> None:
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=worker_count,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Maximum number of independent API worker processes to run concurrently "
            f"(default: {DEFAULT_NUM_WORKERS})."
        ),
    )
