"""
core/job_queue.py

Bounded asyncio job queue with a single worker.
Prevents the AMD Ryzen 7 from being slammed by concurrent renders.

No shared/global singleton — queues are keyed by resource shape, not
pillar identity (see CLAUDE.md's "JobQueue ownership" section). Each
pillar constructs and owns its own instance, sized for its own job shape,
and starts/stops it inside its own start()/stop() lifecycle.

Usage:
    queue = JobQueue(max_size=5)
    await queue.start()
    await queue.enqueue(my_coroutine_fn, arg1, arg2)
    ...
    await queue.stop()
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from core.logger import get_logger

log = get_logger(__name__)


class JobQueue:
    """
    A single-consumer job queue. Jobs are coroutine callables.
    Set max_size to limit how many jobs can queue up before blocking.
    """

    def __init__(self, max_size: int = 10) -> None:
        self._queue: asyncio.Queue[tuple[Callable, tuple, dict]] = asyncio.Queue(
            maxsize=max_size
        )
        self._worker_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background worker. Call once from the main event loop."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker(), name="JobQueueWorker")
        log.info("[JobQueue] Worker started")

    async def stop(self) -> None:
        """
        Stop the worker. This CANCELS the worker task — it does not wait
        for jobs still sitting in the queue to run. If jobs may still be
        queued (or in flight) and you need them to actually finish first,
        `await queue.join()` before calling `stop()`. Calling `stop()`
        alone on a non-empty queue silently abandons whatever hasn't been
        picked up yet.
        """
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        log.info("[JobQueue] Worker stopped")

    async def enqueue(
        self,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Add a coroutine job to the queue.
        Blocks if the queue is full (back-pressure).
        """
        log.debug(f"[JobQueue] Enqueueing job: {fn.__qualname__}")
        await self._queue.put((fn, args, kwargs))

    async def join(self) -> None:
        """
        Block until every job that has been `enqueue()`d so far has been
        picked up AND finished executing (success or failure — the worker's
        `finally: self._queue.task_done()` fires either way).

        This is the actual drain mechanism, distinct from `stop()`: `join()`
        waits, `stop()` cancels. A caller that wants "wait for the pipeline
        to finish, then shut it down" must call `join()` first and `stop()`
        second — never the other way round, and never `stop()` alone if
        anything might still be queued.

        Built on asyncio.Queue's own join()/task_done() protocol, which
        `_worker()` was already satisfying correctly (one task_done() per
        get(), in a finally) — this just exposes it publicly instead of
        leaving callers to reach into `self._queue` directly.
        """
        await self._queue.join()

    def qsize(self) -> int:
        return self._queue.qsize()

    async def _worker(self) -> None:
        while self._running:
            try:
                fn, args, kwargs = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue  # No job yet — check _running flag and retry

            try:
                log.info(f"[JobQueue] Executing job: {fn.__qualname__}")
                await fn(*args, **kwargs)
            except Exception as exc:
                log.exception(f"[JobQueue] Job '{fn.__qualname__}' failed: {exc}")
            finally:
                self._queue.task_done()
