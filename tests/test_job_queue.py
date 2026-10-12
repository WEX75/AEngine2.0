"""
tests/test_job_queue.py — core/job_queue.py's JobQueue.join()/drain() semantics
Apex Engine 2.0

Fully offline, no real pillar classes involved — this file proves
JobQueue's own new join() behavior in isolation, since every pillar's
drain() (Brain, ClipWhisperBridge, ClipRenderer) is a one-line wrapper
around it (see each module's drain() docstring). If join() itself is
correct, those wrappers are correct by construction; this file is where
the actual behavior gets proven once instead of three times over.

Three things need proving:
  1. join() on an empty/never-used queue returns immediately (a caller
     that races ahead of any enqueue() must not hang).
  2. join() blocks until an enqueued job has actually FINISHED running —
     not just been picked up — proving it's a true drain, not stop()'s
     cancel-on-pending-work behavior.
  3. join() still unblocks when a job raises — JobQueue._worker()'s
     `finally: self._queue.task_done()` must fire on failure too, or a
     single bad job would hang every future join() forever.

Run:
    python tests/test_job_queue.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.job_queue import JobQueue

# ── 1. join() on an empty queue returns immediately ───────────────────────────


def test_join_empty_queue_returns_immediately() -> None:
    async def run() -> None:
        queue = JobQueue(max_size=5)
        await queue.start()
        try:
            await asyncio.wait_for(queue.join(), timeout=1.0)
        finally:
            await queue.stop()

    asyncio.run(run())
    print("  [PASS] join() on an empty queue returns immediately\n")


# ── 2. join() blocks until the enqueued job has actually finished ────────────


def test_join_waits_for_job_to_finish_not_just_start() -> None:
    async def run() -> None:
        queue = JobQueue(max_size=5)
        await queue.start()

        started = asyncio.Event()
        finished = asyncio.Event()

        async def slow_job() -> None:
            started.set()
            await asyncio.sleep(0.3)
            finished.set()

        try:
            await queue.enqueue(slow_job)
            await started.wait()
            assert not finished.is_set(), "test is broken — job finished before join() was even called"

            await asyncio.wait_for(queue.join(), timeout=2.0)
            assert finished.is_set(), "join() returned before the job actually finished"
        finally:
            await queue.stop()

    asyncio.run(run())
    print("  [PASS] join() blocks until the running job actually finishes\n")


# ── 3. join() waits for ALL enqueued jobs, not just the first ────────────────


def test_join_waits_for_multiple_jobs() -> None:
    async def run() -> None:
        queue = JobQueue(max_size=5)
        await queue.start()

        completed: list[int] = []

        async def job(n: int) -> None:
            await asyncio.sleep(0.05 * n)
            completed.append(n)

        try:
            for n in (3, 1, 2):
                await queue.enqueue(job, n)

            await asyncio.wait_for(queue.join(), timeout=2.0)
            assert sorted(completed) == [1, 2, 3], f"not all jobs completed: {completed}"
        finally:
            await queue.stop()

    asyncio.run(run())
    print("  [PASS] join() waits for every enqueued job, not just the first\n")


# ── 4. a job that raises must not hang join() forever ────────────────────────


def test_join_unblocks_even_when_job_raises() -> None:
    async def run() -> None:
        queue = JobQueue(max_size=5)
        await queue.start()

        async def failing_job() -> None:
            raise RuntimeError("deliberate failure — job_queue must still call task_done()")

        try:
            await queue.enqueue(failing_job)
            # If task_done() weren't called in a finally, this would hang
            # until the wait_for timeout, and the test would fail loudly.
            await asyncio.wait_for(queue.join(), timeout=2.0)
        finally:
            await queue.stop()

    asyncio.run(run())
    print("  [PASS] join() unblocks even when the job raised\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX — core/job_queue.py JobQueue.join() drain test")
    print("=" * 60 + "\n")
    test_join_empty_queue_returns_immediately()
    test_join_waits_for_job_to_finish_not_just_start()
    test_join_waits_for_multiple_jobs()
    test_join_unblocks_even_when_job_raises()
    print("  All JobQueue.join() tests passed.")
    print("=" * 60 + "\n")
