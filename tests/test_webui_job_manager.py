"""
tests/test_webui_job_manager.py — webui/backend/job_manager.py

Fully offline — every job in this file runs a fake RunnerFn (an
async def matching JobManager's (job, emit) -> list[str] contract), never
apex.run_ingest_vod()/run_render_manual_window(), so nothing here touches
ffmpeg, faster-whisper, or Ollama. Same "never let a test depend on a live
[...] call" discipline CLAUDE.md requires of every other pillar, applied
to this one.

No pytest-asyncio in this project (not a dependency, and no other test
file uses it) — async tests here follow the same `asyncio.run(run())`
pattern already established in tests/test_apex_ingest.py.

Covers job creation/status transitions (queued -> running -> succeeded /
failed), event recording + replay to a late-subscribing WebSocket client,
and the single-pipeline-execution serialization job_manager.py's module
docstring documents as a deliberate thermal/back-pressure choice.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webui.backend.job_manager import JobManager

# ── Fakes ──────────────────────────────────────────────────────────────────────


def _make_runner(*, outputs=None, fail=False, started_event: "asyncio.Event | None" = None,
                  release_event: "asyncio.Event | None" = None):
    """A RunnerFn that emits one log line, optionally waits, then succeeds or fails."""

    async def _runner(job, emit) -> "list[str]":
        if started_event is not None:
            started_event.set()
        if release_event is not None:
            await release_event.wait()
        await emit("log", "log", f"fake stage narration for {job.job_id}")
        if fail:
            raise RuntimeError("boom")
        return outputs if outputs is not None else ["/tmp/fake_output.mp4"]

    return _runner


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    waited = 0.0
    while not predicate():
        if waited >= timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.01)
        waited += 0.01


# ── Status transitions ───────────────────────────────────────────────────────


def test_create_job_returns_immediately_with_queued_status() -> None:
    async def run() -> None:
        jm = JobManager(_make_runner(), _make_runner())
        job = jm.create_job("ingest", {"video_path": "irrelevant.mp4"})
        assert job.job_id
        assert job.status in ("queued", "running")  # create_job doesn't block on the background task

    asyncio.run(run())


def test_successful_job_transitions_to_succeeded_with_output_paths() -> None:
    async def run() -> None:
        jm = JobManager(_make_runner(outputs=["/tmp/clip_a.mp4", "/tmp/clip_b.mp4"]), _make_runner())
        job = jm.create_job("ingest", {"video_path": "irrelevant.mp4"})

        await _wait_until(lambda: job.status in ("succeeded", "failed"))

        assert job.status == "succeeded"
        assert job.output_paths == ["/tmp/clip_a.mp4", "/tmp/clip_b.mp4"]
        assert job.error is None

    asyncio.run(run())


def test_failed_job_transitions_to_failed_with_error_message() -> None:
    async def run() -> None:
        jm = JobManager(_make_runner(fail=True), _make_runner())
        job = jm.create_job("ingest", {"video_path": "irrelevant.mp4"})

        await _wait_until(lambda: job.status in ("succeeded", "failed"))

        assert job.status == "failed"
        assert job.output_paths == []
        assert "boom" in job.error

    asyncio.run(run())


def test_manual_window_kind_dispatches_to_manual_window_runner() -> None:
    async def run() -> None:
        ingest_runner = _make_runner(outputs=["ingest.mp4"])
        manual_runner = _make_runner(outputs=["manual.mp4"])
        jm = JobManager(ingest_runner, manual_runner)

        job = jm.create_job("manual_window", {"video_path": "irrelevant.mp4"})
        await _wait_until(lambda: job.status in ("succeeded", "failed"))

        assert job.output_paths == ["manual.mp4"]

    asyncio.run(run())


# ── Event recording + WebSocket-style replay ─────────────────────────────────


def test_events_are_recorded_in_order_status_first_and_last() -> None:
    async def run() -> None:
        jm = JobManager(_make_runner(), _make_runner())
        job = jm.create_job("ingest", {"video_path": "irrelevant.mp4"})
        await _wait_until(lambda: job.status in ("succeeded", "failed"))

        stages = [e.stage for e in job.events]
        assert stages[0] == "queued"
        assert "running" in stages
        assert "log" in stages
        assert stages[-1] == "succeeded"

    asyncio.run(run())


def test_late_subscriber_is_replayed_full_history_then_terminal_sentinel() -> None:
    async def run() -> None:
        jm = JobManager(_make_runner(), _make_runner())
        job = jm.create_job("ingest", {"video_path": "irrelevant.mp4"})
        await _wait_until(lambda: job.status in ("succeeded", "failed"))

        queue = jm.subscribe(job.job_id)
        replayed = []
        while True:
            event = queue.get_nowait()
            if event is None:
                break
            replayed.append(event)

        assert [e.stage for e in replayed] == [e.stage for e in job.events]

    asyncio.run(run())


def test_live_subscriber_receives_events_as_they_happen() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        jm = JobManager(_make_runner(started_event=started, release_event=release), _make_runner())
        job = jm.create_job("ingest", {"video_path": "irrelevant.mp4"})

        await started.wait()
        queue = jm.subscribe(job.job_id)
        release.set()

        events = []
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            if event is None:
                break
            events.append(event)

        assert job.status == "succeeded"
        assert any(e.stage == "log" for e in events)

    asyncio.run(run())


def test_subscribe_unknown_job_returns_none() -> None:
    jm = JobManager(_make_runner(), _make_runner())
    assert jm.subscribe("does-not-exist") is None


# ── Execution serialization (thermal back-pressure) ──────────────────────────


def test_second_job_stays_queued_until_first_jobs_pipeline_releases() -> None:
    async def run() -> None:
        started_1 = asyncio.Event()
        release_1 = asyncio.Event()
        jm = JobManager(_make_runner(started_event=started_1, release_event=release_1), _make_runner())

        job1 = jm.create_job("ingest", {"video_path": "one.mp4"})
        await started_1.wait()  # job1's pipeline body is now running, holding the lock

        job2 = jm.create_job("ingest", {"video_path": "two.mp4"})
        await asyncio.sleep(0.05)  # give job2's background task a chance to run and block on the lock
        assert job2.status == "queued", "job2 must not start its pipeline body while job1 holds the lock"

        release_1.set()
        await _wait_until(lambda: job1.status == "succeeded")
        await _wait_until(lambda: job2.status == "succeeded")

    asyncio.run(run())
