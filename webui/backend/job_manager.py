"""
webui/backend/job_manager.py

In-memory job lifecycle for the local web control panel: accepts a job,
returns a job_id immediately, and runs the actual pipeline call
(run_ingest_vod / run_render_manual_window, via job_runner.py) as a
background asyncio task — never inline in a request handler. Structured
progress events fan out to however many WebSocket clients are watching a
given job (a client that connects mid-job first replays everything
buffered so far, then gets live events).

No persistence — jobs live only as long as this process does, per the v1
control-panel spec.

Execution serialization
------------------------
CLAUDE.md's thermal/resource rules are explicit that this hardware
throttles under sustained full CPU+GPU load, and a single ingest job
already saturates CPU for minutes (large-v3 Whisper transcription alone is
~154-185s/clip). `JobManager` accepts job-creation requests immediately —
job_id is always returned right away — but only lets ONE job's actual
pipeline call run at a time, via `_pipeline_lock`. Later jobs sit at
status="queued" until the lock frees, the same back-pressure shape as a
`core.job_queue.JobQueue`, just unbounded (this is a single-operator local
tool, not a burst-traffic service, so there's no need for a max_size).

This also bounds the blast radius of a real, pre-existing gap in
apex.py's one-shot design: `run_ingest_vod()` constructs a fresh
StreamIngestHandler/ClipOrchestrator/Brain/etc. every call and subscribes
them onto `core.event_bus.get_bus()`'s process-wide singleton bus, but
never unsubscribes them (`JobQueue.stop()` cancels a worker task; it does
not undo `EventBus.subscribe_fn()` registrations). apex.py never had to
care because the CLI is one process per invocation. A long-lived server
calling these functions repeatedly would otherwise accumulate stale
handlers from every prior job. Serializing execution here doesn't fix
that (out of scope — this task wraps the CLI entry points, it does not
change pipeline logic) but does guarantee only one job's handlers are
ever the *live, currently-enqueuing* ones — a finished job's stale
handlers sit on an already-stopped, non-consuming JobQueue and go
nowhere, rather than actively duplicating work against a concurrently
running job.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

JobKind = Literal["ingest", "manual_window"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]

# (event_type, stage, message, data) -> None. event_type is "status" | "log" | "milestone".
EmitFn = Callable[[str, str, str, "dict[str, Any] | None"], Awaitable[None]]
# Given the job and its emit callback, do the real work and return the
# produced output file path(s). Raise on failure — including treating a
# non-zero exit code from the wrapped apex.py function as a real failure,
# rather than trusting that function's own "success" log line.
RunnerFn = Callable[["Job", EmitFn], Awaitable["list[str]"]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobEvent:
    type: str      # "status" | "log" | "milestone"
    stage: str     # short slug: "queued" | "running" | "succeeded" | "failed" | an EventBus event name | "log"
    message: str
    data: "dict[str, Any] | None" = None
    ts: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "stage": self.stage, "message": self.message, "data": self.data, "ts": self.ts}


@dataclass
class Job:
    job_id: str
    kind: JobKind
    params: "dict[str, Any]"
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now)
    output_paths: "list[str]" = field(default_factory=list)
    error: "str | None" = None
    events: "list[JobEvent]" = field(default_factory=list)
    subscribers: "list[asyncio.Queue]" = field(default_factory=list, repr=False)

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "output_paths": self.output_paths,
            "error": self.error,
        }
        if include_events:
            d["events"] = [e.to_dict() for e in self.events]
        return d


class JobManager:
    def __init__(self, ingest_runner: RunnerFn, manual_window_runner: RunnerFn) -> None:
        self._jobs: "dict[str, Job]" = {}
        self._ingest_runner = ingest_runner
        self._manual_window_runner = manual_window_runner
        self._pipeline_lock = asyncio.Lock()

    def create_job(self, kind: JobKind, params: "dict[str, Any]") -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, params=params)
        self._jobs[job.job_id] = job
        asyncio.create_task(self._run(job), name=f"job-{job.job_id}")
        return job

    def get(self, job_id: str) -> "Job | None":
        return self._jobs.get(job_id)

    def list_jobs(self) -> "list[Job]":
        return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def subscribe(self, job_id: str) -> "asyncio.Queue | None":
        """
        Returns a fresh queue that receives every event from now on, after
        first being fed everything already buffered on the job (so a
        client connecting mid-job doesn't miss the start). `None` puts a
        terminal sentinel onto the queue once the job reaches a final
        status, so the WebSocket handler knows when to stop and close.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        q: "asyncio.Queue" = asyncio.Queue()
        for event in job.events:
            q.put_nowait(event)
        if job.status in ("succeeded", "failed"):
            q.put_nowait(None)
        else:
            job.subscribers.append(q)
        return q

    def unsubscribe(self, job_id: str, q: "asyncio.Queue") -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            try:
                job.subscribers.remove(q)
            except ValueError:
                pass

    async def _emit(self, job: Job, type_: str, stage: str, message: str, data: "dict[str, Any] | None" = None) -> None:
        event = JobEvent(type=type_, stage=stage, message=message, data=data)
        job.events.append(event)
        for q in list(job.subscribers):
            q.put_nowait(event)

    async def _run(self, job: Job) -> None:
        await self._emit(job, "status", "queued", "Job accepted — waiting for the pipeline to be free.")
        async with self._pipeline_lock:
            job.status = "running"
            await self._emit(job, "status", "running", "Pipeline started.")

            async def emit(type_: str, stage: str, message: str, data: "dict[str, Any] | None" = None) -> None:
                await self._emit(job, type_, stage, message, data)

            runner = self._ingest_runner if job.kind == "ingest" else self._manual_window_runner
            try:
                output_paths = await runner(job, emit)
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                await self._emit(job, "status", "failed", f"Job failed: {exc}")
            else:
                job.status = "succeeded"
                job.output_paths = output_paths
                await self._emit(job, "status", "succeeded", "Job complete.", {"output_paths": output_paths})

        for q in list(job.subscribers):
            q.put_nowait(None)
        job.subscribers.clear()
