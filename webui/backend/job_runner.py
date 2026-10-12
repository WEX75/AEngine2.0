"""
webui/backend/job_runner.py

The two RunnerFn implementations JobManager actually executes. Both wrap
apex.py's one-shot CLI entry points directly — `apex.run_ingest_vod()` /
`apex.run_render_manual_window()` — and reuse its own path-validation
helpers (`apex._resolve_ingest_paths()` / `apex._resolve_manual_video_path()`)
rather than re-checking file existence a second way. Nothing in apex.py or
any pillar module is modified; this file only observes.

Progress, two channels, combined
---------------------------------
1. EventBus milestones — `bus.subscribe_fn()` spies on
   `core.event_bus.get_bus()` for STREAM_VOD_INGESTED / MOMENT_DETECTED /
   CLIP_SCRIPT_READY / CLIP_RENDER_READY, filtered to this job's vod_id,
   forwarding each payload as a structured "milestone" event. This is the
   exact spy technique tools/manual_smoke_test_e2e.py already uses
   (`bus.subscribe_fn(EVENT, _spy)` collecting into a list) to observe
   pipeline progress without touching it — adapted here to push onto
   JobManager's per-job event log/queue instead of a Python list.
2. Log capture — apex.py already narrates each stage via
   `log.info("[Apex] ...")` (audio derived, STREAM_VOD_INGESTED
   published, pipeline drained, etc.) — the exact same narration
   `tools/manual_smoke_test_e2e.py`'s `_Stage` context manager prints to a
   terminal. Since JobManager serializes execution to one job at a time
   (job_manager.py), it's safe to attach one temporary logging.Handler to
   the root logger for the duration of this job's run and forward every
   record from a project logger as a "log" event, rather than
   re-implementing "what stage are we in" as a second, parallel
   heuristic. Detached again in a `finally`, and filtered to this
   project's own loggers (apex/sentinel/brain/cutter/orchestrator/core) so
   uvicorn's request logs and library internals (faster-whisper, aiohttp,
   etc.) don't spam the panel.

--render-manual-window has no MOMENT_DETECTED/CLIP_SCRIPT_READY (that path
deliberately skips Sentinel/Brain — see apex.py's own header comment above
`_resolve_manual_video_path`) and no STREAM_VOD_INGESTED either (that's an
--ingest-vod-only event) — only CLIP_RENDER_READY fires. Log capture is
where most of its progress narration comes from.

Exit-code handling
-------------------
Both apex.py functions return 0/1 rather than raising on a drain timeout,
and apex.py's own JobQueue swallows worker exceptions internally and still
marks jobs done (see CLAUDE.md / the code review that flagged this) — so
"drain finished without hanging" is not proof of success. This layer
treats a non-zero return as a real failure (raises RuntimeError), which is
the one signal from these functions that's actually trustworthy. A
successful drain that produced zero rendered clips (e.g. nothing crossed
the detection threshold, or intake was rejected for insufficient signal)
is NOT treated as a failure here — that's a legitimate outcome, visible to
the operator via the forwarded log lines and an empty output list, not an
error this wrapper invents.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import apex
from core.event_bus import (
    CLIP_RENDER_READY,
    CLIP_SCRIPT_READY,
    MOMENT_DETECTED,
    STREAM_VOD_INGESTED,
    get_bus,
)

if TYPE_CHECKING:
    from webui.backend.job_manager import EmitFn, Job

_MILESTONE_EVENTS = (STREAM_VOD_INGESTED, MOMENT_DETECTED, CLIP_SCRIPT_READY, CLIP_RENDER_READY)
_RELEVANT_LOGGER_PREFIXES = ("apex", "sentinel", "brain", "cutter", "orchestrator", "core")


def _payload_to_dict(payload: Any) -> "dict[str, Any]":
    d = dataclasses.asdict(payload)
    ts = d.get("timestamp")
    if hasattr(ts, "isoformat"):
        d["timestamp"] = ts.isoformat()
    return d


def _milestone_message(event_name: str, payload: Any) -> str:
    if event_name == STREAM_VOD_INGESTED:
        return f"VOD ingested — duration={payload.duration_sec:.1f}s"
    if event_name == MOMENT_DETECTED:
        return (
            f"Moment detected [{payload.window_start:.1f}s → {payload.window_end:.1f}s] "
            f"composite_score={payload.composite_score:.2f}"
        )
    if event_name == CLIP_SCRIPT_READY:
        return f"Clip script ready — segments [{payload.start_segment_id}, {payload.end_segment_id}]: {payload.hook_reason}"
    if event_name == CLIP_RENDER_READY:
        return f"Clip rendered → {payload.output_path} ({payload.duration_sec:.1f}s, {payload.encoder_used})"
    return event_name


def _make_milestone_handler(emit: "EmitFn", vod_id: str, event_name: str, output_paths: "list[str]"):
    async def _handler(payload: Any) -> None:
        if getattr(payload, "vod_id", None) != vod_id:
            return  # another job's event on the shared singleton bus — not ours
        if event_name == CLIP_RENDER_READY and getattr(payload, "output_path", ""):
            output_paths.append(payload.output_path)
        await emit("milestone", event_name, _milestone_message(event_name, payload), _payload_to_dict(payload))

    return _handler


class _LogForwarder(logging.Handler):
    """Forwards project log records as "log" JobEvents while attached."""

    def __init__(self, emit: "EmitFn", loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(level=logging.INFO)
        self._emit = emit
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.split(".")[0] not in _RELEVANT_LOGGER_PREFIXES:
            return
        message = record.getMessage()
        emit_fn = self._emit

        def _schedule() -> None:
            self._loop.create_task(emit_fn("log", "log", message))

        # emit() can run off the event loop (e.g. a log call made from
        # inside asyncio.to_thread'd Whisper/ffmpeg work) — call_soon_threadsafe
        # is the correct way to get back onto the loop from another thread,
        # and is equally correct when already called from the loop thread.
        self._loop.call_soon_threadsafe(_schedule)


async def run_ingest_job(job: "Job", emit: "EmitFn") -> "list[str]":
    p = job.params
    try:
        video_path, chat_log_path, subtitle_track_path = apex._resolve_ingest_paths(
            p["video_path"], p.get("chat_log_path"), p.get("subtitle_track_path"),
        )
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc

    vod_id = p.get("vod_id") or video_path.stem
    drain_timeout = p.get("drain_timeout") or apex._DEFAULT_DRAIN_TIMEOUT_SEC

    bus = get_bus()
    output_paths: "list[str]" = []
    handlers = {name: _make_milestone_handler(emit, vod_id, name, output_paths) for name in _MILESTONE_EVENTS}
    for name, handler in handlers.items():
        bus.subscribe_fn(name, handler)

    log_handler = _LogForwarder(emit, asyncio.get_running_loop())
    logging.getLogger().addHandler(log_handler)
    try:
        exit_code = await apex.run_ingest_vod(
            video_path=video_path,
            streamer_id=p["streamer_id"],
            chat_log_path=chat_log_path,
            vod_id=vod_id,
            drain_timeout=drain_timeout,
            subtitle_track_path=subtitle_track_path,
        )
    finally:
        logging.getLogger().removeHandler(log_handler)
        for name, handler in handlers.items():
            bus.unsubscribe(name, handler)

    if exit_code != 0:
        raise RuntimeError(
            f"run_ingest_vod exited with code {exit_code} (vod_id={vod_id}) — see the log events above for "
            "which stage failed or timed out."
        )
    return output_paths


async def run_manual_window_job(job: "Job", emit: "EmitFn") -> "list[str]":
    p = job.params
    try:
        video_path = apex._resolve_manual_video_path(p["video_path"])
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc

    vod_id = p.get("vod_id") or video_path.stem
    drain_timeout = p.get("drain_timeout") or apex._DEFAULT_DRAIN_TIMEOUT_SEC

    bus = get_bus()
    output_paths: "list[str]" = []
    handler = _make_milestone_handler(emit, vod_id, CLIP_RENDER_READY, output_paths)
    bus.subscribe_fn(CLIP_RENDER_READY, handler)

    log_handler = _LogForwarder(emit, asyncio.get_running_loop())
    logging.getLogger().addHandler(log_handler)
    try:
        exit_code = await apex.run_render_manual_window(
            video_path=video_path,
            streamer_id=p["streamer_id"],
            start=p["start"],
            end=p["end"],
            vod_id=vod_id,
            drain_timeout=drain_timeout,
        )
    finally:
        logging.getLogger().removeHandler(log_handler)
        bus.unsubscribe(CLIP_RENDER_READY, handler)

    if exit_code != 0:
        raise RuntimeError(f"run_render_manual_window exited with code {exit_code} (vod_id={vod_id}).")
    return output_paths
