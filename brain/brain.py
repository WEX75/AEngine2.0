"""
brain/brain.py  —  The Brain Orchestrator (Pillar 2)
Apex Engine 2.0

Subscribes to TREND_FOUND events on the shared EventBus, generates a
viral script via ScriptGenerator, and publishes a SCRIPT_READY event.
Also exposes submit_clip_script_job() for the stream-clipping mode's clip
segment selection (ClipScriptGenerator) — see "One shared queue" below.

Concurrency model
-----------------
Script generation is CPU/IO heavy (10–30s on AMD Ryzen).  Multiple
TREND_FOUND events could arrive within a single scan cycle (top-5 trends).
To protect the AMD hardware from RAM thrashing, generation requests are
queued through core.JobQueue — a single-worker async queue that processes
them one at a time while the EventBus stays free to receive new events.

One shared queue for both LLM-call-shaped jobs (CLAUDE.md "JobQueue
ownership")
------------------------------------------------------------------------
Brain owns exactly one JobQueue. Both trend-to-script generation
(ScriptGenerator, via TREND_FOUND) and clip segment selection
(ClipScriptGenerator, via submit_clip_script_job()) enqueue onto it — same
resource shape, one short Ollama call each, no FFmpeg, no sustained
CPU+GPU dual load. Do NOT give clip jobs a second queue here; that would
be fragmentation the resource-shape rule doesn't ask for. (Cutter, when
built, is a genuinely different shape — CPU+GPU+encode across 3 stages —
and gets its own queue, per the same CLAUDE.md section.)

submit_clip_script_job() is a direct method call, not an EventBus
subscription: MOMENT_DETECTED doesn't carry the Whisper segment array or
the source video path (see brain/clip_whisper_bridge.py's docstring for
why), so there's nothing for Brain to usefully subscribe to yet. A caller
that already has `segments` (today: brain.clip_whisper_bridge, used
standalone) calls this once a future orchestrator wires the two together.

Lifecycle
---------
  Brain is instantiated once by apex.py.
  Brain.start() subscribes to the EventBus and starts the job worker.
  Brain.stop() drains the queue gracefully before shutting down.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Sequence

from brain.clip_script_generator import (
    ClipScriptGenerationError,
    ClipScriptGenerator,
    TranscriptSegment,
)
from brain.models import ViralScript
from brain.ollama_client import OllamaClient, OllamaError
from brain.script_generator import ScriptGenerator, ScriptGenerationError
from core.config_loader import get_config
from core.event_bus import (
    CLIP_SCRIPT_READY,
    SCRIPT_READY,
    TREND_FOUND,
    ClipScriptReadyPayload,
    EventBus,
    MomentDetectedPayload,
    ScriptReadyPayload,
    TrendFoundPayload,
    get_bus,
)
from core.job_queue import JobQueue
from core.logger import get_logger

log = get_logger(__name__)

_CLIP_DEFAULT_TARGET_DURATION_RANGE: tuple[float, float] = (12.0, 75.0)


class Brain:
    """
    The Brain — Pillar 2 Orchestrator.

    EventBus flow (trend pipeline):
        [TREND_FOUND payload] → Brain._on_trend_found()
                              → JobQueue
                              → ScriptGenerator.generate()
                              → [SCRIPT_READY payload]

    Direct-call flow (stream-clipping pipeline):
        submit_clip_script_job(moment, segments)
                              → JobQueue (same queue as above)
                              → ClipScriptGenerator.generate()
                              → [CLIP_SCRIPT_READY payload]
                                (ClipScriptGenerator publishes it itself)
    """

    def __init__(
        self,
        generator:      ScriptGenerator     | None = None,
        clip_generator: ClipScriptGenerator | None = None,
        ollama_client:  OllamaClient        | None = None,
        bus:            EventBus            | None = None,
        queue_max_size: int = 5,
    ) -> None:
        self._bus            = bus or get_bus()
        self._generator      = generator or ScriptGenerator()
        self._clip_generator = clip_generator or ClipScriptGenerator(bus=self._bus)
        self._health_client  = ollama_client or OllamaClient()
        self._queue          = JobQueue(max_size=queue_max_size)   # shared by both job kinds
        self._running        = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to EventBus and start the job queue worker."""
        if self._running:
            log.warning("[Brain] Already running — ignoring duplicate start().")
            return

        # Verify Ollama is reachable before subscribing
        await self._check_ollama_health()

        self._bus.subscribe_fn(TREND_FOUND, self._on_trend_found)
        await self._queue.start()
        self._running = True

        log.info("[Brain] Online — subscribed to TREND_FOUND, job queue started.")

    async def stop(self) -> None:
        """
        Unsubscribe and stop the job queue worker. Does NOT wait for
        queued jobs to finish — call `await brain.drain()` first if that
        matters (see `drain()`).
        """
        if not self._running:
            return

        self._bus.unsubscribe(TREND_FOUND, self._on_trend_found)
        await self._queue.stop()
        self._running = False
        log.info("[Brain] Stopped.")

    async def drain(self) -> None:
        """
        Block until every job enqueued so far (TREND_FOUND scripts AND
        submit_clip_script_job() calls — same shared queue) has finished.
        Call this BEFORE stop() when a caller (e.g. apex.py's --ingest-vod
        path) needs to know the pipeline has actually processed everything,
        not just that no new jobs are being accepted.
        """
        await self._queue.join()

    # ── EventBus handler ──────────────────────────────────────────────────────

    async def _on_trend_found(self, payload: TrendFoundPayload) -> None:
        """
        Called by the EventBus when a TREND_FOUND event is published.
        Immediately enqueues the generation job and returns — never blocks
        the EventBus.
        """
        log.info(
            f"[Brain] Received TREND_FOUND — "
            f"trend_id={payload.trend_id} | "
            f"score={payload.viral_score:.3f} | "
            f"topic='{payload.topic[:50]}'"
        )
        await self._queue.enqueue(self._generate_and_emit, payload)

    # ── Clip segment selection (direct call, not a bus subscription) ─────────

    async def submit_clip_script_job(
        self,
        moment:                 MomentDetectedPayload,
        segments:               Sequence[TranscriptSegment],
        target_duration_range:  tuple[float, float] = _CLIP_DEFAULT_TARGET_DURATION_RANGE,
    ) -> None:
        """
        Enqueue a clip segment-selection job onto Brain's JobQueue — the
        same queue TREND_FOUND jobs use (see module docstring: same
        resource shape, one shared queue per CLAUDE.md's JobQueue
        ownership section). Returns once the job is queued, not once it
        completes — CLIP_SCRIPT_READY is published later, from inside the
        queue worker, same fire-and-forget shape as _on_trend_found.
        """
        log.info(
            f"[Brain] Submitting clip script job — vod_id={moment.vod_id} "
            f"window=[{moment.window_start:.1f}s, {moment.window_end:.1f}s] "
            f"({len(segments)} segments)"
        )
        await self._queue.enqueue(
            self._generate_clip_and_emit, moment, segments, target_duration_range
        )

    async def select_clip_script(
        self,
        moment:                 MomentDetectedPayload,
        segments:               Sequence[TranscriptSegment],
        target_duration_range:  tuple[float, float] = _CLIP_DEFAULT_TARGET_DURATION_RANGE,
        publish:                bool = True,
    ) -> ClipScriptReadyPayload:
        """
        Like submit_clip_script_job(), but RETURNS the selection result to
        the caller (and can defer publication via publish=False). The job
        still runs on Brain's own queue — this method just bridges the
        queue's fire-and-forget worker back to an awaitable result via a
        Future, so callers that must INSPECT the selection before letting
        downstream act on it (clip_whisper_bridge's transcript-boundary
        widen-retry) can do so without running the Ollama call outside
        Brain's queue.

        Unlike the fire-and-forget path, generation errors PROPAGATE to
        the awaiting caller (ClipScriptGenerationError / OllamaError) —
        the caller decides how to handle them; the queue worker itself is
        protected either way (the exception is captured into the Future,
        never raised inside the worker).
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ClipScriptReadyPayload] = loop.create_future()

        async def _select_job() -> None:
            try:
                payload = await self._clip_generator.generate(
                    moment, segments, target_duration_range, publish=publish
                )
                future.set_result(payload)
            except Exception as exc:
                future.set_exception(exc)

        await self._queue.enqueue(_select_job)
        return await future

    async def publish_clip_script(self, payload: ClipScriptReadyPayload) -> None:
        """
        Publish a CLIP_SCRIPT_READY payload previously returned by
        select_clip_script(..., publish=False), once the caller has decided
        it's the selection downstream should actually render.
        """
        log.info(
            f"[Brain] >> Emitting CLIP_SCRIPT_READY (deferred) | "
            f"vod_id={payload.vod_id} | segments "
            f"[{payload.start_segment_id}, {payload.end_segment_id}]"
        )
        await self._bus.publish(CLIP_SCRIPT_READY, payload)

    # ── Job worker ────────────────────────────────────────────────────────────

    async def _generate_and_emit(self, payload: TrendFoundPayload) -> None:
        """
        Runs inside the JobQueue's single worker.
        Generates the script, then emits SCRIPT_READY or logs the failure.
        """
        try:
            script: ViralScript = await self._generator.generate(payload)
            await self._emit_script_ready(script)

        except ScriptGenerationError as exc:
            log.error(
                f"[Brain] Script generation failed for "
                f"trend_id={payload.trend_id}: {exc}"
            )
        except OllamaError as exc:
            log.error(
                f"[Brain] Ollama error for trend_id={payload.trend_id}: {exc}"
            )
        except Exception as exc:
            log.exception(
                f"[Brain] Unexpected error for trend_id={payload.trend_id}: {exc}"
            )

    async def _emit_script_ready(self, script: ViralScript) -> None:
        """Convert ViralScript → ScriptReadyPayload and publish."""
        emit_payload = ScriptReadyPayload(
            trend_id    = script.trend_id,
            topic       = script.topic,
            hook        = script.hook,
            body        = script.body,
            cta         = script.cta,
            full_script = script.full_script,
        )

        log.info(
            f"[Brain] >> Emitting SCRIPT_READY | "
            f"trend_id={script.trend_id} | "
            f"{script.summary()}"
        )
        await self._bus.publish(SCRIPT_READY, emit_payload)

    async def _generate_clip_and_emit(
        self,
        moment:                 MomentDetectedPayload,
        segments:               Sequence[TranscriptSegment],
        target_duration_range:  tuple[float, float],
    ) -> None:
        """
        Runs inside the JobQueue's single worker — same worker
        _generate_and_emit runs on, just a different queued job type.
        Unlike ScriptGenerator, ClipScriptGenerator.generate() already
        publishes CLIP_SCRIPT_READY itself on success, so this wrapper
        only needs to catch and log failures (same error-isolation
        pattern as _generate_and_emit — one bad job must never take down
        the worker or block jobs queued behind it).
        """
        try:
            await self._clip_generator.generate(moment, segments, target_duration_range)

        except ClipScriptGenerationError as exc:
            log.error(
                f"[Brain] Clip script generation failed for "
                f"vod_id={moment.vod_id}: {exc}"
            )
        except OllamaError as exc:
            log.error(
                f"[Brain] Ollama error for vod_id={moment.vod_id}: {exc}"
            )
        except Exception as exc:
            log.exception(
                f"[Brain] Unexpected error generating clip script for "
                f"vod_id={moment.vod_id}: {exc}"
            )

    # ── Startup health check ──────────────────────────────────────────────────

    async def _check_ollama_health(self) -> None:
        """
        Checks Ollama connectivity at startup.
        Logs a clear warning (does NOT raise) so the engine can still boot
        and handle the error when the first actual generation request arrives.
        Uses self._health_client (injectable) rather than constructing a
        fresh OllamaClient — so tests can substitute a fake and never touch
        the network (CLAUDE.md testing philosophy).
        """
        cfg    = get_config()
        host   = cfg.get("brain", {}).get("ollama_host", "http://localhost:11434")
        model  = cfg.get("brain", {}).get("model", "llama3:8b")

        is_up = await self._health_client.health_check()
        if is_up:
            log.info(
                f"[Brain] Ollama health check passed — "
                f"host={host}, model={model}"
            )
        else:
            log.warning(
                f"[Brain] Ollama is NOT reachable at {host}. "
                f"Script generation will fail until Ollama is started. "
                f"Run: ollama serve  (then: ollama pull {model})"
            )
