"""
apex.py — Apex Engine 2.0 Entry Point

Boots all three pillars and wires them together via the EventBus.

Usage:
    python apex.py                  # Run full autonomous engine
    python apex.py --sentinel-only  # Run only The Sentinel (for testing)
    python apex.py --dry-run        # Sentinel dry-run, no events, no renders
    python apex.py --ingest-vod <path> --streamer-id <id> [--chat-log <path>] [--subtitle-track <path>]
                                     # One-shot stream-clipping VOD intake
                                     # (see run_ingest_vod()'s docstring for
                                     # which VOD types are actually
                                     # processable today)
    python apex.py --render-manual-window <path> --streamer-id <id> --start <sec> --end <sec>
                                     # One-shot manual render: skip Sentinel
                                     # detection and Brain's LLM segment
                                     # selection entirely and render exactly
                                     # the given window (see
                                     # run_render_manual_window()'s docstring)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from core.config_loader import get_config
from core.event_bus import STREAM_VOD_INGESTED, get_bus, StreamVodIngestedPayload
from core.logger import get_logger
from sentinel.sentinel import Sentinel

log = get_logger(__name__)

_DEFAULT_DRAIN_TIMEOUT_SEC = 1800.0  # 30 min — generous above the smoke tool's own 900s/clip ceiling for multiple candidates


def _stream_clipping_enabled(cfg: dict) -> bool:
    """
    Small, deliberately pure helper (no I/O, no side effects) so the
    stream_clipping.enabled config flag's read-path is unit-testable in
    isolation — see tests/test_clip_orchestrator.py — without needing to
    mock main()'s whole pillar-construction sequence.
    """
    return bool(cfg.get("stream_clipping", {}).get("enabled", False))


async def main(sentinel_only: bool = False, dry_run: bool = False) -> None:
    cfg = get_config()

    if dry_run:
        cfg["sentinel"]["dry_run"] = True

    bus = get_bus()

    # ── Wire up pillars (Brain + Factory — enabled in future phases) ──────────
    # Brain owns its own JobQueue and an async health check, so it uses the
    # richer start()/stop() lifecycle (see brain/brain.py + CLAUDE.md's
    # "JobQueue ownership" section) rather than a bare sync register(bus) —
    # that's the pattern Brain itself already documents and implements.
    brain = None
    clip_bridge = None
    clip_renderer = None
    clip_orchestrator = None
    if not sentinel_only:
        try:
            from brain.brain import Brain  # type: ignore[import]
            brain = Brain()
            await brain.start()
            log.info("[Apex] 🧠 Brain (Pillar 2) registered.")
        except ImportError:
            log.warning(
                "[Apex] Brain module not yet installed — running Sentinel only. "
                "This is expected for Phase 1."
            )

        # ── Stream-clipping mode (behind config flag) ──────────────────────────
        # Reuses the SAME Brain instance constructed above — never a second one.
        # CLAUDE.md's "JobQueue ownership" section is explicit that Brain owns
        # exactly one queue for every LLM-call-shaped job, trend script
        # generation AND clip segment selection both; a second Brain here would
        # silently create a second queue and defeat that design (see
        # orchestrator/clip_orchestrator.py's docstring, "Brain instance reuse").
        if brain is not None and _stream_clipping_enabled(cfg):
            from brain.clip_whisper_bridge import ClipWhisperBridge
            from cutter.clip_renderer import ClipRenderer
            from orchestrator.clip_orchestrator import ClipOrchestrator

            clip_bridge = ClipWhisperBridge(brain=brain)
            clip_renderer = ClipRenderer(bus=bus)
            clip_orchestrator = ClipOrchestrator(bridge=clip_bridge, renderer=clip_renderer, bus=bus)

            await clip_bridge.start()
            await clip_renderer.start()
            await clip_orchestrator.start()
            log.info(
                "[Apex] 🎬 Stream-clipping mode online "
                "(ClipWhisperBridge + ClipRenderer + ClipOrchestrator)."
            )
        elif _stream_clipping_enabled(cfg):
            log.warning(
                "[Apex] stream_clipping.enabled is true but Brain isn't available — "
                "stream-clipping mode needs Brain's queue for segment selection. Skipping."
            )

        try:
            from factory.factory import Factory  # type: ignore[import]
            factory = Factory()
            factory.register(bus)
            log.info("[Apex] 🏭 Factory (Pillar 3) registered.")
        except ImportError:
            log.warning(
                "[Apex] Factory module not yet installed — running Sentinel only. "
                "This is expected for Phase 1."
            )

    # ── Launch The Sentinel ───────────────────────────────────────────────────
    sentinel = Sentinel()
    log.info("[Apex] 👁️  Sentinel (Pillar 1) starting...")

    try:
        if dry_run or sentinel_only:
            # Single-shot for testing
            await sentinel.run_once()
        else:
            # Autonomous loop (runs forever until Ctrl+C)
            await sentinel.run_forever()
    except KeyboardInterrupt:
        log.info("[Apex] 🛑 Shutdown requested by user (Ctrl+C)")
    finally:
        await sentinel.stop()
        # Shutdown order matters: orchestrator first (unsubscribes, so no new
        # moments/clips enter the pipeline), then bridge/renderer (let any
        # in-flight job finish handing off), then Brain LAST — bridge's job
        # body calls brain.submit_clip_script_job(), so Brain needs to still
        # be running for that hand-off to land cleanly. Note this is a
        # best-effort ordering, not a guarantee: core.job_queue.JobQueue.stop()
        # cancels its worker rather than waiting for the queue to fully drain,
        # so a job mid-hand-off at the exact moment of shutdown can still be
        # lost — a pre-existing JobQueue limitation, not something fixed here.
        if clip_orchestrator is not None:
            await clip_orchestrator.stop()
        if clip_bridge is not None:
            await clip_bridge.stop()
        if clip_renderer is not None:
            await clip_renderer.stop()
        if brain is not None:
            await brain.stop()
        log.info("[Apex] 👋 Apex Engine 2.0 shut down cleanly.")


def _resolve_ingest_paths(
    video: str, chat_log: str | None, subtitle_track: str | None = None
) -> tuple[Path, Path | None, Path | None]:
    """
    Pure-ish validation helper (I/O-bound existence checks only, no side
    effects) so the --ingest-vod path-resolution logic is unit-testable
    without constructing the whole pillar stack — same pattern as
    _stream_clipping_enabled() above.

    Raises FileNotFoundError with a clear message on a missing/non-file path.
    """
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"--ingest-vod path does not exist or is not a file: {video_path}")

    chat_log_path: Path | None = None
    if chat_log is not None:
        chat_log_path = Path(chat_log).resolve()
        if not chat_log_path.is_file():
            raise FileNotFoundError(f"--chat-log path does not exist or is not a file: {chat_log_path}")

    subtitle_track_path: Path | None = None
    if subtitle_track is not None:
        subtitle_track_path = Path(subtitle_track).resolve()
        if not subtitle_track_path.is_file():
            raise FileNotFoundError(
                f"--subtitle-track path does not exist or is not a file: {subtitle_track_path}"
            )

    return video_path, chat_log_path, subtitle_track_path


def _build_ingest_detector(has_chat_log: bool, has_subtitle_track: bool = False,
                           config: dict | None = None):
    """
    Pure construction helper, unit-testable in isolation.

    Three cases (see CLAUDE.md Pillar 1's "which VOD types are actually
    processable today" note, and the CRITICAL PRODUCT DECISION in
    sentinel/stream_ingest.py):

      has_chat_log=True                      -> unchanged default weights
                                                 (0.6/0.4 chat/audio, no
                                                 semantic contribution) —
                                                 the common case, untouched
                                                 by subtitle support existing.
      has_chat_log=False, has_subtitle_track=True
                                              -> the settings.yaml "interview"
                                                 profile via
                                                 build_moment_detector_from_config
                                                 (chat_weight=0.0,
                                                 audio_weight=0.3,
                                                 semantic_weight=0.7, plus
                                                 the keyword vocabulary and
                                                 the profile's audio_z_floor).
                                                 Going through the config
                                                 factory is load-bearing:
                                                 an earlier revision built
                                                 StreamMomentDetector(...)
                                                 directly with the same
                                                 weights but NO keyword
                                                 lists, which left the
                                                 semantic signal silently
                                                 inert in the real CLI
                                                 (every cue scored 0.0
                                                 against empty keyword
                                                 sets) — found during
                                                 end-to-end validation.
                                                 `config` is a test-only
                                                 injection point (hermetic
                                                 tests, no dependency on
                                                 the real settings.yaml);
                                                 None means the real config.
      has_chat_log=False, has_subtitle_track=False
                                              -> the original audio-only
                                                 weights (chat_weight=0.0,
                                                 audio_weight=1.0), kept
                                                 here for backward-compat/
                                                 documentation clarity only.
                                                 This combination never
                                                 actually reaches detect():
                                                 StreamIngestHandler.
                                                 on_vod_ingested() raises
                                                 InsufficientSignalError for
                                                 it first (no chat AND no
                                                 subtitles is the known-
                                                 inadequate signal set a
                                                 known from testing — see
                                                 sentinel/stream_ingest.py).
    """
    from sentinel.stream_ingest import build_moment_detector_from_config
    from sentinel.stream_moment_detector import StreamMomentDetector

    if has_chat_log:
        return StreamMomentDetector()
    if has_subtitle_track:
        return build_moment_detector_from_config(
            "interview", has_chat=False, has_subtitles=True, config=config
        )
    return StreamMomentDetector(chat_weight=0.0, audio_weight=1.0)


async def run_ingest_vod(
    video_path:           Path,
    streamer_id:          str,
    chat_log_path:        Path | None,
    vod_id:               str,
    drain_timeout:        float = _DEFAULT_DRAIN_TIMEOUT_SEC,
    subtitle_track_path:  Path | None = None,
) -> int:
    """
    One-shot VOD intake for the stream-clipping pipeline. Derives the audio
    track, publishes a well-formed STREAM_VOD_INGESTED, and blocks until
    every job that event triggers — detection, transcription, segment
    selection, render — has actually FINISHED, not just been accepted onto
    a queue, before returning.

    "Drained" detection — how it actually works
    ---------------------------------------------
    core.job_queue.JobQueue.stop() CANCELS its worker; it does not wait for
    queued/in-flight jobs to finish (see that class's own stop() docstring,
    and CLAUDE.md's JobQueue caveats #1/#2). The real drain primitive is the
    new JobQueue.join() (wraps asyncio.Queue's join()/task_done() protocol),
    exposed here as .drain() on Brain / ClipWhisperBridge / ClipRenderer.

    EventBus.publish() awaits every subscriber via asyncio.gather() before
    returning (core/event_bus.py). StreamIngestHandler.on_vod_ingested and
    ClipOrchestrator's handlers are both ordinary subscribers, so by the
    time `await bus.publish(STREAM_VOD_INGESTED, ...)` below returns:
      - detection has already run, and every resulting MOMENT_DETECTED has
        already been published AND handled — including every
        clip_bridge.submit_moment() call the orchestrator made, which is
        itself awaited (i.e. at minimum ENQUEUED onto the bridge's queue,
        though back-pressure could briefly block it if the queue were
        already full).

    What's left is a strict pipeline, one stage at a time — draining out of
    this order can return before the next stage's work has even been
    handed to it:
      1. bridge.drain()   — waits for every enqueued transcription job to
                             finish. Each one, on success, itself calls AND
                             AWAITS brain.submit_clip_script_job() — which
                             only enqueues onto Brain's queue, so by the
                             time step 1 finishes, every clip-script job
                             that will ever exist for this VOD is enqueued.
      2. brain.drain()    — waits for every enqueued clip-script job to
                             finish. ClipScriptGenerator.generate() awaits
                             bus.publish(CLIP_SCRIPT_READY) itself, which
                             synchronously runs the orchestrator's handler,
                             which awaits renderer.submit_render() (another
                             enqueue) — so by the time step 2 finishes,
                             every render job is enqueued.
      3. renderer.drain() — waits for every enqueued render to finish
                             (CLIP_RENDER_READY published, or the failure
                             logged) — the last stage, nothing downstream.

    drain_timeout wraps the whole three-stage sequence: a stuck ffmpeg or
    Ollama call must not hang this process forever. On timeout this
    function logs an error and returns 1 rather than raising — whatever
    finished before the timeout has already been published for real; only
    the wait is abandoned.

    Which VOD types are actually processable today (see CLAUDE.md Pillar 1
    for the full rationale)
    ------------------------------------------------------------------------
      - Chat-having content (--chat-log provided): always fine, default
        0.6/0.4 chat/audio fusion, unaffected by anything below.
      - Chat-less content WITH an existing subtitle/caption file
        (--subtitle-track — e.g. YouTube auto-captions, a client-provided
        transcript): fine, uses audio+semantic fusion
        (_build_ingest_detector(has_chat_log=False, has_subtitle_track=True)).
      - Chat-less content with NEITHER --chat-log NOR --subtitle-track:
        NOT processable — rejected at intake (InsufficientSignalError,
        sentinel/stream_ingest.py) rather than silently falling back to the
        known-inadequate audio-only signal set. A future
        feature (running faster-whisper over the full VOD upfront to
        generate a subtitle track when none exists) was deliberately NOT
        built here — it would require loading/transcribing the full VOD
        before detection even runs, which violates Pillar 1's "CPU only,
        chat+audio ONLY, never the full VOD up front" boundary, costs
        potentially tens of minutes to hours per VOD (per
        brain/clip_whisper_bridge.py's measured ~154-185s to transcribe a
        single ~12-75s candidate window), and would run Whisper a second
        time over overlapping audio (clip_whisper_bridge.py already
        transcribes the detected window for captioning) — a real design
        problem needing its own pass, not a CLI flag.

    Also fixes a real, previously-unwired gap: apex.py's normal main() loop
    constructs ClipOrchestrator (which only populates its vod_id registry
    on STREAM_VOD_INGESTED) but never constructs/registers
    sentinel.stream_ingest.StreamIngestHandler — the thing that actually
    turns STREAM_VOD_INGESTED into MOMENT_DETECTED. Without it, nothing
    downstream of ingestion would ever run. --ingest-vod wires it directly
    since it's this feature's whole reason to exist; main()'s long-running
    stream-clipping mode has the same gap and still needs it wired
    separately — flagged, not fixed here, since main() has no current
    caller that publishes STREAM_VOD_INGESTED yet.
    """
    from brain.brain import Brain
    from brain.clip_whisper_bridge import ClipWhisperBridge
    from cutter.clip_renderer import ClipRenderer
    from orchestrator.clip_orchestrator import ClipOrchestrator
    from sentinel.audio_extractor import extract_audio_dbfs
    from sentinel.stream_ingest import StreamIngestHandler

    bus = get_bus()

    dbfs, duration_sec = await asyncio.to_thread(extract_audio_dbfs, video_path)
    log.info(
        f"[Apex] Audio derived — duration={duration_sec:.1f}s buckets={len(dbfs)} "
        f"dBFS min/mean/max={min(dbfs):.1f}/{sum(dbfs) / len(dbfs):.1f}/{max(dbfs):.1f}"
    )

    audio_fd, audio_track_path = tempfile.mkstemp(suffix=".json", prefix="apex_ingest_audiotrack_")
    os.close(audio_fd)
    with open(audio_track_path, "w", encoding="utf-8") as f:
        json.dump({"bucket_sec": 1.0, "dbfs": dbfs}, f)

    owns_chat_log_file = chat_log_path is None
    has_subtitle_track = subtitle_track_path is not None
    if owns_chat_log_file:
        chat_fd, chat_log_str = tempfile.mkstemp(suffix=".jsonl", prefix="apex_ingest_chatlog_")
        os.close(chat_fd)  # empty file
        if has_subtitle_track:
            log.info("[Apex] No --chat-log provided — running audio+subtitle (semantic) detection.")
        else:
            log.warning(
                "[Apex] No --chat-log and no --subtitle-track provided — this VOD will be REJECTED "
                "at intake (InsufficientSignalError). Audio-only detection for chat-less content is "
                "a known-inadequate signal set (CLAUDE.md Pillar 1) — provide one or the other."
            )
    else:
        chat_log_str = str(chat_log_path)

    brain = Brain(bus=bus)
    clip_bridge = ClipWhisperBridge(brain=brain)
    clip_renderer = ClipRenderer(bus=bus)
    clip_orchestrator = ClipOrchestrator(bridge=clip_bridge, renderer=clip_renderer, bus=bus)
    stream_ingest_handler = StreamIngestHandler(
        detector=_build_ingest_detector(
            has_chat_log=not owns_chat_log_file, has_subtitle_track=has_subtitle_track,
        )
    )

    await brain.start()
    await clip_bridge.start()
    await clip_renderer.start()
    await clip_orchestrator.start()
    stream_ingest_handler.register(bus)
    log.info("[Apex] 🎬 Stream-clipping pipeline online for one-shot VOD intake.")

    try:
        payload = StreamVodIngestedPayload(
            vod_id=vod_id,
            streamer_id=streamer_id,
            source_path=str(video_path),
            duration_sec=duration_sec,
            chat_log_path=chat_log_str,
            audio_track_path=audio_track_path,
            subtitle_track_path=str(subtitle_track_path) if subtitle_track_path else "",
        )
        log.info(f"[Apex] Publishing STREAM_VOD_INGESTED — vod_id={vod_id} streamer_id={streamer_id}")
        await bus.publish(STREAM_VOD_INGESTED, payload)

        log.info(
            f"[Apex] Detection complete, downstream jobs enqueued — draining pipeline "
            f"(timeout={drain_timeout:.0f}s)..."
        )
        try:
            await asyncio.wait_for(_drain_ingest_pipeline(clip_bridge, brain, clip_renderer), timeout=drain_timeout)
        except asyncio.TimeoutError:
            log.error(
                f"[Apex] Pipeline drain timed out after {drain_timeout:.0f}s for vod_id={vod_id} — "
                "a stage is still running. Whatever completed before the timeout was published for "
                "real; only the wait was abandoned."
            )
            return 1

        log.info(f"[Apex] ✅ Pipeline drained — vod_id={vod_id} fully processed.")
        return 0
    finally:
        # Same shutdown order as main()'s stream-clipping block: orchestrator
        # first (stop accepting new work), then bridge/renderer, Brain last
        # (bridge's hand-off target must still be running during its own
        # stop() — though by this point everything is already drained, so
        # stop() here is just releasing resources, not racing in-flight work).
        await clip_orchestrator.stop()
        await clip_bridge.stop()
        await clip_renderer.stop()
        await brain.stop()
        try:
            os.remove(audio_track_path)
        except OSError:
            pass
        if owns_chat_log_file:
            try:
                os.remove(chat_log_str)
            except OSError:
                pass


async def _drain_ingest_pipeline(clip_bridge, brain, clip_renderer) -> None:
    """
    The strict, order-dependent three-stage drain sequence — see
    run_ingest_vod()'s docstring for why the order is load-bearing.
    Factored out so asyncio.wait_for() can time out the whole sequence in
    one place.
    """
    await clip_bridge.drain()
    await brain.drain()
    await clip_renderer.drain()


# ── --render-manual-window: skip Sentinel + Brain, render an operator-given window ──
#
# This path is entirely separate from --ingest-vod above — it does not call
# run_ingest_vod(), StreamIngestHandler, StreamMomentDetector, or Brain, and
# nothing above this point is modified by it. It exists for already-curated
# source content (e.g. a pre-edited clip library) where an operator has
# already identified the good moment themselves: there is nothing for
# Sentinel's chat/audio/semantic fusion to detect and nothing for Brain's
# LLM to select — both of those stages assume real detection/selection
# uncertainty that simply isn't present here.
#
# What IS still real, not skipped: Whisper transcription of the given
# window (for genuine word-timestamped captions — CLAUDE.md Pillar 2 is
# unconditional that timestamps come from real Whisper output, "Code maps
# segment_id -> real Whisper timestamp," not something this path gets to
# opt out of just because it isn't going through Brain) and the full real
# ClipRenderer (crop_mode/caption_styles/visual_style/slow_zoom all resolve
# per-streamer exactly as they do for the automatic pipeline, since this
# calls the same ClipRenderer class, not a simplified stand-in).


def _resolve_manual_video_path(video: str) -> Path:
    """
    Path validation for --render-manual-window. Deliberately a SEPARATE
    function from _resolve_ingest_paths() above rather than reusing it:
    that function's FileNotFoundError text hardcodes "--ingest-vod"
    wording, and reusing it here would surface a misleading error message
    under a different flag. Same existence-check behavior as that
    function's video-path branch, just with the correct flag name — not a
    second copy of --ingest-vod's chat-log/subtitle-track handling, which
    this path has no equivalent of.
    """
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"--render-manual-window path does not exist or is not a file: {video_path}")
    return video_path


def _resolve_manual_window(
    streamer_id: str,
    start:       float,
    end:         float,
    config:      dict | None = None,
) -> tuple[float, float]:
    """
    Pure, side-effect-free (besides logging) validation/clamp of an
    operator-specified [start, end) window against the streamer's resolved
    target_duration_range — the same per-streamer override
    orchestrator.clip_orchestrator.ClipOrchestrator applies for the
    automatic detection path (settings.yaml stream_clipping.
    target_duration_ranges), reusing its exact validation function rather
    than a second copy of the same numeric checks.

    There is no LLM auto-repair on this path (ClipScriptGenerator's
    pad/trim logic lives entirely downstream of an LLM SELECTION this path
    doesn't have — there's no wider transcript to pad into, no LLM pick to
    trim), so the range is enforced directly, here, with different rules
    for each direction:
      * end <= start                        -> ValueError (nonsensical window).
      * duration < target_duration_range[0] -> ValueError. Auto-repair's
        pad-short-window rule extends into ADJACENT transcript segments it
        already has; this path has no such margin and no transcript yet
        (transcription happens AFTER this call) — safely inventing extra
        content the operator didn't select isn't this function's call to
        make, so it fails loudly instead.
      * duration > target_duration_range[1] -> HARD CLAMP: end is trimmed
        back to start + max_sec, logged as a WARNING. Mirrors
        ClipScriptGenerator's own "trim overlong window" auto-repair rule
        (brain/clip_script_generator.py), just applied directly since
        there's no LLM/auto-repair step here to do it.

    `config` is a test-only injection point (hermetic tests, no dependency
    on the real settings.yaml); None means the real config.
    """
    from orchestrator.clip_orchestrator import (
        _DEFAULT_TARGET_DURATION_RANGE,
        _validate_target_duration_range,
    )

    if end <= start:
        raise ValueError(
            f"--render-manual-window: invalid window — end ({end}) must be greater than start ({start})"
        )

    cfg = config if config is not None else get_config()
    raw_ranges = cfg.get("stream_clipping", {}).get("target_duration_ranges", {}) or {}
    range_for_streamer = raw_ranges.get(streamer_id)
    target_range = (
        _validate_target_duration_range(streamer_id, range_for_streamer)
        if range_for_streamer is not None
        else _DEFAULT_TARGET_DURATION_RANGE
    )

    min_sec, max_sec = target_range
    duration = end - start
    if duration < min_sec:
        raise ValueError(
            f"--render-manual-window: window duration {duration:.1f}s is shorter than streamer "
            f"{streamer_id!r}'s target_duration_range minimum ({min_sec:.1f}s) — provide a longer "
            "--start/--end window; there is no transcript here to safely pad into."
        )
    if duration > max_sec:
        clamped_end = start + max_sec
        log.warning(
            f"[Apex] --render-manual-window: window duration {duration:.1f}s exceeds streamer "
            f"{streamer_id!r}'s target_duration_range maximum ({max_sec:.1f}s) — hard-clamping "
            f"end from {end:.1f}s to {clamped_end:.1f}s."
        )
        end = clamped_end

    return start, end


async def run_render_manual_window(
    video_path:    Path,
    streamer_id:   str,
    start:         float,
    end:           float,
    vod_id:        str,
    drain_timeout: float = _DEFAULT_DRAIN_TIMEOUT_SEC,
    extractor  = None,
    transcriber = None,
    renderer   = None,
    config:     dict | None = None,
) -> int:
    """
    One-shot manual render for an already-curated source window. Skips
    Sentinel detection (StreamMomentDetector/StreamIngestHandler) and
    Brain's LLM segment selection (ClipScriptGenerator) entirely — the
    operator's --start/--end IS the selection. Still runs real Whisper
    transcription (for real caption timestamps, never fabricated — see
    this section's header comment) and the real ClipRenderer.

    Pipeline
    --------
        _resolve_manual_window()   — validate/clamp against target_duration_range
        --> seek-extract [start, end) audio (real ffmpeg, or injected `extractor`)
        --> transcribe (real Whisper, or injected `transcriber`)
        --> build ClipScriptReadyPayload directly from the transcript
            (start/end_segment_id spans every recovered segment — there is
            no id range to pick, unlike ClipScriptGenerator's selection)
        --> renderer.submit_render() --> renderer.drain()

    No transcribe margin (unlike clip_whisper_bridge.py's
    _DEFAULT_TRANSCRIBE_MARGIN_SEC): that margin exists so
    ClipScriptGenerator's auto-repair has real adjacent segments to pad
    into when the LLM's pick or the detected window falls short of
    target_duration_range's minimum. Neither applies here — the window is
    already validated against target_duration_range by
    _resolve_manual_window() before transcription ever runs, so there is
    nothing left to pad into more transcript for. Extracting exactly what
    the operator asked for, no more, is the correct behavior for a
    pre-curated window.

    Zero transcribed segments (e.g. a genuinely silent window) does not
    fail the render — it renders with one empty caption_segment spanning
    the window, i.e. no visible caption text, which is the truthful
    output for silent audio, not an error.

    `extractor`/`transcriber`/`renderer` are injectable (same DI pattern as
    ClipWhisperBridge/ClipRenderer) so tests never need a real ffmpeg
    binary, a loaded Whisper model, or a real ClipRenderer queue. None of
    the three defaults construct or touch Brain, Sentinel, or the
    ClipOrchestrator registry — this function's only dependency on the
    rest of the stream-clipping pipeline is ClipRenderer itself and the
    settings.yaml target_duration_ranges lookup (via
    _resolve_manual_window(), which reuses
    orchestrator.clip_orchestrator's validation function, not the
    orchestrator class). `config` is the same test-only injection point
    _resolve_manual_window() takes (hermetic tests, no dependency on the
    real settings.yaml); None means the real config.
    """
    from brain.clip_script_generator import TranscriptSegment, WordTiming, caption_segments_from_transcript
    from brain.clip_whisper_bridge import _WhisperTranscriber, _default_extract_window_audio
    from core.event_bus import ClipScriptReadyPayload, get_bus
    from cutter.clip_renderer import ClipRenderer

    start, end = _resolve_manual_window(streamer_id, start, end, config=config)

    extractor = extractor or _default_extract_window_audio
    transcriber = transcriber or _WhisperTranscriber()
    renderer = renderer or ClipRenderer(bus=get_bus())

    audio_fd, audio_tmp_path = tempfile.mkstemp(suffix=".wav", prefix="apex_manual_window_")
    os.close(audio_fd)
    try:
        log.info(
            f"[Apex] --render-manual-window: extracting+transcribing [{start:.1f}s, {end:.1f}s) "
            f"from {video_path} (streamer_id={streamer_id})"
        )
        await asyncio.to_thread(extractor, str(video_path), start, end, audio_tmp_path)
        raw_segments = await asyncio.to_thread(transcriber, audio_tmp_path)
    finally:
        try:
            os.remove(audio_tmp_path)
        except OSError:
            pass

    if raw_segments:
        # Whisper transcribes the extracted clip, so its own timestamps
        # start at 0 — shift every segment (and word) by +start, same
        # VOD-absolute convention clip_whisper_bridge.py uses.
        segments = [
            TranscriptSegment(
                id=s.id, start=s.start + start, end=s.end + start, text=s.text,
                words=tuple(
                    WordTiming(word=w.word, start=w.start + start, end=w.end + start)
                    for w in s.words
                ),
            )
            for s in raw_segments
        ]
    else:
        log.warning(
            f"[Apex] --render-manual-window: no speech detected in [{start:.1f}s, {end:.1f}s) — "
            "rendering with no caption text (not an error)."
        )
        segments = [TranscriptSegment(id=0, start=start, end=end, text="")]

    payload = ClipScriptReadyPayload(
        vod_id=vod_id,
        start_segment_id=segments[0].id,
        end_segment_id=segments[-1].id,
        caption_segments=caption_segments_from_transcript(segments),
        hook_reason="Manual window (operator-selected — no Brain/LLM selection)",
    )

    await renderer.start()
    try:
        await renderer.submit_render(payload, source_path=str(video_path), streamer_id=streamer_id)
        try:
            await asyncio.wait_for(renderer.drain(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            log.error(
                f"[Apex] --render-manual-window: render timed out after {drain_timeout:.0f}s "
                f"for vod_id={vod_id}."
            )
            return 1
        log.info(f"[Apex] ✅ --render-manual-window complete — vod_id={vod_id}.")
        return 0
    finally:
        await renderer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apex Engine 2.0")
    parser.add_argument(
        "--sentinel-only",
        action="store_true",
        help="Run only The Sentinel (skip Brain and Factory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan trends and print results without rendering any videos.",
    )
    parser.add_argument(
        "--ingest-vod",
        metavar="PATH",
        help="One-shot stream-clipping VOD intake: derive audio, publish STREAM_VOD_INGESTED, "
             "and block until the pipeline drains. Requires --streamer-id.",
    )
    parser.add_argument("--streamer-id", help="Required with --ingest-vod.")
    parser.add_argument(
        "--chat-log",
        metavar="PATH",
        help="Optional chat log (JSON-Lines) for --ingest-vod. Omit for chat-less content "
             "(e.g. podcasts/webinars/interviews with no chat channel) — see --subtitle-track.",
    )
    parser.add_argument(
        "--subtitle-track",
        metavar="PATH",
        help="Optional .vtt/.srt subtitle/caption file for --ingest-vod (e.g. YouTube auto-captions, "
             "a client-provided transcript). REQUIRED when --chat-log is omitted: chat-less content "
             "with no subtitle track either is rejected at intake (InsufficientSignalError) rather than "
             "silently falling back to audio-only detection, which CLAUDE.md Pillar 1 found inadequate "
             "for this content shape. Generating a subtitle track automatically (full-VOD transcription) "
             "is not supported — provide an existing one.",
    )
    parser.add_argument("--vod-id", help="Optional vod_id override for --ingest-vod/--render-manual-window (default: video filename stem).")
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=_DEFAULT_DRAIN_TIMEOUT_SEC,
        help=f"Seconds to wait for --ingest-vod's/--render-manual-window's pipeline to drain before "
             f"giving up (default: {_DEFAULT_DRAIN_TIMEOUT_SEC:.0f}).",
    )
    parser.add_argument(
        "--render-manual-window",
        metavar="PATH",
        help="One-shot manual render: skip Sentinel detection and Brain's LLM segment selection "
             "entirely, and render exactly the given --start/--end window from an already-curated "
             "source (e.g. a pre-edited clip library). Requires --streamer-id, --start, --end. "
             "Entirely separate from --ingest-vod — does not touch Sentinel, Brain, or the "
             "ClipOrchestrator registry.",
    )
    parser.add_argument("--start", type=float, help="Window start, in seconds. Required with --render-manual-window.")
    parser.add_argument("--end", type=float, help="Window end, in seconds. Required with --render-manual-window.")
    args = parser.parse_args()

    if args.ingest_vod:
        if not args.streamer_id:
            parser.error("--ingest-vod requires --streamer-id")
        try:
            video_path, chat_log_path, subtitle_track_path = _resolve_ingest_paths(
                args.ingest_vod, args.chat_log, args.subtitle_track,
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))  # exits the process — never returns

        vod_id = args.vod_id or video_path.stem
        try:
            exit_code = asyncio.run(
                run_ingest_vod(
                    video_path=video_path,
                    streamer_id=args.streamer_id,
                    chat_log_path=chat_log_path,
                    vod_id=vod_id,
                    drain_timeout=args.drain_timeout,
                    subtitle_track_path=subtitle_track_path,
                )
            )
        except KeyboardInterrupt:
            sys.exit(130)
        sys.exit(exit_code)

    if args.render_manual_window:
        if not args.streamer_id:
            parser.error("--render-manual-window requires --streamer-id")
        if args.start is None or args.end is None:
            parser.error("--render-manual-window requires --start and --end")
        try:
            video_path = _resolve_manual_video_path(args.render_manual_window)
        except FileNotFoundError as exc:
            parser.error(str(exc))  # exits the process — never returns

        vod_id = args.vod_id or video_path.stem
        try:
            exit_code = asyncio.run(
                run_render_manual_window(
                    video_path=video_path,
                    streamer_id=args.streamer_id,
                    start=args.start,
                    end=args.end,
                    vod_id=vod_id,
                    drain_timeout=args.drain_timeout,
                )
            )
        except KeyboardInterrupt:
            sys.exit(130)
        except ValueError as exc:
            parser.error(str(exc))  # exits the process — never returns
        sys.exit(exit_code)

    try:
        asyncio.run(main(sentinel_only=args.sentinel_only, dry_run=args.dry_run))
    except KeyboardInterrupt:
        sys.exit(0)
