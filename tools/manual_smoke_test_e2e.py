"""
tools/manual_smoke_test_e2e.py — Apex Engine 2.0 real end-to-end pipeline smoke test

Manual script, NOT a pytest test — deliberately named/placed outside
tests/ and outside pytest's default collection globs (test_*.py /
*_test.py), same reasoning as tools/manual_smoke_test_brain.py.

NAMING NOTE: the task that requested this script asked for
tools/e2e_smoke_test.py. That name ends in "_test.py", which IS one of
pytest's two default collection globs — verified empirically (a file
literally named that way gets collected; "manual_smoke_test_e2e.py" does
not). That's the opposite of the explicit "NOT pytest-collected"
requirement, so this file uses the same manual_smoke_test_*.py convention
already established in this repo instead.

Runs the REAL, unmocked pipeline end-to-end on real local files, for the
first time in this repo's history — nothing here is a fake or a mock:

    real chat log (JSON-Lines) + a real video file
        --> real ffmpeg full-audio extraction + real per-second dBFS
            computation (see _extract_audio_dbfs — see note below)
        --> real StreamMomentDetector.detect(), via the real
            StreamIngestHandler / STREAM_VOD_INGESTED -> MOMENT_DETECTED
        --> real seek-only ffmpeg window extraction (subprocess, not mocked)
        --> real faster-whisper transcription — large-v3/int8 ACTUALLY
            loaded, not injected
        --> real Ollama llama3:8b segment selection — live HTTP call
        --> real CLIP_SCRIPT_READY payload, printed with per-stage timing

Its whole purpose is catching what mocked tests structurally cannot: an
ffmpeg command that's never actually been executed, a Whisper model
that's never actually been loaded, a prompt template that's never
actually met a real transcript. If it finds bugs, this script reports
them — it does not fix pipeline code; that's a separate, deliberate task.

Audio derivation now lives in sentinel/audio_extractor.py
------------------------------------------------------------
This script used to carry its own copy of the ffmpeg + RMS/dBFS logic
(flagged at the time as a gap nothing upstream filled). That's now
sentinel/audio_extractor.extract_audio_dbfs — a real, importable,
Sentinel-owned module (see its own docstring for placement reasoning).
This script just calls it; no duplicated logic.

Requirements
------------
  ffmpeg on PATH
  Ollama running with llama3:8b pulled: ollama serve / ollama pull llama3:8b
  faster-whisper's large-v3 weights (downloaded automatically on first
  use if not already cached — ~1.5GB)
  A real (or hand-written) chat log JSON-Lines file — one message per
  line: {"timestamp": <float, seconds from video start>, "text": <str>}

Usage
-----
  python tools/manual_smoke_test_e2e.py --video path/to/video.mp4 --chat path/to/chat.jsonl
  python tools/manual_smoke_test_e2e.py --video ... --chat ... --all-candidates
  python tools/manual_smoke_test_e2e.py --video ... --chat ... --vod-id my_vod --target-min 15 --target-max 60
  python tools/manual_smoke_test_e2e.py --video ... --chat ... --render

--render continues the proven pipeline through a real cutter/clip_renderer.py
render (real ffmpeg crop/stack/caption burn-in, real encoder probe) for
each produced CLIP_SCRIPT_READY, printing the final CLIP_RENDER_READY
output path and per-stage timing. Omitting --render leaves this script's
existing behavior — stop after printing CLIP_SCRIPT_READY — completely
unchanged; render is additive, not a replacement default.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import (
    CLIP_RENDER_READY,
    CLIP_SCRIPT_READY,
    MOMENT_DETECTED,
    STREAM_VOD_INGESTED,
    ClipRenderReadyPayload,
    ClipScriptReadyPayload,
    EventBus,
    MomentDetectedPayload,
    StreamVodIngestedPayload,
)
from brain.brain import Brain
from brain.clip_whisper_bridge import ClipWhisperBridge
from cutter.clip_renderer import ClipRenderer
from sentinel.audio_extractor import extract_audio_dbfs
from sentinel.stream_ingest import StreamIngestHandler
from sentinel.stream_moment_detector import StreamMomentDetector

_HEARTBEAT_SEC = 15.0
_CLIP_TIMEOUT_SEC = 900.0   # 15 min/candidate ceiling — generous above the ~3-4 min baseline in CLAUDE.md


# ── Small output helpers ──────────────────────────────────────────────────────


def _banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


class _Stage:
    """Prints a start line, then an elapsed/failed end line. Does not swallow exceptions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._t0 = 0.0

    def __enter__(self) -> "_Stage":
        print(f"\n[STAGE] {self.name} — starting...")
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.monotonic() - self._t0
        if exc_type is None:
            print(f"[STAGE] {self.name} — done in {elapsed:.1f}s")
        else:
            print(f"[STAGE] {self.name} — FAILED after {elapsed:.1f}s: {exc_type.__name__}: {exc}")
        return False  # never suppress — let real failures propagate with full context


# ── Stage 1: real audio extraction + dBFS computation (sentinel/audio_extractor.py) ─


def _extract_audio_dbfs(video_path: Path, bucket_sec: float = 1.0) -> tuple[list[float], float]:
    """Thin wrapper: calls the real sentinel.audio_extractor + prints a summary line."""
    dbfs, duration_sec = extract_audio_dbfs(video_path, bucket_sec=bucket_sec)
    print(
        f"    duration={duration_sec:.1f}s  buckets={len(dbfs)}  "
        f"dBFS min/mean/max={min(dbfs):.1f}/{sum(dbfs)/len(dbfs):.1f}/{max(dbfs):.1f}"
    )
    return dbfs, duration_sec


# ── Stage 2: real detection, via the real STREAM_VOD_INGESTED wiring ─────────


async def _run_detection(
    bus: EventBus,
    video_path: Path,
    chat_path: Path,
    vod_id: str,
    streamer_id: str,
    dbfs: list[float],
    duration_sec: float,
) -> list[MomentDetectedPayload]:
    fd, audio_track_json = tempfile.mkstemp(suffix=".json", prefix="apex_smoke_audiotrack_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"bucket_sec": 1.0, "dbfs": dbfs}, f)

        received: list[MomentDetectedPayload] = []

        async def _spy(payload: MomentDetectedPayload) -> None:
            received.append(payload)

        bus.subscribe_fn(MOMENT_DETECTED, _spy)

        handler = StreamIngestHandler(detector=StreamMomentDetector())
        handler.register(bus)

        payload = StreamVodIngestedPayload(
            vod_id=vod_id,
            streamer_id=streamer_id,
            source_path=str(video_path),
            duration_sec=duration_sec,
            chat_log_path=str(chat_path),
            audio_track_path=audio_track_json,
        )
        await bus.publish(STREAM_VOD_INGESTED, payload)
        return received
    finally:
        try:
            os.remove(audio_track_json)
        except OSError:
            pass


# ── Stage 3: real extraction + real Whisper + real Ollama, for one moment ────


async def _run_clip_pipeline(
    bus: EventBus,
    video_path: Path,
    moment: MomentDetectedPayload,
    target_duration_range: tuple[float, float],
) -> ClipScriptReadyPayload | None:
    received: list[ClipScriptReadyPayload] = []

    async def _spy(payload: ClipScriptReadyPayload) -> None:
        received.append(payload)

    bus.subscribe_fn(CLIP_SCRIPT_READY, _spy)

    brain = Brain(bus=bus)   # real ScriptGenerator, real ClipScriptGenerator, real OllamaClient
    bridge = ClipWhisperBridge(brain=brain)   # real ffmpeg extractor, real faster-whisper — no fakes

    await brain.start()      # real Ollama health check happens here
    await bridge.start()
    try:
        await bridge.submit_moment(moment, source_path=video_path, target_duration_range=target_duration_range)

        waited = 0.0
        while waited < _CLIP_TIMEOUT_SEC:
            if received:
                return received[0]
            await asyncio.sleep(1.0)
            waited += 1.0
            if waited % _HEARTBEAT_SEC == 0:
                print(
                    f"    ...still working ({waited:.0f}s elapsed, timeout at "
                    f"{_CLIP_TIMEOUT_SEC:.0f}s) — extraction/Whisper/Ollama can "
                    f"legitimately take minutes, see CLAUDE.md's hardware table"
                )
        print(f"    [TIMEOUT] no CLIP_SCRIPT_READY after {_CLIP_TIMEOUT_SEC:.0f}s for vod_id={moment.vod_id}")
        return None
    finally:
        await bridge.stop()
        await brain.stop()


def _print_clip_result(result: ClipScriptReadyPayload) -> None:
    _banner(f"CLIP_SCRIPT_READY — vod_id={result.vod_id}")
    print(f"  segments selected : [{result.start_segment_id}, {result.end_segment_id}]")
    print(f"  hook_reason       : {result.hook_reason}")
    print(f"  caption_segments  : {len(result.caption_segments)}")
    for c in result.caption_segments:
        print(f"    [{c['id']:>3}] {c['start']:>7.1f}s - {c['end']:>7.1f}s  {c['text']}")


# ── Stage 4 (--render only): real ffmpeg crop/stack/caption render ───────────


async def _run_render_pipeline(
    bus: EventBus,
    video_path: Path,
    clip: ClipScriptReadyPayload,
    streamer_id: str,
) -> ClipRenderReadyPayload | None:
    received: list[ClipRenderReadyPayload] = []

    async def _spy(payload: ClipRenderReadyPayload) -> None:
        received.append(payload)

    bus.subscribe_fn(CLIP_RENDER_READY, _spy)

    renderer = ClipRenderer(bus=bus)   # real encoder probe, real ffmpeg — no fakes
    await renderer.start()             # h264_amf vs libx264 probe happens here, logged
    try:
        await renderer.submit_render(clip, source_path=video_path, streamer_id=streamer_id)

        waited = 0.0
        while waited < _CLIP_TIMEOUT_SEC:
            if received:
                return received[0]
            await asyncio.sleep(1.0)
            waited += 1.0
            if waited % _HEARTBEAT_SEC == 0:
                print(f"    ...still rendering ({waited:.0f}s elapsed, timeout at {_CLIP_TIMEOUT_SEC:.0f}s)")
        print(f"    [TIMEOUT] no CLIP_RENDER_READY after {_CLIP_TIMEOUT_SEC:.0f}s for vod_id={clip.vod_id}")
        return None
    finally:
        await renderer.stop()


def _print_render_result(result: ClipRenderReadyPayload) -> None:
    _banner(f"CLIP_RENDER_READY — vod_id={result.vod_id}")
    print(f"  output_path    : {result.output_path}")
    print(f"  duration_sec   : {result.duration_sec:.1f}s")
    print(f"  encoder_used   : {result.encoder_used}")
    print(f"  render_seconds : {result.render_seconds:.1f}s")


# ── Entry point ───────────────────────────────────────────────────────────────


async def _amain(args: argparse.Namespace) -> int:
    video_path: Path = args.video
    chat_path: Path = args.chat

    if not video_path.is_file():
        print(f"[FAILED] --video does not exist or is not a file: {video_path}")
        return 1
    if not chat_path.is_file():
        print(f"[FAILED] --chat does not exist or is not a file: {chat_path}")
        return 1

    bus = EventBus()
    t_start = time.monotonic()

    _banner("APEX ENGINE 2.0 — REAL END-TO-END SMOKE TEST (no mocks)")
    print(f"  video : {video_path}")
    print(f"  chat  : {chat_path}")
    print(f"  vod_id: {args.vod_id}")

    with _Stage("1. Real ffmpeg full-audio extraction + dBFS computation") as stage:
        try:
            dbfs, duration_sec = _extract_audio_dbfs(video_path)
        except Exception:
            traceback.print_exc()
            return 1

    with _Stage("2. Real detection (STREAM_VOD_INGESTED -> StreamMomentDetector -> MOMENT_DETECTED)"):
        try:
            candidates = await _run_detection(
                bus, video_path, chat_path, args.vod_id, args.streamer_id, dbfs, duration_sec
            )
        except Exception:
            traceback.print_exc()
            return 1

    print(f"\n  {len(candidates)} MOMENT_DETECTED candidate(s) found:")
    for c in candidates:
        print(
            f"    vod_id={c.vod_id}  [{c.window_start:.1f}s -> {c.window_end:.1f}s]  "
            f"composite_score={c.composite_score:.2f}  tokens={c.top_spam_tokens}"
        )

    if not candidates:
        print("\n  Nothing detected — nothing to run through the clip pipeline. Exiting.")
        return 0

    to_process = candidates if args.all_candidates else [max(candidates, key=lambda c: c.composite_score)]
    if not args.all_candidates and len(candidates) > 1:
        print(
            f"\n  Processing only the top-scoring candidate by default "
            f"(composite_score={to_process[0].composite_score:.2f}). Pass --all-candidates to process all {len(candidates)}."
        )

    results: list[ClipScriptReadyPayload] = []
    for i, moment in enumerate(to_process, start=1):
        with _Stage(
            f"3.{i} Real extraction + Whisper + Ollama for "
            f"[{moment.window_start:.1f}s -> {moment.window_end:.1f}s]"
        ):
            try:
                result = await _run_clip_pipeline(
                    bus, video_path, moment, (args.target_min, args.target_max)
                )
            except Exception:
                traceback.print_exc()
                continue
        if result is not None:
            results.append(result)
            _print_clip_result(result)

    render_results: list[ClipRenderReadyPayload] = []
    if args.render:
        for i, clip in enumerate(results, start=1):
            with _Stage(f"4.{i} Real render (ffmpeg crop/stack + caption burn-in) for vod_id={clip.vod_id}"):
                try:
                    render_result = await _run_render_pipeline(bus, video_path, clip, args.streamer_id)
                except Exception:
                    traceback.print_exc()
                    continue
            if render_result is not None:
                render_results.append(render_result)
                _print_render_result(render_result)

    total = time.monotonic() - t_start
    if args.render:
        _banner(
            f"DONE — {len(results)}/{len(to_process)} clip(s) selected, "
            f"{len(render_results)}/{len(results)} rendered, in {total:.1f}s total"
        )
        return 0 if render_results else 1
    _banner(f"DONE — {len(results)}/{len(to_process)} clip(s) produced in {total:.1f}s total")
    return 0 if results else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apex Engine 2.0 — real, unmocked end-to-end pipeline smoke test"
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to a real video file")
    parser.add_argument("--chat", required=True, type=Path, help="Path to a chat log JSON-Lines file")
    parser.add_argument("--vod-id", default="smoke_test_vod")
    parser.add_argument("--streamer-id", default="smoke_test_streamer")
    parser.add_argument("--all-candidates", action="store_true", help="Process every detected candidate, not just the top-scoring one")
    parser.add_argument("--target-min", type=float, default=12.0, help="target_duration_range minimum seconds")
    parser.add_argument("--target-max", type=float, default=75.0, help="target_duration_range maximum seconds")
    parser.add_argument(
        "--render", action="store_true",
        help="Continue through a real cutter/clip_renderer.py render for each CLIP_SCRIPT_READY "
             "produced (real ffmpeg crop/stack/caption burn-in, real encoder probe). "
             "Without this flag, behavior is unchanged from before --render existed.",
    )
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
