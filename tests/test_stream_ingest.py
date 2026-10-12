"""
tests/test_stream_ingest.py  —  Pillar 1 integration layer offline test
Apex Engine 2.0

Fully-mocked, offline, deterministic — no live stream, no Ollama, no
network (per CLAUDE.md's testing philosophy). Proves the wiring only:
StreamIngestHandler.on_vod_ingested loads fixture chat/audio files written
in sentinel/stream_ingest.py's documented on-disk format, runs the
(already-tested) StreamMomentDetector, and publishes MOMENT_DETECTED events
on a real (in-process, in-memory) EventBus with the correct field values.
Detection correctness itself is NOT re-tested here — that's
tests/test_stream_sentinel.py and tests/test_stream_sentinel_scale.py's job.

Run:
    python tests/test_stream_ingest.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import (
    MOMENT_DETECTED,
    STREAM_VOD_INGESTED,
    EventBus,
    MomentDetectedPayload,
    StreamVodIngestedPayload,
)
from sentinel.stream_ingest import InsufficientSignalError, StreamIngestHandler
from sentinel.stream_moment_detector import ChatMessage, StreamMomentDetector
from tests.test_stream_sentinel import (
    AUDIO_SPIKE_DUR,
    AUDIO_SPIKE_START,
    CHAT_SPIKE_DUR,
    CHAT_SPIKE_START,
    DURATION_SEC,
    make_audio_track,
    make_chat_log,
)

# ── Fixture-file writers (mirror sentinel/stream_ingest.py's documented format) ─


def _write_chat_log(path: Path, messages: list[ChatMessage]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for m in messages:
            fh.write(json.dumps({"timestamp": m.timestamp, "text": m.text}) + "\n")


def _write_audio_track(path: Path, dbfs: list[float], bucket_sec: float = 1.0) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"bucket_sec": bucket_sec, "dbfs": dbfs}, fh)


def _write_subtitle_vtt(path: Path, cues: list[tuple[float, float, str]]) -> None:
    """Writes a minimal, clean .vtt file — cue timing/text messiness itself
    is covered by tests/test_stream_subtitle_loader.py, not re-tested here."""
    def _ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("WEBVTT\n\n")
        for start, end, text in cues:
            fh.write(f"{_ts(start)} --> {_ts(end)}\n{text}\n\n")


def _make_spy() -> tuple[list[MomentDetectedPayload], Callable[[MomentDetectedPayload], Awaitable[None]]]:
    received: list[MomentDetectedPayload] = []

    async def _spy(payload: MomentDetectedPayload) -> None:
        received.append(payload)

    return received, _spy


# ── 1. Positive: STREAM_VOD_INGESTED → exactly one MOMENT_DETECTED ───────────


def test_ingest_publishes_moment_detected_for_spike() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chat = make_chat_log(
            DURATION_SEC, base_rate=5,
            spike_start=CHAT_SPIKE_START, spike_duration=CHAT_SPIKE_DUR,
            spike_rate=30, seed=42,
        )
        audio = make_audio_track(
            DURATION_SEC, base_dbfs=-32.0,
            spike_start=AUDIO_SPIKE_START, spike_duration=AUDIO_SPIKE_DUR,
            spike_dbfs=-12.0, seed=1337,
        )
        chat_path = tmp_path / "chat.jsonl"
        audio_path = tmp_path / "audio.json"
        _write_chat_log(chat_path, chat)
        _write_audio_track(audio_path, audio)

        bus = EventBus()
        handler = StreamIngestHandler()
        handler.register(bus)
        received, spy = _make_spy()
        bus.subscribe_fn(MOMENT_DETECTED, spy)

        payload = StreamVodIngestedPayload(
            vod_id="vod_ingest_001",
            streamer_id="streamer_1",
            source_path="/fake/vods/vod_ingest_001.mp4",
            duration_sec=float(DURATION_SEC),
            chat_log_path=str(chat_path),
            audio_track_path=str(audio_path),
        )

        published = asyncio.run(bus.publish(STREAM_VOD_INGESTED, payload))

        print(f"  MOMENT_DETECTED events received by spy: {len(received)}")
        for m in received:
            print(f"    [{m.window_start:.1f}s → {m.window_end:.1f}s] "
                  f"composite={m.composite_score:.2f} tokens={m.top_spam_tokens}")

        assert len(received) == 1, f"expected exactly 1 MOMENT_DETECTED, got {len(received)}"
        m = received[0]

        assert m.vod_id == "vod_ingest_001"
        assert m.detection_method == "chat_audio_zscore_fusion"
        assert m.window_start <= CHAT_SPIKE_START <= m.window_end
        assert m.composite_score >= 2.5
        assert m.chat_score > 0 and m.audio_score > 0
        assert math.isfinite(m.composite_score)
        hype = {"clip", "omegalul", "w", "lmfaooo", "way", "no", "it", "now"}
        assert hype & set(m.top_spam_tokens)
        print("  [PASS] STREAM_VOD_INGESTED wiring publishes correct MOMENT_DETECTED\n")


# ── 2. Negative: flat baseline VOD → zero MOMENT_DETECTED events ─────────────


def test_ingest_flat_baseline_publishes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        duration = 300
        flat_chat = [ChatMessage(timestamp=float(s) + 0.5, text="hi")
                     for s in range(duration) for _ in range(4)]
        flat_audio = [-30.0] * duration

        chat_path = tmp_path / "chat.jsonl"
        audio_path = tmp_path / "audio.json"
        _write_chat_log(chat_path, flat_chat)
        _write_audio_track(audio_path, flat_audio)

        bus = EventBus()
        handler = StreamIngestHandler()
        handler.register(bus)
        received, spy = _make_spy()
        bus.subscribe_fn(MOMENT_DETECTED, spy)

        payload = StreamVodIngestedPayload(
            vod_id="vod_ingest_flat_001",
            streamer_id="streamer_1",
            source_path="/fake/vods/vod_ingest_flat_001.mp4",
            duration_sec=float(duration),
            chat_log_path=str(chat_path),
            audio_track_path=str(audio_path),
        )

        asyncio.run(bus.publish(STREAM_VOD_INGESTED, payload))

        assert received == [], (
            f"flat baseline VOD published {len(received)} MOMENT_DETECTED "
            f"event(s) — ingestion wiring must not fabricate candidates"
        )
        print("  [PASS] flat-baseline VOD publishes zero MOMENT_DETECTED events\n")


# ── 3. Mixed resolution: audio_track_path's own bucket_sec is honored ────────
# Proves the loader reads the audio file's declared bucket_sec and plumbs it
# through to StreamMomentDetector.detect's audio_bucket_sec parameter — not
# just the wiring's happy path at matching resolution.


def test_ingest_honors_audio_track_bucket_sec() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_bucket_sec = 0.5

        chat = make_chat_log(
            DURATION_SEC, base_rate=5,
            spike_start=CHAT_SPIKE_START, spike_duration=CHAT_SPIKE_DUR,
            spike_rate=30, seed=42,
        )
        audio = make_audio_track(
            DURATION_SEC, base_dbfs=-32.0,
            spike_start=AUDIO_SPIKE_START, spike_duration=AUDIO_SPIKE_DUR,
            spike_dbfs=-12.0, seed=1337, bucket_sec=audio_bucket_sec,
        )
        chat_path = tmp_path / "chat.jsonl"
        audio_path = tmp_path / "audio.json"
        _write_chat_log(chat_path, chat)
        _write_audio_track(audio_path, audio, bucket_sec=audio_bucket_sec)

        assert len(audio) == int(DURATION_SEC / audio_bucket_sec), (
            "fixture didn't actually generate audio at 0.5s resolution — test is broken"
        )

        bus = EventBus()
        handler = StreamIngestHandler()
        handler.register(bus)
        received, spy = _make_spy()
        bus.subscribe_fn(MOMENT_DETECTED, spy)

        payload = StreamVodIngestedPayload(
            vod_id="vod_ingest_mixed_res_001",
            streamer_id="streamer_1",
            source_path="/fake/vods/vod_ingest_mixed_res_001.mp4",
            duration_sec=float(DURATION_SEC),
            chat_log_path=str(chat_path),
            audio_track_path=str(audio_path),
        )

        asyncio.run(bus.publish(STREAM_VOD_INGESTED, payload))

        assert len(received) == 1, (
            f"expected exactly 1 MOMENT_DETECTED with 0.5s-resolution audio, "
            f"got {len(received)} — audio_bucket_sec likely not read from the fixture file"
        )
        m = received[0]
        assert m.window_start <= CHAT_SPIKE_START <= m.window_end
        print("  [PASS] audio_track_path's declared bucket_sec (0.5s) is honored end-to-end\n")


# ── 4. Handler must be registered before it can process events ──────────────


def test_unregistered_handler_raises() -> None:
    handler = StreamIngestHandler()
    payload = StreamVodIngestedPayload(
        vod_id="vod_x", streamer_id="s", source_path="/x.mp4",
        duration_sec=1.0, chat_log_path="/x.jsonl", audio_track_path="/x.json",
    )
    try:
        asyncio.run(handler.on_vod_ingested(payload))
        raise AssertionError("expected RuntimeError for unregistered handler")
    except RuntimeError:
        print("  [PASS] unregistered handler raises RuntimeError instead of silently no-op'ing\n")


# ── 5. subtitle_track_path wired end-to-end: no chat, semantic-only detection ──

def test_ingest_recovers_archetype_b_moment_via_subtitle_track() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        duration = 300
        spike_start = 150.0

        audio = [-38.0] * duration  # flat/weak interview audio, deliberately never enough alone

        cues: list[tuple[float, float, str]] = [
            (float(s), float(s) + 1.0, "so tell me about your day")
            for s in range(0, duration, 3)
            if not (spike_start <= s < spike_start + 6.0)
        ]
        cues += [
            (spike_start + i, spike_start + i + 1.0, "that was an insane clutch headshot")
            for i in range(6)
        ]
        cues.sort(key=lambda c: c[0])

        chat_path = tmp_path / "chat.jsonl"
        audio_path = tmp_path / "audio.json"
        subtitle_path = tmp_path / "captions.vtt"
        _write_chat_log(chat_path, [])  # no chat channel for this content type
        _write_audio_track(audio_path, audio)
        _write_subtitle_vtt(subtitle_path, cues)

        bus = EventBus()
        detector = StreamMomentDetector(
            chat_weight=0.0, audio_weight=0.3, semantic_weight=0.7,
            strong_keywords=["headshot", "clutch"], generic_keywords=["insane"],
        )
        handler = StreamIngestHandler(detector=detector)
        handler.register(bus)
        received, spy = _make_spy()
        bus.subscribe_fn(MOMENT_DETECTED, spy)

        payload = StreamVodIngestedPayload(
            vod_id="vod_archetype_b_ingest_001",
            streamer_id="streamer_interview",
            source_path="/fake/vods/vod_archetype_b_ingest_001.mp4",
            duration_sec=float(duration),
            chat_log_path=str(chat_path),
            audio_track_path=str(audio_path),
            subtitle_track_path=str(subtitle_path),
        )

        asyncio.run(bus.publish(STREAM_VOD_INGESTED, payload))

        matches = [m for m in received if m.window_start <= spike_start <= m.window_end]
        assert len(matches) == 1, (
            f"expected exactly 1 matching MOMENT_DETECTED for the interview moment "
            f"recovered via subtitle_track_path, got {len(matches)} (of {len(received)} total)"
        )
        m = matches[0]
        assert m.chat_score == 0.0
        assert m.semantic_score > 0.0
        print("  [PASS] subtitle_track_path wired end-to-end, recovers a no-chat/weak-audio moment\n")


# ── 6. Reject at intake: no chat AND no subtitle track → InsufficientSignalError ──
# CRITICAL PRODUCT DECISION (CLAUDE.md Pillar 1): Archetype-B-shaped content
# (no chat) with no subtitle_track_path either must be rejected at intake,
# not silently processed on a known-inadequate audio-only signal set (the
# known failure mode this intake policy prevents).

def test_ingest_rejects_when_no_chat_and_no_subtitles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        duration = 60
        chat_path = tmp_path / "chat.jsonl"
        audio_path = tmp_path / "audio.json"
        _write_chat_log(chat_path, [])
        _write_audio_track(audio_path, [-30.0] * duration)

        bus = EventBus()
        handler = StreamIngestHandler()
        handler.register(bus)

        payload = StreamVodIngestedPayload(
            vod_id="vod_insufficient_001",
            streamer_id="streamer_interview",
            source_path="/fake/vods/vod_insufficient_001.mp4",
            duration_sec=float(duration),
            chat_log_path=str(chat_path),
            audio_track_path=str(audio_path),
            subtitle_track_path="",  # not provided
        )

        # Direct call: the handler itself must raise, not silently no-op.
        try:
            asyncio.run(handler.on_vod_ingested(payload))
            raise AssertionError("expected InsufficientSignalError for no-chat/no-subtitle VOD")
        except InsufficientSignalError:
            pass

        # Via the bus: EventBus.publish() catches+logs handler exceptions
        # (core/event_bus.py) rather than propagating them — so the
        # end-to-end observable behavior is "zero MOMENT_DETECTED events",
        # not a crash.
        received, spy = _make_spy()
        bus.subscribe_fn(MOMENT_DETECTED, spy)
        asyncio.run(bus.publish(STREAM_VOD_INGESTED, payload))
        assert received == [], (
            "no-chat/no-subtitle VOD must publish zero MOMENT_DETECTED events via the bus"
        )
        print("  [PASS] no-chat + no-subtitle VOD rejected at intake (InsufficientSignalError)\n")


def test_ingest_with_chat_but_no_subtitles_is_not_rejected() -> None:
    """Guards against an overly-broad reject rule: ordinary chat-having
    content (the common case, Archetype A) must be entirely unaffected by
    the new subtitle-required policy."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chat = make_chat_log(
            DURATION_SEC, base_rate=5,
            spike_start=CHAT_SPIKE_START, spike_duration=CHAT_SPIKE_DUR,
            spike_rate=30, seed=42,
        )
        audio = make_audio_track(
            DURATION_SEC, base_dbfs=-32.0,
            spike_start=AUDIO_SPIKE_START, spike_duration=AUDIO_SPIKE_DUR,
            spike_dbfs=-12.0, seed=1337,
        )
        chat_path = tmp_path / "chat.jsonl"
        audio_path = tmp_path / "audio.json"
        _write_chat_log(chat_path, chat)
        _write_audio_track(audio_path, audio)

        bus = EventBus()
        handler = StreamIngestHandler()
        handler.register(bus)
        received, spy = _make_spy()
        bus.subscribe_fn(MOMENT_DETECTED, spy)

        payload = StreamVodIngestedPayload(
            vod_id="vod_ingest_chat_no_subs_001",
            streamer_id="streamer_1",
            source_path="/fake/vods/vod_ingest_chat_no_subs_001.mp4",
            duration_sec=float(DURATION_SEC),
            chat_log_path=str(chat_path),
            audio_track_path=str(audio_path),
            subtitle_track_path="",
        )

        asyncio.run(bus.publish(STREAM_VOD_INGESTED, payload))
        assert len(received) == 1, (
            f"chat-having content with no subtitles must not be rejected, got {len(received)} event(s)"
        )
        print("  [PASS] chat-having content with no subtitles proceeds normally (not rejected)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX SENTINEL — Pillar 1 Ingest→Bus Integration Test")
    print("=" * 60 + "\n")
    test_ingest_publishes_moment_detected_for_spike()
    test_ingest_flat_baseline_publishes_nothing()
    test_ingest_honors_audio_track_bucket_sec()
    test_unregistered_handler_raises()
    test_ingest_recovers_archetype_b_moment_via_subtitle_track()
    test_ingest_rejects_when_no_chat_and_no_subtitles()
    test_ingest_with_chat_but_no_subtitles_is_not_rejected()
    print("  All Pillar 1 ingest integration tests passed.")
    print("=" * 60 + "\n")
