"""
tests/test_clip_orchestrator.py  —  stream-clipping orchestrator test
Apex Engine 2.0

Fully-mocked, offline, deterministic — no real ffmpeg, no real Whisper, no
live Ollama call. clip_whisper_bridge and clip_renderer are both replaced
with lightweight fakes that just record what they were called with; this
file's job is proving ClipOrchestrator's OWN logic (registry + wiring +
unknown-vod_id handling), not re-proving bridge/renderer internals already
covered by their own test files.

Five things need proving:
  1. The full event chain resolves paths/streamer_id correctly:
     STREAM_VOD_INGESTED -> MOMENT_DETECTED -> bridge.submit_moment() with
     the right source_path -> CLIP_SCRIPT_READY -> renderer.submit_render()
     with the right source_path/streamer_id.
  2. A MOMENT_DETECTED or CLIP_SCRIPT_READY for a vod_id never registered
     (or already evicted) is logged and dropped, not crashed.
  3. The registry is bounded — inserting past max_registry_size evicts
     the least-recently-used entry.
  4. Accessing an entry (a MOMENT_DETECTED/CLIP_SCRIPT_READY hit) refreshes
     its recency, protecting still-in-flight VODs from premature eviction
     by newer, unrelated VODs.
  5. apex.py's stream_clipping.enabled config flag reads correctly both
     ways (the actual on/off switch this orchestrator is wired behind).

Run:
    python tests/test_clip_orchestrator.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import (
    CLIP_SCRIPT_READY,
    MOMENT_DETECTED,
    STREAM_VOD_INGESTED,
    ClipScriptReadyPayload,
    EventBus,
    MomentDetectedPayload,
    StreamVodIngestedPayload,
)
from orchestrator.clip_orchestrator import ClipOrchestrator

# ── Fakes ──────────────────────────────────────────────────────────────────────


class _FakeBridge:
    """Stands in for ClipWhisperBridge — records submit_moment() calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str, tuple[float, float]]] = []

    async def submit_moment(self, moment, source_path, target_duration_range=None) -> None:
        self.calls.append((moment, source_path, target_duration_range))


class _FakeRenderer:
    """Stands in for ClipRenderer — records submit_render() calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str, str]] = []

    async def submit_render(self, clip, source_path, streamer_id, **kwargs) -> None:
        self.calls.append((clip, source_path, streamer_id))


def _make_ingested(vod_id: str, source_path: str, streamer_id: str) -> StreamVodIngestedPayload:
    return StreamVodIngestedPayload(
        vod_id=vod_id,
        streamer_id=streamer_id,
        source_path=source_path,
        duration_sec=600.0,
        chat_log_path=f"/fake/{vod_id}_chat.jsonl",
        audio_track_path=f"/fake/{vod_id}_audio.json",
    )


def _make_moment(vod_id: str, window_start: float = 10.0) -> MomentDetectedPayload:
    return MomentDetectedPayload(
        vod_id=vod_id, window_start=window_start, window_end=window_start + 20.0, composite_score=5.0,
    )


def _make_clip(vod_id: str) -> ClipScriptReadyPayload:
    return ClipScriptReadyPayload(
        vod_id=vod_id,
        start_segment_id=0,
        end_segment_id=1,
        caption_segments=[{"id": 0, "start": 10.0, "end": 15.0, "text": "hi"}],
        hook_reason="test",
    )


# ── 1. Full event chain: paths/streamer_id resolved correctly end to end ─────

def test_full_event_chain_resolves_paths_correctly() -> None:
    bus = EventBus()
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    orchestrator = ClipOrchestrator(bridge=bridge, renderer=renderer, bus=bus)

    ingested = _make_ingested("vod_1", source_path="/videos/vod_1.mp4", streamer_id="streamer_a")
    moment = _make_moment("vod_1")
    clip = _make_clip("vod_1")

    async def run() -> None:
        await orchestrator.start()
        try:
            await bus.publish(STREAM_VOD_INGESTED, ingested)
            await bus.publish(MOMENT_DETECTED, moment)
            await bus.publish(CLIP_SCRIPT_READY, clip)
        finally:
            await orchestrator.stop()

    asyncio.run(run())

    assert len(bridge.calls) == 1
    called_moment, source_path, target_range = bridge.calls[0]
    assert called_moment is moment
    assert source_path == "/videos/vod_1.mp4"
    assert target_range == (12.0, 75.0)

    assert len(renderer.calls) == 1
    called_clip, source_path2, streamer_id = renderer.calls[0]
    assert called_clip is clip
    assert source_path2 == "/videos/vod_1.mp4"
    assert streamer_id == "streamer_a"
    print("  [PASS] full chain resolves source_path/streamer_id correctly for both hand-offs\n")


# ── 2. Unknown vod_id: logged and dropped, never crashes ─────────────────────

def test_unknown_vod_id_dropped_without_crash() -> None:
    bus = EventBus()
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    orchestrator = ClipOrchestrator(bridge=bridge, renderer=renderer, bus=bus)

    moment = _make_moment("vod_never_ingested")
    clip = _make_clip("vod_also_never_ingested")

    async def run() -> None:
        await orchestrator.start()
        try:
            # No STREAM_VOD_INGESTED published for either vod_id.
            await bus.publish(MOMENT_DETECTED, moment)
            await bus.publish(CLIP_SCRIPT_READY, clip)
        finally:
            await orchestrator.stop()

    asyncio.run(run())   # must not raise

    assert bridge.calls == [], "bridge must never be called for an unregistered vod_id"
    assert renderer.calls == [], "renderer must never be called for an unregistered vod_id"
    print("  [PASS] MOMENT_DETECTED/CLIP_SCRIPT_READY for unknown vod_id dropped cleanly, no crash\n")


# ── 3. Registry is bounded: over-capacity insert evicts the LRU entry ────────

def test_registry_evicts_least_recently_used_when_over_capacity() -> None:
    bus = EventBus()
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    orchestrator = ClipOrchestrator(bridge=bridge, renderer=renderer, bus=bus, max_registry_size=2)

    async def run() -> None:
        await orchestrator.start()
        try:
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_a", "/v/a.mp4", "s"))
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_b", "/v/b.mp4", "s"))
            # Registry now at capacity (2): vod_a, vod_b — neither touched since insert.
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_c", "/v/c.mp4", "s"))
            # vod_a is the least-recently-used -> should have been evicted.
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_a"))
        finally:
            await orchestrator.stop()

    asyncio.run(run())

    assert bridge.calls == [], "vod_a should have been evicted (LRU) — its MOMENT_DETECTED must be dropped"
    print("  [PASS] registry bounded at max_registry_size — inserting past capacity evicts the LRU entry\n")


# ── 4. Access refreshes recency, protecting in-flight VODs from eviction ─────

def test_registry_access_refreshes_recency() -> None:
    bus = EventBus()
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    orchestrator = ClipOrchestrator(bridge=bridge, renderer=renderer, bus=bus, max_registry_size=2)

    async def run() -> None:
        await orchestrator.start()
        try:
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_a", "/v/a.mp4", "s"))
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_b", "/v/b.mp4", "s"))
            # Touch vod_a via a MOMENT_DETECTED hit -> refreshes its recency to "most recent".
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_a"))
            # Now vod_b is the LRU one, not vod_a. Inserting vod_c should evict vod_b.
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_c", "/v/c.mp4", "s"))
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_b"))   # should be dropped
            await bus.publish(CLIP_SCRIPT_READY, _make_clip("vod_a"))   # should still work
        finally:
            await orchestrator.stop()

    asyncio.run(run())

    assert len(bridge.calls) == 1, f"expected exactly 1 bridge call (vod_a's), got {len(bridge.calls)}"
    assert bridge.calls[0][0].vod_id == "vod_a"
    assert len(renderer.calls) == 1
    assert renderer.calls[0][0].vod_id == "vod_a"
    print("  [PASS] accessing an entry refreshes its recency — protects it from eviction by newer VODs\n")


# ── 5. apex.py's stream_clipping.enabled config flag reads correctly both ways ─

def test_stream_clipping_config_flag_reads_correctly() -> None:
    from apex import _stream_clipping_enabled

    assert _stream_clipping_enabled({}) is False, "missing section must default to disabled"
    assert _stream_clipping_enabled({"stream_clipping": {}}) is False, "missing key must default to disabled"
    assert _stream_clipping_enabled({"stream_clipping": {"enabled": False}}) is False
    assert _stream_clipping_enabled({"stream_clipping": {"enabled": True}}) is True

    from core.config_loader import get_config
    real_cfg = get_config()
    assert _stream_clipping_enabled(real_cfg) is False, (
        "settings.yaml's real default must be disabled — trend-mode and clip-mode "
        "must be able to run independently, and clip-mode is opt-in"
    )
    print("  [PASS] stream_clipping.enabled flag reads correctly (default off, explicit on/off both honored)\n")


# ── 6. Per-streamer target_duration_range override ───────────────────────────
# Three things need proving:
#   a. Unset (no target_duration_ranges configured) preserves today's exact
#      behavior — the constructor's target_duration_range default reaches
#      bridge.submit_moment() unchanged.
#   b. A configured per-streamer override is resolved correctly and reaches
#      bridge.submit_moment() instead of the default, while an UNCONFIGURED
#      streamer in the same orchestrator still gets the default.
#   c. Malformed entries raise ValueError at construction, never silently
#      misbehave at render time.
#   d. The override doesn't just reach bridge.submit_moment() — it flows
#      all the way through ClipWhisperBridge -> Brain.select_clip_script()
#      -> ClipScriptGenerator.generate(), observably changing the final
#      CLIP_SCRIPT_READY selection (auto-repair trims/pads differently for
#      a tighter range than for the default).

def test_target_duration_range_unset_preserves_default_behavior() -> None:
    bus = EventBus()
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    orchestrator = ClipOrchestrator(bridge=bridge, renderer=renderer, bus=bus)

    async def run() -> None:
        await orchestrator.start()
        try:
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_1", "/v/1.mp4", "streamer_a"))
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_1"))
        finally:
            await orchestrator.stop()

    asyncio.run(run())

    assert len(bridge.calls) == 1
    _, _, target_range = bridge.calls[0]
    assert target_range == (12.0, 75.0), "no override configured -> the constructor default must reach submit_moment()"
    print("  [PASS] unset target_duration_ranges preserves the constructor default end to end\n")


def test_target_duration_range_override_resolved_per_streamer() -> None:
    bus = EventBus()
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    orchestrator = ClipOrchestrator(
        bridge=bridge, renderer=renderer, bus=bus,
        target_duration_ranges={"streamer_short": (10.0, 20.0)},
    )

    async def run() -> None:
        await orchestrator.start()
        try:
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_short", "/v/short.mp4", "streamer_short"))
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_other", "/v/other.mp4", "streamer_unconfigured"))
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_short"))
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_other"))
        finally:
            await orchestrator.stop()

    asyncio.run(run())

    assert len(bridge.calls) == 2
    ranges_by_vod = {call[0].vod_id: call[2] for call in bridge.calls}
    assert ranges_by_vod["vod_short"] == (10.0, 20.0), "configured streamer must get its per-streamer override"
    assert ranges_by_vod["vod_other"] == (12.0, 75.0), "unconfigured streamer in the SAME orchestrator must still get the default"
    print("  [PASS] target_duration_range override resolved per streamer_id, default preserved for the rest\n")


def test_invalid_target_duration_range_fails_loudly_at_construction() -> None:
    bridge, renderer = _FakeBridge(), _FakeRenderer()
    bad_ranges = [
        {"streamer_x": (75.0, 12.0)},     # min > max
        {"streamer_x": (0.0, 20.0)},      # min not > 0
        {"streamer_x": (10.0, 10.0)},     # min == max
        {"streamer_x": "not_a_range"},    # not a 2-element sequence
        {"streamer_x": (10.0,)},          # wrong arity
    ]
    for bad in bad_ranges:
        try:
            ClipOrchestrator(bridge=bridge, renderer=renderer, bus=EventBus(), target_duration_ranges=bad)
        except ValueError as exc:
            print(f"  {bad} raised: {exc}")
        else:
            raise AssertionError(f"invalid target_duration_range {bad!r} must raise ValueError")
    print("  [PASS] malformed target_duration_range entries all raise ValueError at construction\n")


def test_target_duration_range_override_reaches_clip_script_generator() -> None:
    # Real ClipWhisperBridge + real Brain (fake Ollama only) wired through a
    # real ClipOrchestrator — proves the per-streamer override isn't just
    # resolved locally, it actually changes ClipScriptGenerator.generate()'s
    # OWN auto-repair output for a real selection.
    from brain.brain import Brain
    from brain.clip_script_generator import ClipScriptGenerator
    from brain.clip_whisper_bridge import ClipWhisperBridge
    from brain.script_generator import ScriptGenerator
    from tests.test_clip_script_generator import _FakeOllamaClient, _valid_json
    from tests.test_clip_whisper_bridge import SEGMENT_TEMPLATE, _TrackingTranscriber, _FakeExtractor

    bus = EventBus()
    received: list[ClipScriptReadyPayload] = []

    async def _spy(payload: ClipScriptReadyPayload) -> None:
        received.append(payload)

    bus.subscribe_fn(CLIP_SCRIPT_READY, _spy)

    # Same canned selection (segments 0-2, i.e. [0.0s, 19.0s) — 19s span)
    # returned for BOTH vods; any difference in the final CLIP_SCRIPT_READY
    # must come from auto-repair reacting to a different target_duration_range,
    # not from a different LLM response.
    fake_clip_ollama = _FakeOllamaClient([_valid_json(0, 2), _valid_json(0, 2)])
    brain = Brain(
        generator=ScriptGenerator(client=_FakeOllamaClient([])),
        clip_generator=ClipScriptGenerator(client=fake_clip_ollama, bus=bus),
        ollama_client=_FakeOllamaClient([]),
        bus=bus,
    )
    bridge = ClipWhisperBridge(
        brain=brain, extractor=_FakeExtractor(), transcriber=_TrackingTranscriber(),
    )
    renderer = _FakeRenderer()
    orchestrator = ClipOrchestrator(
        bridge=bridge, renderer=renderer, bus=bus,
        # Default (12, 75) leaves the 19s selection untouched; "streamer_tight"
        # is pinned well below it, forcing auto-repair to trim.
        target_duration_ranges={"streamer_tight": (1.0, 3.0)},
    )

    async def run() -> None:
        await brain.start()
        await bridge.start()
        await orchestrator.start()
        try:
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_default", "/v/default.mp4", "streamer_default"))
            await bus.publish(STREAM_VOD_INGESTED, _make_ingested("vod_tight", "/v/tight.mp4", "streamer_tight"))
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_default", window_start=100.0))
            await bus.publish(MOMENT_DETECTED, _make_moment("vod_tight", window_start=100.0))
            await bridge.drain()
            await brain.drain()
        finally:
            await orchestrator.stop()
            await bridge.stop()
            await brain.stop()

    asyncio.run(run())

    assert len(received) == 2, f"expected 2 real CLIP_SCRIPT_READY events, got {len(received)}"
    by_vod = {c.vod_id: c for c in received}

    default_clip = by_vod["vod_default"]
    tight_clip = by_vod["vod_tight"]

    default_duration = default_clip.caption_segments[-1]["end"] - default_clip.caption_segments[0]["start"]
    tight_duration = tight_clip.caption_segments[-1]["end"] - tight_clip.caption_segments[0]["start"]
    print(f"  default range (12,75) -> duration {default_duration}s | tight range (1,3) -> duration {tight_duration}s")

    assert default_clip.end_segment_id == 2, "default (12,75) range must leave the LLM's own 19s selection untouched"
    assert default_duration == 19.0
    assert tight_duration < default_duration, (
        "the per-streamer (1.0, 3.0) override must have forced ClipScriptGenerator's OWN auto-repair "
        "to trim the selection tighter than the default-range clip got"
    )
    print("  [PASS] per-streamer target_duration_range override reaches ClipScriptGenerator.generate() "
          "and observably changes its real auto-repair output\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX — Stream-Clipping Orchestrator Test")
    print("=" * 60 + "\n")
    test_full_event_chain_resolves_paths_correctly()
    test_unknown_vod_id_dropped_without_crash()
    test_registry_evicts_least_recently_used_when_over_capacity()
    test_registry_access_refreshes_recency()
    test_stream_clipping_config_flag_reads_correctly()
    test_target_duration_range_unset_preserves_default_behavior()
    test_target_duration_range_override_resolved_per_streamer()
    test_invalid_target_duration_range_fails_loudly_at_construction()
    test_target_duration_range_override_reaches_clip_script_generator()
    print("  All orchestrator tests passed.")
    print("=" * 60 + "\n")
