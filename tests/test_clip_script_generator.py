"""
tests/test_clip_script_generator.py  —  Pillar 2 clip-selection offline test
Apex Engine 2.0

Fully-mocked, offline, deterministic — no live Ollama call (per CLAUDE.md's
testing philosophy). OllamaClient is replaced with a fake that returns a
canned queue of raw text responses and records every prompt it was called
with, so we can assert both on the final CLIP_SCRIPT_READY event and on the
retry behavior (call count, and that the second prompt actually carries the
error context).

Run:
    python tests/test_clip_script_generator.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import (
    CLIP_SCRIPT_READY,
    EventBus,
    MomentDetectedPayload,
)
from brain.clip_script_generator import (
    ClipScriptGenerationError,
    ClipScriptGenerator,
    TranscriptSegment,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

# 10 segments, 3s each, ids 0-9, spanning 170.0s -> 200.0s (30s total).
SEGMENTS: list[TranscriptSegment] = [
    TranscriptSegment(id=i, start=170.0 + i * 3.0, end=170.0 + (i + 1) * 3.0, text=f"segment {i} text")
    for i in range(10)
]

MOMENT = MomentDetectedPayload(
    vod_id="vod_clip_001",
    window_start=176.0,
    window_end=191.0,
    composite_score=6.4,
    chat_score=7.1,
    audio_score=5.0,
    top_spam_tokens=["clip", "it", "omegalul"],
    detection_method="chat_audio_zscore_fusion",
)


class _FakeOllamaClient:
    """
    Returns a pre-queued sequence of raw text responses; records prompts.
    Also usable as a fake for OllamaClient.health_check() (always reports
    healthy) — reused by tests/test_brain.py so Brain's startup health
    check never touches the network either.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("_FakeOllamaClient ran out of canned responses")
        text = self._responses.pop(0)
        return {"text": text, "model": "fake-model", "done": True, "tokens": 0}

    async def health_check(self) -> bool:
        return True


def _make_spy() -> tuple[list[Any], Any]:
    received: list[Any] = []

    async def _spy(payload: Any) -> None:
        received.append(payload)

    return received, _spy


def _valid_json(start: int, end: int, confidence: float = 0.87, reason: str = "escalating hype moment") -> str:
    return json.dumps({
        "start_segment_id": start,
        "end_segment_id": end,
        "confidence_score": confidence,
        "hook_reason": reason,
    })


# ── 1. Valid response → exactly one CLIP_SCRIPT_READY, no repair, one call ───

def test_valid_response_publishes_clip_script_ready() -> None:
    fake = _FakeOllamaClient([_valid_json(2, 7)])
    bus = EventBus()
    received, spy = _make_spy()
    bus.subscribe_fn(CLIP_SCRIPT_READY, spy)

    generator = ClipScriptGenerator(client=fake, bus=bus)
    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS))

    print(f"  Ollama calls: {len(fake.calls)}")
    print(f"  start={payload.start_segment_id} end={payload.end_segment_id} "
          f"captions={len(payload.caption_segments)} reason='{payload.hook_reason}'")

    assert len(fake.calls) == 1, f"expected exactly 1 Ollama call, got {len(fake.calls)}"
    assert len(received) == 1, f"expected exactly 1 CLIP_SCRIPT_READY, got {len(received)}"
    assert received[0] is payload

    assert payload.vod_id == "vod_clip_001"
    assert payload.start_segment_id == 2
    assert payload.end_segment_id == 7
    assert payload.hook_reason == "escalating hype moment"
    assert len(payload.caption_segments) == 6  # ids 2..7 inclusive
    assert [c["id"] for c in payload.caption_segments] == [2, 3, 4, 5, 6, 7]
    assert payload.caption_segments[0]["text"] == "segment 2 text"
    print("  [PASS] valid LLM response -> correct CLIP_SCRIPT_READY, no repair needed\n")


# ── 2. Malformed JSON → retry fires once, second attempt succeeds ────────────

def test_malformed_json_triggers_retry_then_succeeds() -> None:
    garbage = "Sure! Here's the JSON you asked for: {start_segment_id: 2, broken..."
    fake = _FakeOllamaClient([garbage, _valid_json(3, 6, reason="recovered on retry")])
    bus = EventBus()
    received, spy = _make_spy()
    bus.subscribe_fn(CLIP_SCRIPT_READY, spy)

    generator = ClipScriptGenerator(client=fake, bus=bus)
    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS))

    print(f"  Ollama calls: {len(fake.calls)}")

    assert len(fake.calls) == 2, f"expected exactly 2 Ollama calls (1 retry), got {len(fake.calls)}"
    assert "REJECTED" in fake.calls[1], (
        "second prompt must carry the error context from the first (reflection) — "
        "the retry template block wasn't triggered"
    )
    assert "REJECTED" not in fake.calls[0], "first prompt must not pre-emptively include retry error context"

    assert len(received) == 1
    assert payload.start_segment_id == 3
    assert payload.end_segment_id == 6
    assert payload.hook_reason == "recovered on retry"
    print("  [PASS] malformed JSON triggers exactly one reflection retry, which then succeeds\n")


def test_retry_exhausted_raises_cleanly() -> None:
    """Both attempts malformed — must raise a clean, typed error, not crash or hang."""
    fake = _FakeOllamaClient(["not json at all", "{\"still\": \"not a clip script\"}"])
    bus = EventBus()
    generator = ClipScriptGenerator(client=fake, bus=bus)

    try:
        asyncio.run(generator.generate(MOMENT, SEGMENTS))
        raise AssertionError("expected ClipScriptGenerationError after 2 bad attempts")
    except ClipScriptGenerationError as exc:
        assert len(fake.calls) == 2, f"expected exactly 2 attempts before giving up, got {len(fake.calls)}"
        print(f"  [PASS] both attempts invalid -> clean ClipScriptGenerationError, no crash: {exc}\n")


# ── 3. Out-of-range segment_id → auto-repair engages, NO retry ───────────────

def test_out_of_range_segment_id_triggers_auto_repair() -> None:
    fake = _FakeOllamaClient([_valid_json(-3, 999)])  # well-typed ints, way out of bounds
    bus = EventBus()
    received, spy = _make_spy()
    bus.subscribe_fn(CLIP_SCRIPT_READY, spy)

    generator = ClipScriptGenerator(client=fake, bus=bus)
    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS))

    print(f"  Ollama calls: {len(fake.calls)}  ->  repaired start={payload.start_segment_id} end={payload.end_segment_id}")

    assert len(fake.calls) == 1, (
        f"out-of-range ids are well-typed — must be auto-repaired, NOT trigger a retry "
        f"(got {len(fake.calls)} calls)"
    )
    assert payload.start_segment_id == 0, "start_segment_id must be clamped to the lowest valid id (0)"
    assert payload.end_segment_id == 9, "end_segment_id must be clamped to the highest valid id (9)"
    assert len(payload.caption_segments) == 10
    print("  [PASS] out-of-range ids auto-repaired (clamped) without any retry\n")


def test_end_before_start_is_swapped() -> None:
    fake = _FakeOllamaClient([_valid_json(7, 3)])  # end < start, both in-range
    bus = EventBus()
    generator = ClipScriptGenerator(client=fake, bus=bus)

    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS))

    assert len(fake.calls) == 1, "swap repair must not trigger a retry"
    assert payload.start_segment_id == 3
    assert payload.end_segment_id == 7
    print("  [PASS] end_segment_id < start_segment_id auto-repaired by swapping\n")


def test_short_window_is_padded_to_target_minimum() -> None:
    fake = _FakeOllamaClient([_valid_json(5, 5)])  # single 3s segment, min default is 12s
    bus = EventBus()
    generator = ClipScriptGenerator(client=fake, bus=bus)

    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS))

    assert len(fake.calls) == 1, "duration padding must not trigger a retry"
    assert payload.start_segment_id == 5
    assert payload.end_segment_id == 8, f"expected padding out to id 8 (12.0s), got {payload.end_segment_id}"
    print("  [PASS] too-short window padded outward to the target minimum duration\n")


def test_long_window_is_trimmed_to_target_maximum() -> None:
    fake = _FakeOllamaClient([_valid_json(0, 9)])  # full 30s range
    bus = EventBus()
    generator = ClipScriptGenerator(client=fake, bus=bus)

    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS, target_duration_range=(1.0, 10.0)))

    assert len(fake.calls) == 1, "duration trimming must not trigger a retry"
    assert payload.start_segment_id == 0
    assert payload.end_segment_id == 2, f"expected trim down to id 2 (9.0s <= 10.0s max), got {payload.end_segment_id}"
    print("  [PASS] too-long window trimmed down to the target maximum duration\n")


# ── 8. publish=False: selection without publication ──────────────────────────
# The transcript-boundary widen-retry (brain/clip_whisper_bridge.py) needs a
# first-attempt selection it can INSPECT without the truncated result ever
# reaching the bus — downstream would render it.

def test_generate_publish_false_returns_payload_without_publishing() -> None:
    fake = _FakeOllamaClient([_valid_json(2, 7)])
    bus = EventBus()
    received, spy = _make_spy()
    bus.subscribe_fn(CLIP_SCRIPT_READY, spy)

    generator = ClipScriptGenerator(client=fake, bus=bus)
    payload = asyncio.run(generator.generate(MOMENT, SEGMENTS, publish=False))

    assert payload.start_segment_id == 2 and payload.end_segment_id == 7
    assert received == [], (
        "publish=False must NOT publish CLIP_SCRIPT_READY — a boundary-"
        "touching selection published here would get rendered truncated"
    )
    print("  [PASS] publish=False returns the selection without publishing\n")


# ── 9. Prompt: narrative resolution over mechanical minimum ──────────────────

def test_prompt_instructs_narrative_resolution_over_minimum() -> None:
    generator = ClipScriptGenerator(client=_FakeOllamaClient([]), bus=EventBus())
    prompt = generator._render_prompt(MOMENT, SEGMENTS, (12.0, 75.0), None)

    assert "narrative resolution" in prompt.lower(), (
        "prompt must tell the LLM to end at a genuine narrative resolution "
        "(payoff delivered, reaction completed), not merely meet the minimum"
    )
    assert "floor" in prompt.lower(), (
        "prompt must frame the duration minimum as a floor, not the goal"
    )
    print("  [PASS] prompt instructs narrative-resolution endings over mechanical minimums\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX BRAIN — Pillar 2 Clip Script Generator Test")
    print("=" * 60 + "\n")
    test_valid_response_publishes_clip_script_ready()
    test_malformed_json_triggers_retry_then_succeeds()
    test_retry_exhausted_raises_cleanly()
    test_out_of_range_segment_id_triggers_auto_repair()
    test_end_before_start_is_swapped()
    test_short_window_is_padded_to_target_minimum()
    test_long_window_is_trimmed_to_target_maximum()
    test_generate_publish_false_returns_payload_without_publishing()
    test_prompt_instructs_narrative_resolution_over_minimum()
    print("  All Pillar 2 clip-script tests passed.")
    print("=" * 60 + "\n")
