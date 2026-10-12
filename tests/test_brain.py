"""
tests/test_brain.py  —  Pillar 2 orchestrator (Brain) offline test
Apex Engine 2.0

Fully-mocked, offline, deterministic — fakes every OllamaClient touchpoint
(trend-script generation, clip-script generation, AND the startup health
check) so this never depends on a live Ollama call, per CLAUDE.md's
testing philosophy.

Focus: Brain's JobQueue wiring specifically. Per CLAUDE.md's "JobQueue
ownership" section, both ScriptGenerator (trend-to-script, via TREND_FOUND)
and ClipScriptGenerator (clip segment selection, via
submit_clip_script_job()) share Brain's single queue — same resource
shape, one queue. This test enqueues one of each, in a known order, and
proves both complete correctly through the one shared single-worker queue
with no starvation/reordering (the practical, testable proxy for "no
head-of-line blocking" between same-shape jobs).

Run:
    python tests/test_brain.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import (
    CLIP_SCRIPT_READY,
    SCRIPT_READY,
    TREND_FOUND,
    EventBus,
    MomentDetectedPayload,
    TrendFoundPayload,
)
from brain.brain import Brain
from brain.clip_script_generator import ClipScriptGenerator, TranscriptSegment
from brain.script_generator import ScriptGenerator
from tests.test_clip_script_generator import _FakeOllamaClient, _valid_json


def _valid_script_json(
    hook: str = "The shocking hook.",
    body: list[str] | None = None,
    cta: str = "Follow before they remove this.",
) -> str:
    return json.dumps({
        "hook": hook,
        "body": body or [f"point {i}" for i in range(1, 6)],
        "cta": cta,
    })


TREND = TrendFoundPayload(
    trend_id="trend_001",
    topic="test topic for brain queue sharing",
    hook_type="top_5",
    hook_count=5,
    viral_score=0.91,
    source="google_trends",
    source_url="https://trends.google.com/x",
    keywords=["alpha", "beta"],
    raw_title="raw title",
)

MOMENT = MomentDetectedPayload(
    vod_id="vod_queue_001",
    window_start=10.0,
    window_end=30.0,
    composite_score=5.0,
    chat_score=4.0,
    audio_score=3.0,
    top_spam_tokens=["lol"],
    detection_method="chat_audio_zscore_fusion",
)

SEGMENTS = [
    TranscriptSegment(id=i, start=10.0 + i * 5.0, end=10.0 + (i + 1) * 5.0, text=f"seg {i}")
    for i in range(4)
]


def _make_tagging_spy(tag: str, order: list[str], received: list) -> Callable[[object], Awaitable[None]]:
    async def _spy(payload) -> None:
        order.append(tag)
        received.append(payload)
    return _spy


# ── 1. Both job kinds enqueue onto and complete via Brain's ONE queue ────────

def test_shared_queue_processes_script_and_clip_jobs_in_fifo_order() -> None:
    bus = EventBus()

    completion_order: list[str] = []
    script_received: list = []
    clip_received: list = []
    bus.subscribe_fn(SCRIPT_READY, _make_tagging_spy("script", completion_order, script_received))
    bus.subscribe_fn(CLIP_SCRIPT_READY, _make_tagging_spy("clip", completion_order, clip_received))

    fake_script_ollama = _FakeOllamaClient([_valid_script_json()])
    fake_clip_ollama   = _FakeOllamaClient([_valid_json(0, 3, reason="fifo test clip")])
    fake_health        = _FakeOllamaClient([])  # only health_check() should ever be called on this one

    generator      = ScriptGenerator(client=fake_script_ollama)
    clip_generator = ClipScriptGenerator(client=fake_clip_ollama, bus=bus)

    brain = Brain(
        generator=generator,
        clip_generator=clip_generator,
        ollama_client=fake_health,
        bus=bus,
        queue_max_size=5,
    )

    # Confirms Brain literally owns one JobQueue instance shared by both paths.
    assert brain._generator is generator
    assert brain._clip_generator is clip_generator

    async def run() -> None:
        await brain.start()
        try:
            # Enqueue the clip job FIRST, then the trend job — if the shared
            # queue starved or reordered jobs, completion_order would show it.
            await brain.submit_clip_script_job(MOMENT, SEGMENTS)
            await bus.publish(TREND_FOUND, TREND)

            for _ in range(100):
                if len(script_received) and len(clip_received):
                    break
                await asyncio.sleep(0.02)
        finally:
            await brain.stop()

    asyncio.run(run())

    print(f"  completion order: {completion_order}")
    print(f"  fake_health calls: generate={len(fake_health.calls)} (must be 0)")

    assert len(clip_received) == 1, f"expected 1 CLIP_SCRIPT_READY, got {len(clip_received)}"
    assert len(script_received) == 1, f"expected 1 SCRIPT_READY, got {len(script_received)}"
    assert completion_order == ["clip", "script"], (
        f"expected FIFO completion order [clip, script] (submission order), "
        f"got {completion_order} — shared queue reordered or starved a job"
    )

    clip = clip_received[0]
    assert clip.vod_id == "vod_queue_001"
    assert clip.start_segment_id == 0
    assert clip.end_segment_id == 3

    script = script_received[0]
    assert script.trend_id == "trend_001"
    assert script.hook == "The shocking hook."

    assert fake_health.calls == [], "health-check client must only be used for health_check(), never generate()"
    print("  [PASS] clip job (queued first) and script job (queued second) "
          "both complete correctly, in FIFO order, on Brain's one shared queue\n")


# ── 2. A failing job on the shared queue must not block the job behind it ────

def test_failed_job_does_not_block_the_next_queued_job() -> None:
    bus = EventBus()
    completion_order: list[str] = []
    script_received: list = []
    clip_received: list = []
    bus.subscribe_fn(SCRIPT_READY, _make_tagging_spy("script", completion_order, script_received))
    bus.subscribe_fn(CLIP_SCRIPT_READY, _make_tagging_spy("clip", completion_order, clip_received))

    # Clip job's Ollama call returns garbage on both attempts (retry also
    # fails) -> ClipScriptGenerationError, swallowed by _generate_clip_and_emit.
    fake_clip_ollama = _FakeOllamaClient(["not json", "still not json"])
    fake_script_ollama = _FakeOllamaClient([_valid_script_json()])
    fake_health = _FakeOllamaClient([])

    generator      = ScriptGenerator(client=fake_script_ollama)
    clip_generator = ClipScriptGenerator(client=fake_clip_ollama, bus=bus)
    brain = Brain(
        generator=generator, clip_generator=clip_generator,
        ollama_client=fake_health, bus=bus, queue_max_size=5,
    )

    async def run() -> None:
        await brain.start()
        try:
            await brain.submit_clip_script_job(MOMENT, SEGMENTS)  # will fail
            await bus.publish(TREND_FOUND, TREND)                  # must still succeed

            for _ in range(100):
                if len(script_received):
                    break
                await asyncio.sleep(0.02)
        finally:
            await brain.stop()

    asyncio.run(run())

    assert clip_received == [], "the broken clip job must not have published anything"
    assert len(script_received) == 1, (
        "a failed clip job must not block the script job queued behind it on the shared worker"
    )
    print("  [PASS] failed clip job is isolated — script job behind it on the "
          "shared queue still completes\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX BRAIN — Pillar 2 Shared JobQueue Test")
    print("=" * 60 + "\n")
    test_shared_queue_processes_script_and_clip_jobs_in_fifo_order()
    test_failed_job_does_not_block_the_next_queued_job()
    print("  All Brain shared-queue tests passed.")
    print("=" * 60 + "\n")
