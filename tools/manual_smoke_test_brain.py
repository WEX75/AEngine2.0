"""
tools/manual_smoke_test_brain.py  —  Module 2 (The Brain) Live Ollama Test
Apex Engine 2.0

Manual smoke test, NOT a pytest test — it requires a live, running Ollama
instance and a pulled model, which violates CLAUDE.md's testing philosophy
("never let a test depend on a live Ollama call"). Deliberately kept out of
tests/ and out of the repo root (and out of the test_*.py naming convention)
so pytest never collects it by accident.

Injects mock TrendFoundPayloads directly into ScriptGenerator, bypassing
the EventBus and The Sentinel entirely. Tests the full Ollama → JSON
parsing → ViralScript pipeline with your local llama3:8b model.

Usage
-----
  python tools/manual_smoke_test_brain.py                    # runs all 3 test cases
  python tools/manual_smoke_test_brain.py --topic "custom topic here"  # one custom case

Requirements
------------
  Ollama must be running: ollama serve
  Model must be pulled:   ollama pull llama3:8b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.ollama_client import OllamaClient, OllamaError
from brain.script_generator import ScriptGenerator, ScriptGenerationError
from core.event_bus import TrendFoundPayload


# ── Mock payloads — one per hook type ────────────────────────────────────────

MOCK_PAYLOADS: list[TrendFoundPayload] = [
    TrendFoundPayload(
        trend_id    = "test_top5_001",
        topic       = "world records broken in 2024",
        hook_type   = "top_5",
        hook_count  = 5,
        viral_score = 0.97,
        source      = "google_trends",
        source_url  = "https://trends.google.com/trends/explore?q=world+records",
        keywords    = ["world", "record", "broken", "fastest", "strongest"],
        raw_title   = "top 5 world records broken in 2024",
    ),
    TrendFoundPayload(
        trend_id    = "test_didyouknow_002",
        topic       = "shocking facts scientists discovered about the deep ocean",
        hook_type   = "did_you_know",
        hook_count  = 5,
        viral_score = 0.74,
        source      = "google_trends",
        source_url  = "https://trends.google.com/trends/explore?q=ocean+discovery",
        keywords    = ["ocean", "scientists", "discovered", "creature", "deep"],
        raw_title   = "shocking facts scientists discovered this week",
    ),
    TrendFoundPayload(
        trend_id    = "test_generic_003",
        topic       = "Copa America final results and record-breaking moments",
        hook_type   = "generic",
        hook_count  = 5,
        viral_score = 0.62,
        source      = "google_trends",
        source_url  = "https://trends.google.com/trends/explore?q=Copa+America",
        keywords    = ["copa", "america", "final", "record", "goal"],
        raw_title   = "Copa America final match results",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _separator(title: str = "", char: str = "─", width: int = 62) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}\n")
    else:
        print("─" * width)


def _print_script(script) -> None:
    """Pretty-prints a ViralScript to stdout."""
    print(f"  Trend ID   : {script.trend_id}")
    print(f"  Topic      : {script.topic}")
    print(f"  Hook Type  : {script.hook_type} ({script.hook_count} points)")
    print(f"  Generated  : {script.generated_at.strftime('%H:%M:%S UTC')}")
    print(f"  Words      : {script.word_count()}")
    print(f"  Est. TTS   : ~{script.estimated_duration_seconds():.0f}s\n")

    print(f"  [HOOK]")
    print(f"    {script.hook}\n")

    print(f"  [BODY]")
    for i, point in enumerate(script.body, 1):
        print(f"    {i}. {point}")

    print(f"\n  [CTA]")
    print(f"    {script.cta}")

    print(f"\n  [FULL SCRIPT FOR TTS]")
    print(f"    {script.full_script}")


# ── Step 1: Health check ──────────────────────────────────────────────────────

async def check_ollama() -> bool:
    _separator("Ollama Health Check")
    client = OllamaClient()

    print("  Checking connection to Ollama...")
    is_up = await client.health_check()

    if is_up:
        print("  [OK] Ollama is running and reachable.")
        return True
    else:
        print("  [FAIL] Ollama is NOT reachable.")
        print()
        print("  Fix: Open a new terminal and run:")
        print("       ollama serve")
        print()
        print("  Then ensure the model is downloaded:")
        print("       ollama pull llama3:8b")
        print()
        return False


# ── Step 2: Run test cases ────────────────────────────────────────────────────

async def run_test(generator: ScriptGenerator, payload: TrendFoundPayload) -> bool:
    """Run a single test case. Returns True on success."""
    _separator(f"Test: {payload.hook_type.upper()}")
    print(f"  Input topic  : {payload.topic}")
    print(f"  Hook type    : {payload.hook_type}")
    print(f"  Hook count   : {payload.hook_count}")
    print(f"  Keywords     : {', '.join(payload.keywords[:5])}")
    print()
    print("  Calling Ollama... (may take 10-30s on CPU)")

    t0 = time.monotonic()
    try:
        script = await generator.generate(payload)
        elapsed = time.monotonic() - t0
        print(f"  [OK] Generated in {elapsed:.1f}s\n")
        _separator("Script Output")
        _print_script(script)

        # Basic assertions
        assert script.hook,              "hook is empty"
        assert len(script.body) >= 3,    f"body has {len(script.body)} items (min 3)"
        assert script.cta,               "cta is empty"
        assert script.full_script,       "full_script is empty"
        assert script.word_count() > 10, "script is too short"

        print(f"\n  [PASS] All assertions passed.")
        return True

    except ScriptGenerationError as exc:
        elapsed = time.monotonic() - t0
        print(f"  [FAIL] ScriptGenerationError after {elapsed:.1f}s: {exc}")
        return False
    except OllamaError as exc:
        elapsed = time.monotonic() - t0
        print(f"  [FAIL] OllamaError after {elapsed:.1f}s: {exc}")
        return False
    except AssertionError as exc:
        print(f"  [FAIL] Assertion: {exc}")
        return False


# ── Step 3: Raw JSON probe ────────────────────────────────────────────────────

async def probe_raw_json(client: OllamaClient) -> None:
    """
    Sends a minimal prompt to test that Ollama's format:json is working.
    Useful for diagnosing JSON formatting issues without the full pipeline.
    """
    _separator("Raw JSON Probe")
    print("  Sending minimal test prompt to verify format:json...\n")

    prompt = (
        'Return ONLY this exact JSON with no other text: '
        '{"status": "ok", "model": "working", "json_mode": true}'
    )

    try:
        result = await client.generate(prompt=prompt, expect_json=True, max_tokens=50)
        raw = result["text"]
        print(f"  Raw response: {raw}")
        data = json.loads(raw)
        print(f"  Parsed OK: {data}")
        print(f"  [OK] format:json is working correctly.")
    except (OllamaError, json.JSONDecodeError) as exc:
        print(f"  [FAIL] {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(custom_topic: str | None = None) -> None:
    print()
    print("=" * 62)
    print("  APEX ENGINE 2.0 — Module 2 (The Brain) Live Test")
    print("=" * 62)

    # Health check
    is_up = await check_ollama()
    if not is_up:
        sys.exit(1)

    client    = OllamaClient()
    generator = ScriptGenerator()

    # Raw JSON probe first
    await probe_raw_json(client)

    # Test cases
    if custom_topic:
        payloads = [TrendFoundPayload(
            trend_id    = "custom_test_001",
            topic       = custom_topic,
            hook_type   = "did_you_know",
            hook_count  = 5,
            viral_score = 1.0,
            source      = "manual",
            source_url  = "",
            keywords    = custom_topic.lower().split()[:5],
            raw_title   = custom_topic,
        )]
    else:
        payloads = MOCK_PAYLOADS

    results: list[bool] = []
    for payload in payloads:
        ok = await run_test(generator, payload)
        results.append(ok)
        if len(payloads) > 1:
            print("\n  (pausing 2s between tests to be gentle on the CPU...)")
            await asyncio.sleep(2)

    # Summary
    _separator("Results Summary")
    passed = sum(results)
    total  = len(results)
    print(f"  {passed}/{total} tests passed.\n")

    if passed == total:
        print("  Module 2 (The Brain): VERIFIED")
        print("  The full TREND_FOUND -> ViralScript pipeline is working.")
    else:
        print("  Some tests failed. Review the output above for details.")
        sys.exit(1)

    print()
    print("=" * 62)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apex Engine 2.0 — Brain Module Live Test"
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Custom topic to test (runs a single test case instead of the defaults).",
    )
    args = parser.parse_args()
    asyncio.run(main(custom_topic=args.topic))
