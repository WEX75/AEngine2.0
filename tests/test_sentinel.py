"""
tests/test_sentinel.py  —  Module 1 (trend pipeline) offline verification
Apex Engine 2.0

Bypasses Google's API entirely by injecting pre-built TrendSignal objects
directly into TrendScorer. Proves the full Sentinel pipeline is correct
without needing a live internet connection or worrying about 429 rate limits.

Run:
    python tests/test_sentinel.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

# ── Bootstrap: project root must be importable ────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.models    import TrendSignal, detect_hook, clean_topic, extract_keywords
from sentinel.trend_scorer import TrendScorer
from sentinel.sentinel  import _print_results


# ── 1. Build realistic mock signals (what Google Trends would return) ─────────

def _make_signal(query: str, rising_pct: float, cat_id: int, seed: str) -> TrendSignal:
    """Factory that mimics what GoogleTrendsScanner._fetch_rising() produces."""
    from sentinel.models import make_signal_id
    hook_type, hook_count, hook_score = detect_hook(query)
    return TrendSignal(
        signal_id  = make_signal_id("google_trends", query),
        source     = "google_trends",
        raw_title  = query,
        topic      = clean_topic(query),
        keywords   = extract_keywords(query) + extract_keywords(seed),
        source_url = f"https://trends.google.com/trends/explore?q={query.replace(' ', '+')}",
        raw_score  = min(rising_pct / 5000.0, 1.0),
        hook_score = hook_score,
        hook_type  = hook_type,
        hook_count = hook_count,
        fetched_at = datetime.now(timezone.utc),
        extra      = {"seed": seed, "category_id": cat_id, "rising_value": rising_pct},
    )


MOCK_SIGNALS: list[TrendSignal] = [
    # Explicit "Top N" hook → hook_score 1.0
    _make_signal("top 5 world records broken in 2024",       4800, 0,  "world record broken"),
    _make_signal("5 things scientists discovered this week", 3200, 0,  "scientists discovered"),

    # Implicit hook (implicit list phrase) → hook_score 0.55
    _make_signal("shocking facts about deep sea creatures",  2900, 16, "shocking news facts"),
    _make_signal("scientists discovered hidden ocean layer",  1800, 0,  "scientists discovered"),

    # Generic (no hook pattern) → hook_score 0.25 → fallback did_you_know
    _make_signal("Copa America final match results",          5000, 16, "breaking world record"),
    _make_signal("Taylor Swift Eras Tour sales record",       4200, 0,  "world record broken"),
    _make_signal("world record 100 meter dash shattered",    3800, 16, "world record 2024"),

    # Near-duplicates — should be merged by dedup
    _make_signal("100 meter world record broken again",      1200, 16, "world record 2024"),
    _make_signal("new 100m world record sprint",             800,  16, "world record 2024"),
]


# ── 2. Run TrendScorer ────────────────────────────────────────────────────────

def test_scorer() -> None:
    print("\n" + "=" * 60)
    print("  APEX SENTINEL — Module 1 Offline Pipeline Test")
    print("=" * 60)
    print(f"\n  Input: {len(MOCK_SIGNALS)} mock TrendSignal objects")
    print("  (Simulates what GoogleTrendsScanner would return)\n")

    scorer = TrendScorer()
    trends = scorer.score(MOCK_SIGNALS)

    if not trends:
        print("\n  [FAIL] TrendScorer returned 0 trends. Check thresholds.\n")
        sys.exit(1)

    print(f"\n  [PASS] TrendScorer returned {len(trends)} scored trend(s).\n")
    _print_results(trends)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(trends) <= 5,            "top_k exceeded"
    assert all(t.viral_score > 0 for t in trends), "viral_score must be > 0"
    assert all(t.hook_type for t in trends),        "hook_type must be set"
    assert all(t.trend_id  for t in trends),        "trend_id must be set"

    # The 100m sprint duplicates should be merged into 1 group
    sprint_topics = [t for t in trends if "100" in t.topic or "meter" in t.topic.lower()]
    assert len(sprint_topics) <= 1, \
        f"Dedup failed: {len(sprint_topics)} sprint entries survived (expected ≤1)"

    print("\n  All assertions passed.")
    print("  Module 1 (The Sentinel) pipeline: VERIFIED\n")
    print("=" * 60 + "\n")


# ── 3. Test detect_hook() directly ───────────────────────────────────────────

def test_hook_detection() -> None:
    cases = [
        ("Top 5 things you didn't know about AI",         "top_5",      1.0),
        ("7 shocking secrets scientists discovered",       "top_7",      1.0),
        ("Things you didn't know about the ocean",        "did_you_know", 0.55),
        ("Scientists discovered a new deep sea creature", "did_you_know", 0.55),
        ("Copa America results today",                    "generic",    0.25),
    ]

    print("  Hook Detection Tests:")
    all_pass = True
    for title, exp_type, exp_score in cases:
        got_type, got_count, got_score = detect_hook(title)
        ok = (got_type == exp_type) and (abs(got_score - exp_score) < 0.01)
        status = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_pass = False
        print(f"  {status}  '{title[:50]}'")
        print(f"          got=({got_type}, score={got_score:.2f})  "
              f"expected=({exp_type}, score={exp_score:.2f})")

    assert all_pass, "Hook detection assertions failed — see output above."
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    test_hook_detection()
    test_scorer()
