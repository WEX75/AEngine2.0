"""
tests/test_stream_sentinel_scale.py  —  Pillar 1 scale / false-positive guardrail
Apex Engine 2.0

Formalizes the earlier ad-hoc scale check into a real, deterministic test:
run the detector over a full 10-hour flat-baseline VOD (jittery chat + audio,
no injected spike) and assert the false-positive candidate count stays
within a fixed ceiling.

Deterministic by construction: a FIXED random seed, not a fresh draw each
run — a flaky scale test destroys trust in an autonomous system faster than
a missing one (per CLAUDE.md's testing philosophy).

Run:
    python tests/test_stream_sentinel_scale.py
(also collectable by pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_stream_sentinel import make_chat_log, make_audio_track
from sentinel.stream_moment_detector import StreamMomentDetector

SEED         = 20260708        # fixed — never randomize this per-run
DURATION_SEC = 10 * 60 * 60    # 10 hours

# Starting guardrail, NOT a derived/statistical constant. Earlier ad-hoc runs
# over similar synthetic jitter observed ~2 false-positive candidates across
# a 10h flat baseline; 5 gives headroom above that observation. Its purpose
# is to catch a scorer that regresses into firing constantly (e.g. a broken
# std floor or threshold change), not to certify a real-world false-positive
# rate. Recalibrate against real stream data before treating this ceiling as
# meaningful for anything user-facing — see CLAUDE.md's recalibration note
# on composite_threshold.
MAX_CANDIDATES = 5


def test_scale_flat_baseline_false_positive_ceiling() -> None:
    detector = StreamMomentDetector()

    # No spike injected: spike_start is placed before t=0 with 0 duration,
    # so `spike_start <= t < spike_start + spike_duration` is never true.
    chat = make_chat_log(
        DURATION_SEC, base_rate=5,
        spike_start=-1.0, spike_duration=0.0, spike_rate=5,
        seed=SEED,
    )
    audio = make_audio_track(
        DURATION_SEC, base_dbfs=-32.0,
        spike_start=-1.0, spike_duration=0.0, spike_dbfs=-32.0,
        seed=SEED + 1,
    )

    candidates = detector.detect("vod_scale_flat_001", chat, audio)

    print(f"  10h flat-baseline candidates: {len(candidates)}  "
          f"(guardrail ceiling={MAX_CANDIDATES})")
    for c in candidates:
        print(f"    [{c.window_start:.1f}s → {c.window_end:.1f}s]  "
              f"composite={c.composite_score:.2f}")

    assert len(candidates) <= MAX_CANDIDATES, (
        f"flat-baseline 10h scale run produced {len(candidates)} candidates, "
        f"exceeding the guardrail ceiling of {MAX_CANDIDATES} — investigate "
        f"before assuming this is normal jitter"
    )
    print(f"  [PASS] 10h flat-baseline run stayed within guardrail ceiling "
          f"({len(candidates)} <= {MAX_CANDIDATES})\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX SENTINEL — Pillar 1 Scale Guardrail Test (10h, seeded)")
    print("=" * 60 + "\n")
    test_scale_flat_baseline_false_positive_ceiling()
    print("  Scale guardrail test passed.")
    print("=" * 60 + "\n")
