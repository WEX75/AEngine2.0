"""
tests/test_stream_sentinel.py  —  Pillar 1 (Stream Moment Detection) offline test
Apex Engine 2.0

Fully-mocked, offline, deterministic — no live stream, no Ollama, no network
(per CLAUDE.md's testing philosophy). Chat and audio are generated as
time-bucketed data with an injected, known spike. The chat and audio spikes
are deliberately offset by ~1s: real spikes never perfectly align, and
perfectly synced mock data would hide fusion misalignment bugs.

Run:
    python tests/test_stream_sentinel.py
(also collectable by pytest)
"""
from __future__ import annotations

import math
import random
import statistics

# ── Bootstrap: project root must be importable ────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.stream_moment_detector import (
    ChatMessage,
    MomentCandidate,
    StreamMomentDetector,
)

# ── Mock generators ───────────────────────────────────────────────────────────

_CALM_LINES  = ["nice", "hello streamer", "gg", "so true lol", "pog", "what game is this"]
_HYPE_LINES  = ["CLIP IT", "OMEGALUL", "NO WAY", "LMFAOOO", "W W W", "CLIP IT NOW"]


def make_chat_log(
    duration_sec:   int,
    base_rate:      int,
    spike_start:    float,
    spike_duration: float,
    spike_rate:     int,
    seed:           int,
) -> list[ChatMessage]:
    """Per-second chat log: `base_rate` ± jitter msgs/s, `spike_rate` ± jitter
    during the spike, hype-spam text inside the spike window."""
    rng = random.Random(seed)
    messages: list[ChatMessage] = []
    for sec in range(duration_sec):
        in_spike = spike_start <= sec < spike_start + spike_duration
        rate = spike_rate if in_spike else base_rate
        count = max(0, rate + rng.randint(-2, 2))
        lines = _HYPE_LINES if in_spike else _CALM_LINES
        for _ in range(count):
            messages.append(ChatMessage(
                timestamp=sec + rng.random(),
                text=rng.choice(lines),
            ))
    return messages


def make_audio_track(
    duration_sec:   int,
    base_dbfs:      float,
    spike_start:    float,
    spike_duration: float,
    spike_dbfs:     float,
    seed:           int,
    jitter_db:      float = 0.8,
    bucket_sec:     float = 1.0,
) -> list[float]:
    """dBFS array at `bucket_sec` resolution (default 1s, matching the
    detector default): `base_dbfs` ± gaussian jitter, jumping to
    `spike_dbfs` during the spike. Pass a finer bucket_sec (e.g. 0.5) to
    generate audio at a different native resolution than chat."""
    rng = random.Random(seed)
    track: list[float] = []
    n_buckets = int(round(duration_sec / bucket_sec))
    for idx in range(n_buckets):
        t = idx * bucket_sec
        in_spike = spike_start <= t < spike_start + spike_duration
        level = spike_dbfs if in_spike else base_dbfs
        track.append(level + rng.gauss(0.0, jitter_db))
    return track


# ── 1. Positive: injected offset spike is recovered ──────────────────────────

CHAT_SPIKE_START  = 180.0   # chat erupts at t=180s for 8s
CHAT_SPIKE_DUR    = 8.0
AUDIO_SPIKE_START = 181.0   # audio peaks ~1s later for 6s — deliberately offset
AUDIO_SPIKE_DUR   = 6.0
DURATION_SEC      = 600


def test_spike_recovered() -> None:
    detector = StreamMomentDetector()
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

    candidates = detector.detect("vod_test_001", chat, audio)

    print(f"  candidates: {len(candidates)}")
    for c in candidates:
        print(f"    [{c.window_start:.1f}s → {c.window_end:.1f}s]  "
              f"composite={c.composite_score:.2f}  chat_z={c.chat_score:.2f}  "
              f"audio_z={c.audio_score:.2f}  tokens={c.top_spam_tokens}")

    # Exactly one candidate: the offset chat/audio spikes must merge into a
    # single moment, and the 592s of jittery baseline must produce none.
    assert len(candidates) == 1, (
        f"expected exactly 1 merged candidate, got {len(candidates)}"
    )
    c = candidates[0]

    # Window recovered within tolerance of the known injection point.
    expected_start = CHAT_SPIKE_START - detector._pad_before        # ≈ 177
    spike_end      = CHAT_SPIKE_START + CHAT_SPIKE_DUR              # 188
    assert abs(c.window_start - expected_start) <= 2.5, (
        f"window_start {c.window_start:.1f} not within 2.5s of {expected_start:.1f}"
    )
    assert c.window_end >= AUDIO_SPIKE_START + AUDIO_SPIKE_DUR, (
        f"window_end {c.window_end:.1f} cuts off the audio payoff"
    )
    assert c.window_end <= spike_end + detector._pad_after + 2.5, (
        f"window_end {c.window_end:.1f} overshoots the spike"
    )

    assert c.composite_score >= 2.5
    assert c.chat_score > 0 and c.audio_score > 0, (
        "both signals must contribute at the peak — fusion, not one channel"
    )

    hype = {"clip", "omegalul", "w", "lmfaooo", "way", "no", "it", "now"}
    assert hype & set(c.top_spam_tokens), (
        f"top_spam_tokens {c.top_spam_tokens} contain no hype-spam tokens"
    )
    print("  [PASS] offset spike recovered as one merged candidate\n")


# ── 2. Negative: flat baseline → zero candidates, via the zero-variance path ─

def test_flat_baseline_zero_candidates() -> None:
    detector = StreamMomentDetector()

    # Perfectly constant series — every trailing baseline window has
    # EXACTLY zero variance. No jitter, or the test would quietly stop
    # exercising the zero-variance path.
    flat_chat  = [ChatMessage(timestamp=float(s) + 0.5, text="hi")
                  for s in range(DURATION_SEC) for _ in range(4)]
    flat_audio = [-30.0] * DURATION_SEC

    # Prove the input actually is the degenerate case: a naive
    # z = (x - mean) / std over any trailing window divides by zero here.
    trailing_window = flat_audio[100:160]
    assert statistics.pstdev(trailing_window) == 0.0, (
        "mock audio baseline is not actually zero-variance — test is broken"
    )
    try:
        _ = (flat_audio[160] - statistics.mean(trailing_window)) / statistics.pstdev(trailing_window)
        raise AssertionError("naive z-score did NOT divide by zero — flat mock is wrong")
    except ZeroDivisionError:
        pass  # confirmed: this data breaks an unguarded z-score

    # Exercise the detector's z-score path directly on the zero-variance
    # series: every value must come back finite (floored divisor) and 0.0
    # (zero deviation from a flat baseline).
    audio_z = detector.rolling_zscores(flat_audio, detector._audio_std_floor)
    assert all(math.isfinite(z) for z in audio_z), "z-scores must be finite on flat input"
    assert all(z == 0.0 for z in audio_z), "flat signal over flat baseline must score 0"

    chat_counts = [4.0] * DURATION_SEC
    chat_z = detector.rolling_zscores(chat_counts, detector._chat_std_floor)
    assert all(math.isfinite(z) for z in chat_z)
    assert all(z == 0.0 for z in chat_z)

    # And end-to-end: the full pipeline over the same data emits nothing.
    candidates = detector.detect("vod_flat_001", flat_chat, flat_audio)
    assert candidates == [], (
        f"flat baseline produced {len(candidates)} candidate(s) — "
        "scorer fires on noise, which makes it useless"
    )
    print("  [PASS] flat (zero-variance) baseline → finite z-scores, 0 candidates\n")


# ── 3. Zero-variance baseline + real spike must STILL be detected ────────────
# Guards the guard: graceful zero-variance handling must not silently
# swallow the strongest signal there is — a dead chat suddenly exploding.

def test_flat_baseline_spike_still_detected() -> None:
    detector = StreamMomentDetector()
    duration = 300
    spike_start = 120.0

    chat: list[ChatMessage] = []          # dead chat: 0 msgs...
    for sec in range(duration):
        if spike_start <= sec < spike_start + 6.0:      # ...then 20 msg/s
            chat.extend(
                ChatMessage(timestamp=sec + i / 20.0, text="CLIP IT")
                for i in range(20)
            )
    audio = [
        -20.0 if spike_start + 1.0 <= sec < spike_start + 6.0 else -50.0  # ~1s offset
        for sec in range(duration)
    ]

    candidates = detector.detect("vod_dead_chat_001", chat, audio)
    assert len(candidates) == 1, (
        f"spike on a zero-variance baseline must be detected, got {len(candidates)}"
    )
    c = candidates[0]
    assert c.window_start <= spike_start <= c.window_end, (
        f"window [{c.window_start:.1f}, {c.window_end:.1f}] misses spike at {spike_start}"
    )
    assert math.isfinite(c.composite_score)
    print(f"  [PASS] spike on dead-flat baseline detected "
          f"(composite={c.composite_score:.1f}, finite)\n")


# ── 4. Mixed resolution: 1s chat buckets fused with 0.5s audio buckets ───────
# Guards the GRANULARITY fix: the fusion logic must align differing native
# resolutions via nearest-timestamp matching, not assume index i in one
# array is the same instant as index i in the other.

def test_differing_resolution_alignment() -> None:
    detector = StreamMomentDetector()   # bucket_sec=1.0 → chat/output resolution
    audio_bucket_sec = 0.5              # audio sampled twice as fine as chat

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

    print(f"  chat buckets (1.0s): {DURATION_SEC}, audio buckets (0.5s): {len(audio)}")

    candidates = detector.detect(
        "vod_test_mixed_res_001", chat, audio, audio_bucket_sec=audio_bucket_sec
    )

    print(f"  candidates: {len(candidates)}")
    for c in candidates:
        print(f"    [{c.window_start:.1f}s → {c.window_end:.1f}s]  "
              f"composite={c.composite_score:.2f}  chat_z={c.chat_score:.2f}  "
              f"audio_z={c.audio_score:.2f}")

    assert len(candidates) == 1, (
        f"expected exactly 1 merged candidate with mixed-resolution input, "
        f"got {len(candidates)}"
    )
    c = candidates[0]
    assert c.window_start <= CHAT_SPIKE_START <= c.window_end, (
        f"window [{c.window_start:.1f}, {c.window_end:.1f}] misses the "
        f"chat spike at {CHAT_SPIKE_START}"
    )
    assert c.chat_score > 0 and c.audio_score > 0, (
        "both signals must still contribute at the peak with mismatched "
        "resolutions — a broken alignment would zero out or misplace one"
    )
    assert c.composite_score >= 2.5
    print("  [PASS] 1s-chat / 0.5s-audio input correctly aligned and fused\n")


# ── 5. Audio-only detection: chat_weight=0.0 (no chat log — Whop podcasts/webinars) ──
# apex.py's --ingest-vod builds StreamMomentDetector(chat_weight=0.0,
# audio_weight=1.0) when no --chat-log is given (see apex.py's
# _build_ingest_detector()). This is already legal per the detector's own
# validation (chat_weight + audio_weight must sum to 1.0, nothing more) —
# this test proves that path actually produces sensible candidates on a
# real audio spike with ZERO chat messages, not just that construction
# doesn't raise.

def test_audio_only_detection_zero_chat_weight() -> None:
    detector = StreamMomentDetector(chat_weight=0.0, audio_weight=1.0)

    # No chat at all — the empty-file case apex.py's audio-only path feeds
    # StreamIngestHandler.load_chat_log() (an empty chat log file parses to []).
    chat: list[ChatMessage] = []
    audio = make_audio_track(
        DURATION_SEC, base_dbfs=-32.0,
        spike_start=AUDIO_SPIKE_START, spike_duration=AUDIO_SPIKE_DUR,
        spike_dbfs=-12.0, seed=1337,
    )

    candidates = detector.detect("vod_audio_only_001", chat, audio)

    print(f"  candidates: {len(candidates)}")
    for c in candidates:
        print(f"    [{c.window_start:.1f}s → {c.window_end:.1f}s]  "
              f"composite={c.composite_score:.2f}  chat_z={c.chat_score:.2f}  "
              f"audio_z={c.audio_score:.2f}")

    # NOTE (flagged, not "fixed" — see CLAUDE.md's own composite_threshold
    # caveat): with only ONE signal contributing, this same ±0.8dB jitter
    # (identical to test_spike_recovered's audio track) that the fused
    # chat+audio test absorbs without a false positive can, on its own,
    # occasionally clear composite_threshold=2.5 purely from noise — no
    # second independent signal exists here to damp it. That's a real,
    # inherent property of audio-only detection at these defaults, worth
    # knowing about for real Whop podcast/webinar content, not a bug this
    # test should paper over by hand-tuning jitter down until it disappears.
    # So this test asserts the REAL spike is found and scored correctly,
    # not that it's the only candidate.
    matches = [c for c in candidates if c.window_start <= AUDIO_SPIKE_START <= c.window_end]
    assert len(matches) == 1, (
        f"expected exactly one candidate window covering the real spike at "
        f"{AUDIO_SPIKE_START}s, got {len(matches)} (candidates: {candidates})"
    )
    c = matches[0]
    # With zero chat messages, chat_score must be exactly 0 (no signal to
    # z-score) — and, critically, that must NOT suppress the composite
    # score, since chat_weight=0.0 means chat contributes nothing to it
    # either way. audio_score alone must clear the threshold.
    assert c.chat_score == 0.0, f"expected chat_score==0.0 with zero chat messages, got {c.chat_score}"
    assert c.audio_score > 0, "audio signal must still contribute with chat_weight=0.0"
    assert c.composite_score == c.audio_score, (
        f"composite_score ({c.composite_score:.2f}) must equal audio_score "
        f"({c.audio_score:.2f}) alone when chat_weight=0.0 — got a chat contribution that shouldn't exist"
    )
    assert c.composite_score >= 2.5
    # The real, deliberate spike_dbfs=-12.0 vs base_dbfs=-32.0 jump must
    # score decisively higher than jitter noise could plausibly produce —
    # proving this isn't itself a borderline/noise-driven detection.
    assert c.composite_score > 10.0, (
        f"real spike's composite_score ({c.composite_score:.2f}) should be far above "
        "threshold, clearly distinguishing it from any noise-driven candidate"
    )
    print("  [PASS] audio-only (chat_weight=0.0) detection recovers the spike from audio alone\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX SENTINEL — Pillar 1 Stream Moment Detection Test")
    print("=" * 60 + "\n")
    test_spike_recovered()
    test_flat_baseline_zero_candidates()
    test_flat_baseline_spike_still_detected()
    test_differing_resolution_alignment()
    test_audio_only_detection_zero_chat_weight()
    print("  All Pillar 1 detection tests passed.")
    print("=" * 60 + "\n")
