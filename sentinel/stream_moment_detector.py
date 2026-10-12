"""
sentinel/stream_moment_detector.py  —  Pillar 1: Stream Moment Detection
Apex Engine 2.0

Fuses chat-rate and audio-loudness spikes into candidate clip windows.
CPU-only, pure stdlib. Ingests a chat log and a bucketed audio-dBFS track
*only* — never the full VOD (per CLAUDE.md, Pillar 1).

Scoring model
-------------
Each signal is scored as a *rolling z-score against a trailing baseline of
that same stream* — never a fixed global threshold, because hype baselines
vary wildly per streamer (a 5 msg/s burst is an event in a sleepy stream
and background noise in a big one).

    composite[i] = 0.6 * chat_z[i] + 0.4 * audio_z[i]

Buckets whose composite crosses the threshold become windows, padded by
(pad_before, pad_after), and overlapping/near-adjacent windows are merged
before candidates are emitted.

Zero-variance baselines
-----------------------
A perfectly flat trailing baseline (dead chat, constant loudness) has zero
standard deviation, which breaks the naive z = (x - mean) / std. We handle
this with a per-signal *floor* on the standard deviation rather than a
skip-guard, because the two failure modes pull in opposite directions:

* flat baseline, flat signal   → (x - mean) = 0  → z = 0 / floor = 0
  (no candidate — silence over silence is not a moment)
* flat baseline, genuine spike → z = (x - mean) / floor — large but FINITE
  (a dead chat suddenly exploding is the strongest signal there is;
  a guard that returned 0 on zero variance would silently swallow it)

The floor also damps overreaction on *near*-flat baselines, where a tiny
std would otherwise turn ±1-message jitter into a multi-sigma "spike".

Warm-up blind spot
------------------
The first `min_baseline_sec` (default 30s) of every detect() call has no
trailing baseline and is explicitly excluded from scoring — see the
WARM-UP note on detect() for exact behavior and the chunked-VOD caveat.

Mixed-resolution fusion
------------------------
chat_messages are always bucketed at the detector's own bucket_sec.
audio_dbfs may be sampled at a different native resolution (audio_bucket_sec)
and is aligned onto the chat/output timeline via nearest-timestamp matching,
not index-for-index zipping — see detect()'s GRANULARITY note.

Third fusion signal: subtitle-based semantic scoring
-----------------------------------------------------
Chat and audio can both be weak for whole content archetypes, so a subtitle-
derived signal supports content without a chat channel or strong audio. Adding
a third signal derived from ingested subtitles/captions lets content like
this still be detected, via keyword density rather than chat/audio energy:

    composite[i] = chat_weight*chat_z[i] + audio_weight*audio_z[i]
                 + semantic_weight*semantic_z[i]

Keyword categories are deliberately split into two weighted groups instead
of one flat list:

  * strong_keywords  — rare, domain-specific, high-signal terms
                       ("headshot", "pentakill", "clutch"). Given
                       strong_keyword_weight (default 3.0) per hit.
  * generic_keywords — common hype filler ("insane", "literally", "crazy").
                       Given generic_keyword_weight (default 1.0) per hit.

Mixing these into one flat list lets common filler dominate the raw hit
count and drown the rare, meaningful signal — a clip with ten "insane"s and
zero "pentakill"s would outscore one with a single real "pentakill" callout
under a flat count, which is backwards. Splitting the categories and
weighting them differently keeps the rare/strong signal from being diluted.

Subtitle cues are converted into a per-bucket semantic score by:
  1. scoring each cue's cleaned text via _score_subtitle_text() (keyword
     hits, weighted by category)
  2. assigning that score to the single output bucket nearest the cue's
     MIDPOINT time

Step 2 deliberately reuses the same nearest-timestamp *principle* already
established for audio/chat fusion (see GRANULARITY above and
_align_nearest below), rather than reinventing a new alignment scheme —
but applied at (cue -> single bucket) granularity instead of
(fine-array -> coarse-array) resampling, because subtitle cues have no
fixed native sample rate the way audio does (cue durations vary wildly:
sub-second word timings up to multi-second pauses). Assigning a cue's full
score to every bucket its span overlaps would also over-weight long cues
relative to short ones purely because they touch more buckets; midpoint
assignment avoids that without needing a second resampling pass.

Zero-variance/std-floor handling for this signal reuses rolling_zscores()
exactly as chat/audio do (semantic_std_floor plays the same role as
chat_std_floor/audio_std_floor_db) — see "Zero-variance baselines" above.

Rectified audio for calm-delivery archetypes (audio_z_floor)
------------------------------------------------------------
Fusing SIGNED audio z-scores assumes below-baseline loudness is counter-
evidence of a moment. For chat-having (gaming) content that assumption
holds — a loudness drop coincident with a chat spike is legitimately
suspicious (bot spam, off-screen chat activity). For interview/podcast
content it can be empirically wrong: striking statements delivered in a
calm register may have below-baseline loudness, even when subtitles provide
a strong semantic signal. Rectifying audio lets it corroborate a moment
without vetoing one solely because delivery is calm.

audio_z_floor (default None = signed, unchanged behavior) rectifies the
audio z-scores at that floor before fusion: audio_z_eff = max(audio_z,
floor). With floor 0.0 (what the interview profile sets in settings.yaml),
audio can still push a marginal semantic moment over the threshold but can
never veto one. This was chosen over the two alternatives against that
same real data:

  * audio_weight=0 — throws away the observed real corroboration (t≈70s/
    81s both had audio_z > +1.4 coincident with keyword hits; dropping
    audio entirely lowers the VOD's strongest moment from 1.85 to 1.35).
  * keeping signed fusion — keeps penalizing calm delivery, which for this
    archetype is normal speaking register, not evidence against a moment.

The floor is deliberately NOT applied to chat_z or semantic_z, and not
enabled by default: it encodes an archetype-specific asymmetry in what
negative audio *means*, not a general "z-scores shouldn't go negative"
rule. MomentCandidate.audio_score reports the RECTIFIED value (the one
actually used in fusion), so composite_score always equals the weighted
sum of the reported per-signal scores.

Weight redistribution for missing signals (chat-less or subtitle-less
content) is NOT performed implicitly inside this class at detect() time —
see redistribute_weights() below for the explicit, standalone formula, and
sentinel/stream_ingest.py for the intake-time policy on what happens when
BOTH chat and subtitles are missing (CLAUDE.md Pillar 1: rejected, not
silently degraded to a known-inadequate audio-only signal set).

Output
------
MomentCandidate mirrors the MOMENT_DETECTED event contract field-for-field
so later EventBus wiring is a pure conversion. This module deliberately
does NOT import the EventBus or JobQueue.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

# ── Data contracts ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChatMessage:
    """One chat-log line: seconds from stream start + raw text."""
    timestamp: float
    text: str


@dataclass(frozen=True)
class SubtitleCue:
    """
    One deduplicated subtitle/caption cue: a text chunk plus the [start, end)
    span it appeared over. Produced by sentinel.stream_ingest.load_subtitle_track
    from a .vtt/.srt file — that loader is responsible for collapsing
    real-world auto-caption messiness (rolling/duplicate text, inline
    timing tags) down to one SubtitleCue per real new chunk of speech, so
    by the time a SubtitleCue reaches this module its `text` is meant to be
    scored exactly once, not deduplicated again.
    """
    start: float
    end:   float
    text:  str


@dataclass(frozen=True)
class MomentCandidate:
    """
    One merged high-score window. Field names match the MOMENT_DETECTED
    event contract in CLAUDE.md so conversion later is mechanical.
    """
    vod_id:           str
    window_start:     float
    window_end:       float
    composite_score:  float          # peak composite z inside the window
    chat_score:       float          # chat z at the peak bucket
    audio_score:      float          # audio z at the peak bucket
    semantic_score:   float = 0.0    # subtitle-keyword z at the peak bucket (0.0 when semantic_weight=0)
    top_spam_tokens:  list[str] = field(default_factory=list)
    detection_method: str = "chat_audio_zscore_fusion"


_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def redistribute_weights(
    chat_weight:     float,
    audio_weight:    float,
    semantic_weight: float,
    *,
    has_chat:        bool,
    has_audio:       bool,
    has_semantic:    bool,
) -> tuple[float, float, float]:
    """
    Exact redistribution formula for missing signals (CLAUDE.md Pillar 1):
    zero out the configured weight for every signal that isn't actually
    active for this particular VOD, then RESCALE the remaining active
    weights so they sum back to 1.0 — their relative ratio to each other is
    preserved, this is a rescale, not a re-derivation from scratch.

        active_i = configured_weight_i if has_signal_i else 0.0
        result_i = active_i / sum(active)

    Raises ValueError if the result would be undefined — i.e. no signal is
    both present AND configured with a nonzero weight, so there is nothing
    left to redistribute onto. Callers (see sentinel/stream_ingest.py) are
    expected to apply the separate, stricter intake policy — reject rather
    than redistribute — for the specific case CLAUDE.md calls out: no chat
    AND no subtitles, which this function alone would otherwise silently
    "solve" by dumping 100% of the weight onto audio, reproducing the exact
    known-inadequate audio-only failure this intake policy prevents.
    This function only does the arithmetic; it does not encode that policy.

    Pure function, no I/O — trivially unit-testable in isolation.
    """
    raw = {
        "chat":     chat_weight if has_chat else 0.0,
        "audio":    audio_weight if has_audio else 0.0,
        "semantic": semantic_weight if has_semantic else 0.0,
    }
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError(
            "redistribute_weights: no active signal has a nonzero configured "
            "weight — nothing to redistribute onto (all signals missing, or "
            "every active signal is configured at weight 0.0)"
        )
    return raw["chat"] / total, raw["audio"] / total, raw["semantic"] / total


class StreamMomentDetector:
    """
    Detects hype moments in one stream from its chat log + audio track.

    Parameter defaults
    ------------------
    bucket_sec=1.0            1s resolution — chat rate is meaningful per
                              second and the arrays stay tiny (a 10h VOD is
                              36k buckets).
    baseline_window_sec=60.0  Trailing baseline ~1 min: long enough to be
                              stable, short enough to track a stream whose
                              energy drifts across an evening.
    min_baseline_sec=30.0     Warm-up: no scoring until at least this much
                              trailing history exists (z is meaningless
                              against a 3-sample baseline).
    chat_weight=0.6 / audio_weight=0.4 / semantic_weight=0.0   Per CLAUDE.md
                              Pillar 1. semantic_weight defaults to 0.0 so
                              constructing a detector with no subtitle-related
                              args at all reproduces the original two-signal
                              fusion exactly (backward compatible).
    composite_threshold=2.5   ~2.5 sigma joint excursion.
    pad_before=3.0            Capture the setup before chat physically
                              reacts (typing latency means chat spikes
                              *after* the moment).
    pad_after=5.0             Capture the payoff/reaction tail.
    merge_gap=2.0             Windows closer than this are one moment.
    chat_std_floor=0.75       Std floor, messages/bucket. Zero-variance
                              guard + jitter damper (±1-2 msg noise on a
                              quiet chat must not read as multi-sigma).
    audio_std_floor_db=0.5    Std floor, dB. Same role for loudness.
    semantic_std_floor=0.5    Std floor, keyword-score units. Same
                              zero-variance-guard role for the subtitle
                              signal (see module docstring's "Third fusion
                              signal" section).
    strong_keywords/generic_keywords   Two weighted keyword categories —
                              see module docstring for why they're split
                              instead of one flat list. Both configurable
                              via settings.yaml (sentinel.stream_moment_detection
                              .semantic_keywords), not hardcoded per-instance.
    strong_keyword_weight=3.0 / generic_keyword_weight=1.0   Per-hit score
                              for each category. strong > generic by design
                              so a handful of rare/domain callouts aren't
                              drowned out by common hype filler.
    audio_z_floor=None        None = signed audio fusion (unchanged default).
                              A float (the interview profile uses 0.0)
                              rectifies audio_z at that floor before fusion
                              so calm delivery can't veto a semantic moment —
                              see the module docstring's "Rectified audio"
                              section for the real-data justification.
    """

    def __init__(
        self,
        bucket_sec:              float = 1.0,
        baseline_window_sec:     float = 60.0,
        min_baseline_sec:        float = 30.0,
        chat_weight:             float = 0.6,
        audio_weight:            float = 0.4,
        semantic_weight:         float = 0.0,
        composite_threshold:     float = 2.5,
        pad_before:              float = 3.0,
        pad_after:               float = 5.0,
        merge_gap:               float = 2.0,
        chat_std_floor:          float = 0.75,
        audio_std_floor_db:      float = 0.5,
        semantic_std_floor:      float = 0.5,
        strong_keywords:         Sequence[str] = (),
        generic_keywords:        Sequence[str] = (),
        strong_keyword_weight:   float = 3.0,
        generic_keyword_weight:  float = 1.0,
        audio_z_floor:           float | None = None,
        top_tokens_k:            int   = 5,
    ) -> None:
        if bucket_sec <= 0:
            raise ValueError("bucket_sec must be > 0")
        if not math.isclose(chat_weight + audio_weight + semantic_weight, 1.0, abs_tol=1e-9):
            raise ValueError("chat_weight + audio_weight + semantic_weight must sum to 1.0")
        if chat_std_floor <= 0 or audio_std_floor_db <= 0 or semantic_std_floor <= 0:
            raise ValueError("std floors must be > 0 (they are the zero-variance guard)")

        self._bucket_sec         = bucket_sec
        self._baseline_window_sec = baseline_window_sec
        self._min_baseline_sec   = min_baseline_sec
        self._chat_weight        = chat_weight
        self._audio_weight       = audio_weight
        self._semantic_weight    = semantic_weight
        self._threshold          = composite_threshold
        self._pad_before         = pad_before
        self._pad_after          = pad_after
        self._merge_gap          = merge_gap
        self._chat_std_floor     = chat_std_floor
        self._audio_std_floor    = audio_std_floor_db
        self._semantic_std_floor = semantic_std_floor
        self._audio_z_floor      = audio_z_floor
        self._top_tokens_k       = top_tokens_k

        # Single-word keywords are matched via O(1) set membership against
        # tokens; multi-word phrases (e.g. "no way") can't be, so they're
        # kept separate and matched via substring containment instead.
        self._strong_word_set, self._strong_phrases = self._split_keywords(strong_keywords)
        self._generic_word_set, self._generic_phrases = self._split_keywords(generic_keywords)
        self._strong_keyword_weight  = strong_keyword_weight
        self._generic_keyword_weight = generic_keyword_weight

    @staticmethod
    def _split_keywords(keywords: Sequence[str]) -> tuple[set[str], list[str]]:
        words: set[str] = set()
        phrases: list[str] = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if not kw_lower:
                continue
            if " " in kw_lower:
                phrases.append(kw_lower)
            else:
                words.add(kw_lower)
        return words, phrases

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def warmup_sec(self) -> float:
        """
        Seconds of trailing history required before scoring begins
        (= min_baseline_sec at construction). Buckets inside this window at
        the start of every detect() call are forced to composite score 0.0
        and can NEVER become a candidate, regardless of what actually
        happened in the audio/chat during that window. See the WARM-UP note
        on detect() for why, and for the chunked-VOD implication.
        """
        return self._min_baseline_sec

    def detect(
        self,
        vod_id:            str,
        chat_messages:     Sequence[ChatMessage],
        audio_dbfs:        Sequence[float],
        audio_bucket_sec:  float | None = None,
        subtitle_cues:     Sequence[SubtitleCue] = (),
    ) -> list[MomentCandidate]:
        """
        Score one stream and return merged candidate windows, in time order.

        audio_dbfs defines the timeline length: it covers
        len(audio_dbfs) * audio_bucket_sec seconds. Chat messages outside
        that range are ignored. Output windows (and chat bucketing) are
        always expressed at the detector's own `bucket_sec` resolution.

        GRANULARITY: audio_bucket_sec lets audio_dbfs be sampled at a
        *different* native resolution than bucket_sec (e.g. 0.5s audio
        buckets against a detector built with bucket_sec=1.0 for chat/
        output). Defaults to bucket_sec (matching resolution) for backward
        compatibility. When resolutions differ, audio is z-scored at its
        own native resolution first (so its trailing-baseline window
        reflects its real sample rate), then resampled onto the chat/
        output timeline by *nearest-timestamp matching* — never by zipping
        arrays index-for-index, which silently misaligns signals whenever
        the two resolutions don't match.

        WARM-UP (first `warmup_sec` seconds, default 30s, of THIS call):
        there is no trailing baseline yet, so every bucket in that window
        is forced to composite score 0.0 — explicitly excluded from the
        hot-bucket search below, not merely "unlikely to cross a positive
        threshold". A genuine spike that falls entirely inside the warm-up
        window will NOT be detected. This is a per-call blind spot: it
        re-triggers at the start of every detect() invocation. If a VOD is
        ever processed in fixed-size chunks rather than one full pass over
        the whole chat/audio track, each chunk boundary reintroduces this
        30s blind spot unless the caller overlaps chunks by at least
        `warmup_sec` so the warm-up region of chunk N+1 is covered by real
        trailing history carried over from the tail of chunk N.
        """
        audio_bucket_sec = self._bucket_sec if audio_bucket_sec is None else audio_bucket_sec
        if audio_bucket_sec <= 0:
            raise ValueError("audio_bucket_sec must be > 0")

        n_audio_buckets = len(audio_dbfs)
        if n_audio_buckets == 0:
            return []

        duration_sec = n_audio_buckets * audio_bucket_sec
        n_buckets = max(1, math.ceil(duration_sec / self._bucket_sec - 1e-9))

        chat_counts = self._bucket_chat(chat_messages, n_buckets)
        chat_z = self.rolling_zscores(chat_counts, self._chat_std_floor)

        audio_z_native = self.rolling_zscores(
            list(audio_dbfs), self._audio_std_floor, audio_bucket_sec
        )
        audio_z = self._align_nearest(
            audio_z_native, audio_bucket_sec, n_buckets, self._bucket_sec
        )
        if self._audio_z_floor is not None:
            # Rectified audio (see module docstring): below-floor loudness
            # is normal register for this archetype, not counter-evidence.
            audio_z = [max(z, self._audio_z_floor) for z in audio_z]

        semantic_counts = self._bucket_subtitles(subtitle_cues, n_buckets)
        semantic_z = self.rolling_zscores(semantic_counts, self._semantic_std_floor)

        composite = [
            self._chat_weight * c + self._audio_weight * a + self._semantic_weight * s
            for c, a, s in zip(chat_z, audio_z, semantic_z)
        ]

        # Explicit warm-up exclusion — independent of threshold sign, so it
        # can't be accidentally bypassed by a non-positive composite_threshold.
        warmup_buckets = max(1, round(self._min_baseline_sec / self._bucket_sec))
        hot = [
            i for i, score in enumerate(composite)
            if i >= warmup_buckets and score >= self._threshold
        ]
        if not hot:
            return []

        return [
            self._build_candidate(vod_id, bucket_group, composite, chat_z,
                                  audio_z, semantic_z, chat_messages, n_buckets)
            for bucket_group in self._merge_hot_buckets(hot)
        ]

    def rolling_zscores(
        self,
        series:      Sequence[float],
        std_floor:   float,
        bucket_sec:  float | None = None,
    ) -> list[float]:
        """
        z-score of each value against the trailing baseline window
        (previous `baseline_window_sec` of buckets, current bucket excluded).

        Public so tests can exercise the zero-variance path directly: the
        divisor is max(trailing_std, std_floor), so a zero-variance baseline
        divides by std_floor — every returned value is finite by
        construction. Buckets with under `min_baseline_sec` of history
        score 0.0 (warm-up — see detect()'s WARM-UP note for the exact
        blind-spot behavior this produces end-to-end).

        bucket_sec: the resolution of `series`, defaults to the detector's
        own bucket_sec. Pass a different value to z-score a series sampled
        at a different native resolution (e.g. audio at 0.5s) — the
        baseline_window_sec / min_baseline_sec windows are recomputed in
        THIS resolution's buckets so the trailing-history semantics (a
        fixed number of *seconds*, not buckets) stay correct regardless of
        sample rate.

        O(n) via running sum / sum-of-squares.
        """
        bucket_sec = self._bucket_sec if bucket_sec is None else bucket_sec
        window       = max(1, int(round(self._baseline_window_sec / bucket_sec)))
        min_baseline = max(1, int(round(self._min_baseline_sec / bucket_sec)))

        n = len(series)
        zscores: list[float] = [0.0] * n

        running_sum   = 0.0
        running_sumsq = 0.0
        count = 0

        for i in range(n):
            if count >= min_baseline:
                mean = running_sum / count
                # Float error can push this a hair negative on flat data —
                # exactly the zero-variance case — so clamp before sqrt.
                variance = max(running_sumsq / count - mean * mean, 0.0)
                std = max(math.sqrt(variance), std_floor)
                zscores[i] = (series[i] - mean) / std

            value = series[i]
            running_sum   += value
            running_sumsq += value * value
            count += 1
            if count > window:
                oldest = series[i - window]
                running_sum   -= oldest
                running_sumsq -= oldest * oldest
                count -= 1

        return zscores

    def _align_nearest(
        self,
        source:             Sequence[float],
        source_bucket_sec:  float,
        n_target:           int,
        target_bucket_sec:  float,
    ) -> list[float]:
        """
        Resample `source` (sampled every source_bucket_sec) onto a timeline
        of n_target buckets (each target_bucket_sec wide) via nearest-
        timestamp matching: each target bucket's center time is matched to
        the source bucket whose center time is closest.

        This is the GRANULARITY fix — fusing two signals sampled at
        different resolutions must never assume index i in one array is
        the same instant as index i in the other. When the resolutions
        already match, this is an identity mapping (target bucket i's
        center falls exactly on source bucket i's center).
        """
        n_source = len(source)
        if n_source == 0:
            return [0.0] * n_target

        aligned: list[float] = [0.0] * n_target
        for i in range(n_target):
            target_center = (i + 0.5) * target_bucket_sec
            approx = int(target_center / source_bucket_sec - 0.5)
            approx = min(max(approx, 0), n_source - 1)

            best_j, best_dist = approx, math.inf
            for cand in (approx - 1, approx, approx + 1):
                if 0 <= cand < n_source:
                    dist = abs((cand + 0.5) * source_bucket_sec - target_center)
                    if dist < best_dist:
                        best_j, best_dist = cand, dist
            aligned[i] = source[best_j]

        return aligned

    # ── Private ───────────────────────────────────────────────────────────────

    def _bucket_chat(
        self, messages: Sequence[ChatMessage], n_buckets: int
    ) -> list[float]:
        counts = [0.0] * n_buckets
        for msg in messages:
            idx = int(msg.timestamp // self._bucket_sec)
            if 0 <= idx < n_buckets:
                counts[idx] += 1.0
        return counts

    def _score_subtitle_text(self, text: str) -> float:
        """
        Keyword-hit score for one cue's text: strong_keyword_weight per
        strong-category hit + generic_keyword_weight per generic-category
        hit. See module docstring for why the two categories are weighted
        differently instead of using one flat keyword list.
        """
        text_lower = text.lower()
        tokens = _TOKEN_RE.findall(text_lower)
        token_set = set(tokens)

        score = 0.0
        for token in tokens:
            if token in self._strong_word_set:
                score += self._strong_keyword_weight
            elif token in self._generic_word_set:
                score += self._generic_keyword_weight

        for phrase in self._strong_phrases:
            if phrase in text_lower:
                score += self._strong_keyword_weight
        for phrase in self._generic_phrases:
            if phrase in text_lower:
                score += self._generic_keyword_weight

        return score

    def _bucket_subtitles(
        self, cues: Sequence[SubtitleCue], n_buckets: int
    ) -> list[float]:
        """
        Assigns each cue's keyword-hit score to the single output bucket
        nearest the cue's MIDPOINT time. See module docstring's "Third
        fusion signal" section for why this reuses the nearest-timestamp
        principle at (cue -> bucket) granularity rather than an
        overlap-onto-every-touched-bucket scheme.
        """
        scores = [0.0] * n_buckets
        for cue in cues:
            cue_score = self._score_subtitle_text(cue.text)
            if cue_score == 0.0:
                continue
            midpoint = (cue.start + cue.end) / 2.0
            idx = int(midpoint // self._bucket_sec)
            idx = min(max(idx, 0), n_buckets - 1)
            scores[idx] += cue_score
        return scores

    def _merge_hot_buckets(self, hot: list[int]) -> list[list[int]]:
        """Group hot bucket indices whose padded windows overlap or sit
        within merge_gap of each other."""
        # Two hot buckets i < j merge when j's padded start reaches back to
        # within merge_gap of i's padded end:
        #   (j*b - pad_before) - ((i+1)*b + pad_after) <= merge_gap
        max_bucket_gap = int(
            (self._pad_before + self._pad_after + self._merge_gap)
            // self._bucket_sec
        ) + 1

        groups: list[list[int]] = [[hot[0]]]
        for idx in hot[1:]:
            if idx - groups[-1][-1] <= max_bucket_gap:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        return groups

    def _build_candidate(
        self,
        vod_id:        str,
        bucket_group:  list[int],
        composite:     list[float],
        chat_z:        list[float],
        audio_z:       list[float],
        semantic_z:    list[float],
        chat_messages: Sequence[ChatMessage],
        n_buckets:     int,
    ) -> MomentCandidate:
        b = self._bucket_sec
        start = max(0.0, bucket_group[0] * b - self._pad_before)
        end   = min(n_buckets * b, (bucket_group[-1] + 1) * b + self._pad_after)

        peak = max(bucket_group, key=lambda i: composite[i])

        return MomentCandidate(
            vod_id           = vod_id,
            window_start     = start,
            window_end       = end,
            composite_score  = composite[peak],
            chat_score       = chat_z[peak],
            audio_score      = audio_z[peak],
            semantic_score   = semantic_z[peak],
            top_spam_tokens  = self._top_spam_tokens(chat_messages, start, end),
            detection_method = (
                "chat_audio_semantic_zscore_fusion"
                if self._semantic_weight > 0.0
                else "chat_audio_zscore_fusion"
            ),
        )

    def _top_spam_tokens(
        self, messages: Sequence[ChatMessage], start: float, end: float
    ) -> list[str]:
        counter: Counter[str] = Counter()
        for msg in messages:
            if start <= msg.timestamp < end:
                counter.update(t.lower() for t in _TOKEN_RE.findall(msg.text))
        return [token for token, _count in counter.most_common(self._top_tokens_k)]
