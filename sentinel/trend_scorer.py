"""
sentinel/trend_scorer.py  —  The Sentinel: Trend Scorer
Apex Engine 2.0  |  Pillar 1

Receives a flat list of TrendSignal objects from all scanners,
deduplicates them by semantic similarity, and computes a unified
viral_score for each unique topic.

Scoring formula (weights live in settings.yaml):
    viral_score = (gt_raw_score × gt_weight)
                + (hook_score   × hook_weight)
                + source_bonus           ← reserved for when Reddit re-joins

Output: a list of ScoredTrend objects, sorted best-first, length ≤ top_k.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from rapidfuzz import fuzz  # type: ignore[import]

from core.config_loader import get_config
from core.logger import get_logger
from sentinel.models import ScoredTrend, TrendSignal

log = get_logger(__name__)


class TrendScorer:
    """
    Merges, deduplicates, and scores all raw signals from The Sentinel's
    scanner(s) into a ranked shortlist of viral-ready trends.
    """

    def __init__(self) -> None:
        cfg = get_config()
        s   = cfg["sentinel"]

        scorer_cfg = s["scorer"]
        gt_cfg     = s["google_trends"]

        self._dedup_threshold:    float = scorer_cfg.get("dedup_threshold",       0.75)
        self._min_hook_conf:      float = scorer_cfg.get("min_hook_confidence",   0.10)
        self._hook_weight:        float = scorer_cfg.get("hook_detection_weight", 0.30)
        self._gt_weight:          float = gt_cfg.get("weight", 0.70)  # 100% of signal weight

        self._top_k:              int   = s.get("top_k_trends",          5)
        self._min_viral_score:    float = s.get("viral_score_threshold", 0.25)

    # ── Public ────────────────────────────────────────────────────────────────

    def score(self, signals: list[TrendSignal]) -> list[ScoredTrend]:
        """
        Full scoring pipeline.

        Args:
            signals: All TrendSignal objects from this scan cycle.

        Returns:
            Top-K ScoredTrend objects above the viral_score threshold,
            sorted best-first.
        """
        if not signals:
            log.warning("[TrendScorer] No signals to score.")
            return []

        log.info(f"[TrendScorer] Scoring {len(signals)} raw signals...")

        # 1. Semantic deduplication
        groups = self._deduplicate(signals)
        log.info(f"[TrendScorer] {len(groups)} unique topic group(s) after dedup.")

        # 2. Score each group
        scored: list[ScoredTrend] = []
        for canonical, group in groups.items():
            result = self._score_group(canonical, group)
            if result is not None:
                scored.append(result)

        # 3. Filter by minimum viral_score
        qualified = [t for t in scored if t.viral_score >= self._min_viral_score]
        log.info(
            f"[TrendScorer] {len(qualified)}/{len(scored)} groups passed "
            f"viral_score threshold ({self._min_viral_score})."
        )

        # 4. Sort and trim to top-K
        top = sorted(qualified, key=lambda t: t.viral_score, reverse=True)[: self._top_k]

        for rank, t in enumerate(top, 1):
            log.info(
                f"[TrendScorer] #{rank} | score={t.viral_score:.3f} | "
                f"hook={t.hook_type}({t.hook_count}) | topic='{t.topic[:55]}'"
            )

        return top

    # ── Private ───────────────────────────────────────────────────────────────

    def _deduplicate(
        self, signals: list[TrendSignal]
    ) -> dict[str, list[TrendSignal]]:
        """
        Groups signals about the same topic using fuzzy token-sort matching.
        O(n²) but signal lists are small (<100 per cycle) — no issue.
        """
        groups:           dict[str, list[TrendSignal]] = {}
        canonical_topics: list[str]                    = []

        for sig in signals:
            topic = sig.topic.lower().strip()
            if not topic:
                continue

            matched: Optional[str] = None
            for ct in canonical_topics:
                if (fuzz.token_sort_ratio(topic, ct) / 100.0) >= self._dedup_threshold:
                    matched = ct
                    break

            if matched is not None:
                groups[matched].append(sig)
            else:
                canonical_topics.append(topic)
                groups[topic] = [sig]

        return groups

    def _score_group(
        self, canonical: str, signals: list[TrendSignal]
    ) -> Optional[ScoredTrend]:
        """
        Computes a ScoredTrend for one deduplicated topic group.

        Hook detection: if no explicit hook is found, we assign 'did_you_know'
        as a fallback — the Brain will shape the final hook structure.
        """
        # ── Best raw score from any signal in the group ───────────────────────
        gt_score = max(s.raw_score for s in signals)

        # ── Best hook signal in the group ─────────────────────────────────────
        best_hook = max(signals, key=lambda s: s.hook_score)
        hook_score = best_hook.hook_score
        hook_type  = best_hook.hook_type
        hook_count = best_hook.hook_count

        # Fallback: assign a generic hook if none detected.
        # The Brain handles hook crafting; the Sentinel just needs a starting point.
        if hook_score < self._min_hook_conf:
            hook_type  = "did_you_know"
            hook_count = 5
            hook_score = self._min_hook_conf   # floor so the group survives

        # ── Composite viral_score ─────────────────────────────────────────────
        raw = (
            gt_score   * self._gt_weight
            + hook_score * self._hook_weight
        )
        viral_score = round(min(raw, 1.0), 4)

        # ── Representative signal (highest raw_score) ─────────────────────────
        best = max(signals, key=lambda s: s.raw_score)

        # ── Merged keyword list (deduplicated) ────────────────────────────────
        seen_kw:  set[str]  = set()
        keywords: list[str] = []
        for sig in signals:
            for kw in sig.keywords:
                if kw not in seen_kw:
                    keywords.append(kw)
                    seen_kw.add(kw)

        trend_id = hashlib.sha1(canonical.encode()).hexdigest()[:16]

        return ScoredTrend(
            trend_id             = trend_id,
            topic                = best.topic or canonical,
            raw_title            = best.raw_title,
            hook_type            = hook_type,
            hook_count           = hook_count,
            viral_score          = viral_score,
            source               = "google_trends",
            source_url           = best.source_url,
            keywords             = keywords[:15],
            component_scores     = {
                "gt_score":    round(gt_score, 4),
                "hook_score":  round(hook_score, 4),
                "gt_weight":   self._gt_weight,
                "hook_weight": self._hook_weight,
            },
            contributing_signals = [s.signal_id for s in signals],
        )
