"""
sentinel/models.py

Shared data models for The Sentinel (Pillar 1).
All scanners and the scorer import from here — no circular deps.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Hook pattern detection ────────────────────────────────────────────────────

# Titles that already contain an explicit "Top N" / "N things" pattern.
# Pattern A: "top/best/worst/most N"       → group(2) holds N
# Pattern B: "N <0-3 words> <plural noun>" → group(3) holds N
#   The plural-noun list is broad enough to catch common viral formats
#   ("7 shocking secrets", "3 dark truths", "5 hidden facts") without
#   false-positives on numbers that appear mid-sentence.
_EXPLICIT_NOUN_LIST = (
    "things?|ways?|reasons?|facts?|secrets?|tips?|signs?|habits?|rules?"
    "|truths?|tricks?|myths?|mistakes?|foods?|places?|people|times?"
    "|lessons?|skills?|methods?|questions?|discoveries|discoveries"
)
_EXPLICIT_NUMBER_RE = re.compile(
    r"\b(top|best|worst|most|least|ranked?)\s+(\d+)\b"
    rf"|\b(\d+)(?:\s+\w+){{0,3}}\s+(?:{_EXPLICIT_NOUN_LIST})\b",
    re.IGNORECASE,
)

# Phrases that imply a list even without an explicit number
_IMPLICIT_LIST_PHRASES: list[str] = [
    "things you didn't know",
    "you won't believe",
    "shocking facts",
    "dark secrets",
    "hidden truth",
    "mind blowing",
    "never knew",
    "scientists discovered",
    "what happens when",
    "terrifying facts",
    "classified secrets",
    "banned from",
    "declassified",
    "nobody talks about",
    "world record",
    "this is why",
]

_HOOK_SCORE_EXPLICIT  = 1.00
_HOOK_SCORE_IMPLICIT  = 0.55
_HOOK_SCORE_GENERIC   = 0.25


def detect_hook(title: str) -> tuple[str, int, float]:
    """
    Inspects a title string for viral "Top N" hook patterns.

    Returns:
        hook_type  – "top_N" | "did_you_know" | "generic"
        hook_count – The N in "Top N" (default 5)
        hook_score – Confidence 0.0–1.0
    """
    m = _EXPLICIT_NUMBER_RE.search(title)
    if m:
        # group(2) → "top/best N"  |  group(3) → "N <plural-noun>"
        n_str = m.group(2) or m.group(3)
        if n_str and n_str.isdigit():
            n = min(int(n_str), 10)
            if n > 1:
                return f"top_{n}", n, _HOOK_SCORE_EXPLICIT

    lower = title.lower()
    for phrase in _IMPLICIT_LIST_PHRASES:
        if phrase in lower:
            return "did_you_know", 5, _HOOK_SCORE_IMPLICIT

    return "generic", 5, _HOOK_SCORE_GENERIC


# ── Text helpers ──────────────────────────────────────────────────────────────

_NOISE_WORDS = re.compile(
    r"\b(TIL|ELI5|AMA|AITA|CMV|OC|PSA|NSFW|UPDATE|EDIT|reddit|r/\w+)\b",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s{2,}")

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "that", "this", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "can", "it", "its", "i", "you", "he", "she",
    "we", "they", "them", "their", "what", "which", "who", "how",
    "when", "where", "why", "not", "no", "so", "if", "up", "out",
    "about", "just", "very",
})


def clean_topic(title: str) -> str:
    """Strip noise from a raw title to produce a clean topic string."""
    cleaned = _NOISE_WORDS.sub("", title)
    cleaned = re.sub(r"[\[\(].*?[\]\)]", "", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip(" :-–—")
    return cleaned[:200]


def extract_keywords(title: str) -> list[str]:
    """Return meaningful keywords from a title, dropping stopwords."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", title.lower())
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            result.append(w)
            seen.add(w)
    return result[:10]


def make_signal_id(source: str, raw_title: str) -> str:
    """Stable 16-char ID for a (source, title) pair."""
    return hashlib.sha1(f"{source}:{raw_title}".encode()).hexdigest()[:16]


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class TrendSignal:
    """
    Raw signal from a single data source, before composite scoring.
    Produced by scanners; consumed by TrendScorer.
    """
    signal_id:  str
    source:     str            # "google_trends" (Reddit: future)
    raw_title:  str
    topic:      str            # Cleaned, noise-stripped topic
    keywords:   list[str]      = field(default_factory=list)
    source_url: str            = ""
    raw_score:  float          = 0.0   # Source-specific velocity (0–1)
    hook_score: float          = 0.0   # "Top N" detectability (0–1)
    hook_type:  str            = "generic"
    hook_count: int            = 5
    fetched_at: datetime       = field(default_factory=lambda: datetime.now(timezone.utc))
    extra:      dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredTrend:
    """
    A fully-scored trend, ready to emit as a TREND_FOUND event.
    Produced by TrendScorer; consumed by The Brain.
    """
    trend_id:              str
    topic:                 str
    hook_type:             str
    hook_count:            int
    viral_score:           float           # 0.0–1.0 composite
    source:                str             # "google_trends" | "combined"
    source_url:            str
    keywords:              list[str]       = field(default_factory=list)
    raw_title:             str             = ""
    scored_at:             datetime        = field(default_factory=lambda: datetime.now(timezone.utc))
    component_scores:      dict[str, float] = field(default_factory=dict)
    contributing_signals:  list[str]       = field(default_factory=list)
