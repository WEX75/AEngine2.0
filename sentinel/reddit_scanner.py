"""
sentinel/reddit_scanner.py

Scans configured subreddits for rising posts that fit a "Top N" viral format.

Returns a list of TrendSignal dataclasses, each with a raw velocity score.
The sentinel.trend_scorer module combines these with Google Trends data.

Requires in .env:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.config_loader import get_config
from core.logger import get_logger

log = get_logger(__name__)

# ── Data contract shared with trend_scorer.py ────────────────────────────────
@dataclass
class TrendSignal:
    """Raw signal from a single source before scoring."""
    signal_id: str             # SHA1 of (source + raw_title)
    source: str                # "reddit" | "google_trends"
    raw_title: str             # Original headline
    topic: str                 # Cleaned, normalised topic
    keywords: list[str] = field(default_factory=list)
    source_url: str = ""
    raw_score: float = 0.0     # Source-specific velocity metric (0.0 – 1.0)
    hook_score: float = 0.0    # How well this fits a "Top N" format (0.0 – 1.0)
    hook_type: str = ""        # "top_5" | "top_4" | "top_3" | "did_you_know" | "generic"
    hook_count: int = 5        # The N in "Top N"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)  # Source-specific metadata


# ── Hook pattern detection (used here and in trend_scorer) ───────────────────

# Explicit list patterns in the title itself
_EXPLICIT_NUMBER_RE = re.compile(
    r"\b(top|best|worst|most|least|ranked?)\s+(\d+)\b"
    r"|\b(\d+)\s+(things?|ways?|reasons?|facts?|secrets?|tips?|signs?|habits?)\b",
    re.IGNORECASE,
)

# Patterns that scream "this can BE a list"
_IMPLICIT_LIST_KEYWORDS = [
    "things you didn't know",
    "you won't believe",
    "shocking facts",
    "dark secrets",
    "hidden truth",
    "mind blowing",
    "never knew",
    "scientists discovered",
    "what happens when",
    "terrifying",
    "classified",
    "banned",
    "declassified",
    "this is why",
    "nobody talks about",
]

_HOOK_SCORE_EXPLICIT = 1.0
_HOOK_SCORE_IMPLICIT = 0.55
_HOOK_SCORE_GENERIC = 0.25


def detect_hook(title: str) -> tuple[str, int, float]:
    """
    Analyses a title string to determine if it maps to a "Top N" hook.

    Returns:
        hook_type (str): "top_5" / "top_4" / "top_3" / "did_you_know" / "generic"
        hook_count (int): The N value (default 5)
        hook_score (float): Confidence 0.0 – 1.0
    """
    m = _EXPLICIT_NUMBER_RE.search(title)
    if m:
        # Extract the number from whichever capture group matched
        n_str = m.group(2) or m.group(3)
        if n_str:
            n = min(int(n_str), 10)  # Cap at 10 for sane video length
            hook_type = f"top_{n}" if n > 1 else "top_5"
            return hook_type, n, _HOOK_SCORE_EXPLICIT

    lower = title.lower()
    for kw in _IMPLICIT_LIST_KEYWORDS:
        if kw in lower:
            return "did_you_know", 5, _HOOK_SCORE_IMPLICIT

    return "generic", 5, _HOOK_SCORE_GENERIC


def _make_signal_id(source: str, raw_title: str) -> str:
    return hashlib.sha1(f"{source}:{raw_title}".encode()).hexdigest()[:16]


# ── Reddit Scanner ────────────────────────────────────────────────────────────

class RedditScanner:
    """
    Scans a list of subreddits for rising hot posts.
    Filters by upvote score, upvote ratio, and post age.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._reddit_cfg = self._cfg["sentinel"]["reddit"]
        self._reddit = None  # Lazy init to avoid import cost at startup

    def _get_client(self):  # type: ignore[return]
        """Lazily initialises the praw Reddit client."""
        if self._reddit is not None:
            return self._reddit

        try:
            import praw  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "[RedditScanner] praw is not installed. "
                "Run: pip install praw>=7.7.1"
            )

        client_id = self._reddit_cfg.get("client_id", "")
        client_secret = self._reddit_cfg.get("client_secret", "")
        user_agent = self._reddit_cfg.get("user_agent", "ApexEngine2.0/1.0")

        if not client_id or not client_secret:
            raise ValueError(
                "[RedditScanner] Reddit credentials missing. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your .env file."
            )

        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        log.info("[RedditScanner] praw client initialised (read-only mode)")
        return self._reddit

    def scan(self) -> list[TrendSignal]:
        """
        Scans all configured subreddits.

        Returns:
            List of TrendSignal objects, sorted by raw_score descending.
        """
        cfg = self._reddit_cfg
        subreddits: list[str] = cfg.get("subreddits", [])
        min_score: int = cfg.get("min_score", 500)
        min_ratio: float = cfg.get("min_upvote_ratio", 0.85)
        lookback_hours: int = cfg.get("lookback_hours", 6)
        lookback_cutoff = time.time() - (lookback_hours * 3600)

        if not cfg.get("enabled", True):
            log.info("[RedditScanner] Reddit scanner is disabled in settings.")
            return []

        reddit = self._get_client()
        signals: list[TrendSignal] = []
        seen_ids: set[str] = set()

        for subreddit_name in subreddits:
            try:
                signals.extend(
                    self._scan_subreddit(
                        reddit,
                        subreddit_name,
                        min_score,
                        min_ratio,
                        lookback_cutoff,
                        seen_ids,
                    )
                )
            except Exception as exc:
                log.warning(
                    f"[RedditScanner] Failed to scan r/{subreddit_name}: {exc}"
                )

        log.info(
            f"[RedditScanner] Scan complete — {len(signals)} signals "
            f"from {len(subreddits)} subreddit(s)"
        )
        return sorted(signals, key=lambda s: s.raw_score, reverse=True)

    def _scan_subreddit(
        self,
        reddit: Any,
        subreddit_name: str,
        min_score: int,
        min_ratio: float,
        lookback_cutoff: float,
        seen_ids: set[str],
    ) -> list[TrendSignal]:
        signals: list[TrendSignal] = []

        subreddit = reddit.subreddit(subreddit_name)

        # Fetch top 50 hot posts — "hot" is more reliable than "rising"
        # for detecting what's actually going viral right now
        posts = list(subreddit.hot(limit=50))
        log.debug(f"[RedditScanner] r/{subreddit_name}: fetched {len(posts)} posts")

        for post in posts:
            # ── Age filter ────────────────────────────────────────────────────
            if post.created_utc < lookback_cutoff:
                continue

            # ── Quality filters ───────────────────────────────────────────────
            if post.score < min_score:
                continue
            if post.upvote_ratio < min_ratio:
                continue
            if post.is_self and len(post.selftext) < 50:
                # Skip nearly-empty self posts (no content to work with)
                continue
            if post.stickied or post.distinguished:
                continue  # Skip mod posts

            # ── Dedup within this scan ────────────────────────────────────────
            sig_id = _make_signal_id("reddit", post.title)
            if sig_id in seen_ids:
                continue
            seen_ids.add(sig_id)

            # ── Hook detection ────────────────────────────────────────────────
            hook_type, hook_count, hook_score = detect_hook(post.title)

            # ── Velocity score ────────────────────────────────────────────────
            # score_per_hour = post.score / max(post age in hours, 0.1)
            age_hours = max((time.time() - post.created_utc) / 3600, 0.1)
            velocity = post.score / age_hours

            # Normalise against a reasonable ceiling (10k upvotes/hour = very viral)
            raw_score = min(velocity / 10_000, 1.0)

            # ── Extract keywords from title ───────────────────────────────────
            keywords = _extract_keywords(post.title)

            signals.append(
                TrendSignal(
                    signal_id=sig_id,
                    source="reddit",
                    raw_title=post.title,
                    topic=_clean_topic(post.title),
                    keywords=keywords,
                    source_url=f"https://reddit.com{post.permalink}",
                    raw_score=raw_score,
                    hook_score=hook_score,
                    hook_type=hook_type,
                    hook_count=hook_count,
                    extra={
                        "subreddit": subreddit_name,
                        "reddit_score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "age_hours": round(age_hours, 2),
                    },
                )
            )

        log.debug(
            f"[RedditScanner] r/{subreddit_name}: "
            f"{len(signals)} qualifying signals"
        )
        return signals


# ── Helpers ───────────────────────────────────────────────────────────────────

# Words to strip when cleaning a Reddit title into a topic
_STRIP_WORDS = re.compile(
    r"\b(TIL|ELI5|AMA|AITA|CMV|OC|PSA|NSFW|UPDATE|EDIT|reddit|r/\w+)\b",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s{2,}")


def _clean_topic(title: str) -> str:
    """Strip Reddit-specific noise from a post title to get a clean topic."""
    cleaned = _STRIP_WORDS.sub("", title)
    cleaned = re.sub(r"[\[\(].*?[\]\)]", "", cleaned)  # Remove [brackets]
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip(" :-–—")
    return cleaned[:200]  # Hard cap


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "that", "this", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "can", "it", "its", "i", "you", "he", "she",
    "we", "they", "them", "their", "what", "which", "who", "how",
    "when", "where", "why", "not", "no", "so", "if", "up", "out",
    "about", "just", "been", "very",
}


def _extract_keywords(title: str) -> list[str]:
    """Extract meaningful keywords from a title, dropping stopwords."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", title.lower())
    return list(dict.fromkeys(w for w in words if w not in _STOPWORDS))[:10]
