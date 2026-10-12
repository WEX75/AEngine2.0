"""
sentinel/google_trends.py  —  The Sentinel: Google Trends Scanner
Apex Engine 2.0  |  Pillar 1

Fetches rising search queries from Google Trends using pytrends.
No API key required. Runs entirely offline-compatible (only makes
HTTP calls to trends.google.com).

Design notes
------------
* Only `related_queries()` is used.  Both `trending_searches()` and
  `realtime_trending_searches()` return HTTP 404 from Google's current
  API surface (verified June 2026) and have been removed.
* A urllib3 v2 compatibility patch is applied at import time because
  pytrends internally uses the old `method_whitelist` kwarg that was
  renamed to `allowed_methods` in urllib3 ≥ 2.0.
* Inter-request delays and 429 back-off are baked in so the scanner
  is safe to run autonomously on the 6-hour schedule.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from core.config_loader import get_config
from core.logger import get_logger
from sentinel.models import (
    TrendSignal,
    clean_topic,
    detect_hook,
    extract_keywords,
    make_signal_id,
)

if TYPE_CHECKING:
    pass  # keeps linters happy without heavy imports at module level

log = get_logger(__name__)

# Rising-query value ceiling for score normalisation.
# pytrends returns integers (% increase) or the string "Breakout" (≥5000%).
_MAX_RISING_PCT: int = 5000

# Evergreen seed keywords that reliably surface viral "Top N" content.
# We build_payload() around these to trigger related_queries().
_SEED_KEYWORDS_BY_CATEGORY: dict[int, list[str]] = {
    0:   ["world record broken", "scientists discovered", "things you didn't know"],
    16:  ["shocking news facts", "world record 2024", "breaking world record"],
}

# Polite inter-request pause (seconds).  Google's unofficial API tolerates
# ~1 request every 4 s before issuing 429s.
_REQUEST_DELAY_S: float = 4.5
_BACKOFF_429_S:   float = 35.0


# ── urllib3 v2 compatibility ──────────────────────────────────────────────────

def _patch_urllib3() -> None:
    """
    Monkey-patches urllib3.Retry so pytrends' legacy `method_whitelist` kwarg
    is silently remapped to `allowed_methods` (renamed in urllib3 ≥ 2.0).
    Idempotent — safe to call multiple times.
    """
    try:
        import urllib3                       # noqa: PLC0415
        import urllib3.util.retry as _rm    # noqa: PLC0415

        if getattr(_rm.Retry, "_apex_patched", False):
            return  # already patched

        _Orig = _rm.Retry

        class _PatchedRetry(_Orig):  # type: ignore[valid-type]
            _apex_patched = True

            def __init__(self, *args, **kwargs):
                if "method_whitelist" in kwargs:
                    if "allowed_methods" not in kwargs:
                        kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
                    else:
                        kwargs.pop("method_whitelist")
                super().__init__(*args, **kwargs)

        _rm.Retry = _PatchedRetry          # type: ignore[assignment]
        urllib3.Retry = _PatchedRetry      # type: ignore[assignment]
        log.debug("[GoogleTrends] urllib3.Retry patched for pytrends compatibility.")
    except Exception as exc:
        log.warning(f"[GoogleTrends] urllib3 patch skipped: {exc}")


# ── Scanner ───────────────────────────────────────────────────────────────────

class GoogleTrendsScanner:
    """
    Queries Google Trends for rising search queries and returns a list of
    TrendSignal objects ranked by raw_score.

    One instance is created per Sentinel scan cycle (cheap — no state held
    between cycles except config).
    """

    def __init__(self) -> None:
        cfg = get_config()
        gt = cfg["sentinel"]["google_trends"]

        self._enabled:    bool       = gt.get("enabled", True)
        self._geo:        str        = gt.get("geo", "US")
        self._timeframe:  str        = gt.get("timeframe", "now 1-d")
        self._categories: list[int]  = gt.get("categories", [0, 16])

    # ── Public ────────────────────────────────────────────────────────────────

    def scan(self) -> list[TrendSignal]:
        """
        Execute a full scan across all configured categories.

        Returns:
            Deduplicated list of TrendSignal, sorted by raw_score descending.
        """
        if not self._enabled:
            log.info("[GoogleTrends] Scanner disabled in settings — skipping.")
            return []

        _patch_urllib3()
        pytrends = self._build_client()

        signals:  list[TrendSignal] = []
        seen_ids: set[str]          = set()

        for cat_id in self._categories:
            seeds = _SEED_KEYWORDS_BY_CATEGORY.get(cat_id, _SEED_KEYWORDS_BY_CATEGORY[0])
            log.info(
                f"[GoogleTrends] Scanning category {cat_id} "
                f"with {len(seeds[:2])} seed(s)..."
            )
            for seed in seeds[:2]:   # max 2 seeds per category → 4 API calls total
                batch = self._fetch_rising(pytrends, seed, cat_id, seen_ids)
                signals.extend(batch)
                log.debug(f"[GoogleTrends]   seed='{seed}' → {len(batch)} signals")
                time.sleep(_REQUEST_DELAY_S)

        log.info(
            f"[GoogleTrends] Scan complete — "
            f"{len(signals)} unique signals across {len(self._categories)} category(ies)."
        )
        return sorted(signals, key=lambda s: s.raw_score, reverse=True)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_client(self):  # type: ignore[return]
        """Lazily imports and configures a TrendReq instance."""
        try:
            from pytrends.request import TrendReq  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "[GoogleTrends] pytrends not installed. "
                "Run: pip install pytrends>=4.9.2"
            )
        # retries=2 with our patched Retry class
        return TrendReq(
            hl="en-US",
            tz=360,
            geo=self._geo,
            retries=2,
            backoff_factor=1.0,
        )

    def _fetch_rising(
        self,
        pytrends,
        seed:     str,
        cat_id:   int,
        seen_ids: set[str],
    ) -> list[TrendSignal]:
        """
        Calls build_payload() + related_queries() for one seed keyword and
        extracts the "rising" sub-table.

        Returns an empty list (never raises) so a single failure doesn't abort
        the whole scan cycle.
        """
        try:
            pytrends.build_payload(
                kw_list=[seed],
                cat=cat_id,
                timeframe=self._timeframe,
                geo=self._geo,
            )
            related = pytrends.related_queries()
        except Exception as exc:
            self._handle_request_error(exc, seed)
            return []

        if seed not in related:
            return []

        rising_df = related[seed].get("rising")
        if rising_df is None or rising_df.empty:
            log.debug(f"[GoogleTrends] No rising queries for seed='{seed}'.")
            return []

        signals: list[TrendSignal] = []

        for _, row in rising_df.iterrows():
            query: str = str(row.get("query", "") or "").strip()
            if not query or len(query) < 4:
                continue

            sig_id = make_signal_id("google_trends", query)
            if sig_id in seen_ids:
                continue
            seen_ids.add(sig_id)

            raw_score  = self._normalise_value(row.get("value", 0))
            hook_type, hook_count, hook_score = detect_hook(query)

            signals.append(TrendSignal(
                signal_id  = sig_id,
                source     = "google_trends",
                raw_title  = query,
                topic      = clean_topic(query),
                keywords   = extract_keywords(query) + extract_keywords(seed),
                source_url = (
                    f"https://trends.google.com/trends/explore"
                    f"?q={query.replace(' ', '+')}&geo={self._geo}"
                ),
                raw_score  = raw_score,
                hook_score = hook_score,
                hook_type  = hook_type,
                hook_count = hook_count,
                extra      = {
                    "seed":          seed,
                    "category_id":   cat_id,
                    "rising_value":  row.get("value", 0),
                },
            ))

        return signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_value(value) -> float:
        """Convert pytrends rising value (int % or 'Breakout') to 0–1 float."""
        if isinstance(value, str) and value.lower() == "breakout":
            return 1.0
        try:
            return min(float(value) / _MAX_RISING_PCT, 1.0)
        except (TypeError, ValueError):
            return 0.40  # Reasonable default for unknown values

    @staticmethod
    def _handle_request_error(exc: Exception, seed: str) -> None:
        """Logs errors with context-appropriate level and blocks on 429."""
        msg = str(exc)
        if "429" in msg or "Too Many Requests" in msg.lower():
            log.warning(
                f"[GoogleTrends] Rate-limited (429) on seed='{seed}'. "
                f"Backing off {_BACKOFF_429_S:.0f}s..."
            )
            time.sleep(_BACKOFF_429_S)
        elif any(code in msg for code in ("400", "404")):
            log.debug(f"[GoogleTrends] HTTP error for seed='{seed}': {exc}")
        else:
            log.warning(f"[GoogleTrends] Unexpected error for seed='{seed}': {exc}")
