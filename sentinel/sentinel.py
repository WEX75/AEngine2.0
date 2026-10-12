"""
sentinel/sentinel.py  —  The Sentinel (Pillar 1 Orchestrator)
Apex Engine 2.0

Runs the Google Trends scanner on a configurable schedule, scores results
via TrendScorer, and publishes TREND_FOUND events to the shared EventBus.

Modes
-----
  Autonomous loop  — called from apex.py; sleeps between cycles.
  CLI dry-run      — scans once, prints a Rich table, exits.

CLI usage
---------
  python -m sentinel.sentinel               # one shot (uses settings.yaml dry_run flag)
  python -m sentinel.sentinel --dry-run     # force dry-run
  python -m sentinel.sentinel --dry-run --limit 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

from core.config_loader import get_config
from core.event_bus import TREND_FOUND, TrendFoundPayload, get_bus
from core.logger import get_logger
from sentinel.google_trends import GoogleTrendsScanner
from sentinel.models import ScoredTrend, TrendSignal
from sentinel.trend_scorer import TrendScorer

log = get_logger(__name__)


class Sentinel:
    """
    Autonomous trend-intelligence module.

    Scan cycle
    ----------
    1. GoogleTrendsScanner.scan()  →  list[TrendSignal]
    2. TrendScorer.score()         →  list[ScoredTrend]  (top-K, filtered)
    3. Publish TREND_FOUND event for each ScoredTrend  (or print in dry-run)
    """

    def __init__(self, dry_run: bool = False) -> None:
        cfg = get_config()
        s   = cfg["sentinel"]

        self._interval_s: float = s.get("scan_interval_hours", 6) * 3600
        self._dry_run:    bool  = dry_run or s.get("dry_run", False)
        self._running:    bool  = False

        self._scanner = GoogleTrendsScanner()
        self._scorer  = TrendScorer()
        self._bus     = get_bus()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Async loop: scan → score → emit, then sleep. Runs until cancelled."""
        self._running = True
        log.info(
            f"[Sentinel] Autonomous loop started "
            f"(interval: {self._interval_s / 3600:.1f}h, dry_run={self._dry_run})"
        )
        while self._running:
            t0 = time.monotonic()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.exception(f"[Sentinel] Unhandled error in scan cycle: {exc}")

            elapsed  = time.monotonic() - t0
            sleep_s  = max(0.0, self._interval_s - elapsed)
            wake_at  = datetime.now(timezone.utc).strftime("%H:%M UTC")
            log.info(
                f"[Sentinel] Cycle done in {elapsed:.1f}s. "
                f"Next scan in {sleep_s / 3600:.1f}h (at ~{wake_at})."
            )
            await asyncio.sleep(sleep_s)

        log.info("[Sentinel] Loop stopped.")

    async def run_once(self) -> list[ScoredTrend]:
        """
        Execute a single scan-score-emit cycle.

        Blocking scanner calls are offloaded to a thread-pool executor so the
        asyncio event loop stays free during the pytrends HTTP calls.

        Returns:
            The list of ScoredTrend objects that were emitted (or printed).
        """
        log.info("[Sentinel] " + "=" * 54)
        log.info("[Sentinel] >> Scan cycle starting...")
        log.info("[Sentinel] " + "=" * 54)

        # ── 1. Scan (blocking → thread pool) ─────────────────────────────────
        log.info("[Sentinel] Running Google Trends scanner...")
        loop = asyncio.get_event_loop()
        signals: list[TrendSignal] = await loop.run_in_executor(
            None, self._scanner.scan
        )
        log.info(f"[Sentinel] Google Trends returned {len(signals)} signal(s).")

        if not signals:
            log.warning("[Sentinel] No signals this cycle — nothing to score.")
            return []

        # ── 2. Score ──────────────────────────────────────────────────────────
        log.info("[Sentinel] Scoring and ranking signals...")
        top: list[ScoredTrend] = self._scorer.score(signals)

        if not top:
            log.warning("[Sentinel] No trends met the viral_score threshold.")
            return []

        log.info(f"[Sentinel] TOP {len(top)} trend(s) qualified.")

        # ── 3. Emit events or print ───────────────────────────────────────────
        if self._dry_run:
            _print_results(top)
        else:
            for trend in top:
                await self._emit(trend)

        return top

    async def stop(self) -> None:
        """Gracefully stop the autonomous loop."""
        self._running = False

    # ── Private ───────────────────────────────────────────────────────────────

    async def _emit(self, trend: ScoredTrend) -> None:
        payload = TrendFoundPayload(
            trend_id    = trend.trend_id,
            topic       = trend.topic,
            hook_type   = trend.hook_type,
            hook_count  = trend.hook_count,
            viral_score = trend.viral_score,
            source      = trend.source,
            source_url  = trend.source_url,
            keywords    = trend.keywords,
            raw_title   = trend.raw_title,
        )
        log.info(
            f"[Sentinel] >> Emitting TREND_FOUND | "
            f"score={trend.viral_score:.3f} | "
            f"topic='{trend.topic[:50]}'"
        )
        await self._bus.publish(TREND_FOUND, payload)


# ── CLI pretty-printer ────────────────────────────────────────────────────────

def _print_results(trends: list[ScoredTrend]) -> None:
    """Renders a Rich table + JSON detail to stdout."""
    try:
        from rich.console import Console   # type: ignore[import]
        from rich.table   import Table     # type: ignore[import]

        console = Console(
            force_terminal=True,
            legacy_windows=False,
            highlight=False,
        )

        table = Table(
            title="APEX SENTINEL -- Trend Report",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )
        table.add_column("#",       style="dim",        width=3)
        table.add_column("Score",   style="bold green", width=7)
        table.add_column("Hook",    style="yellow",     width=12)
        table.add_column("Source",  style="cyan",       width=13)
        table.add_column("Topic",   style="white")

        for i, t in enumerate(trends, 1):
            table.add_row(
                str(i),
                f"{t.viral_score:.3f}",
                f"{t.hook_type}",
                t.source,
                t.topic[:70],
            )

        console.print()
        console.print(table)
        console.print()

        # Full JSON detail for debugging / Brain integration
        console.print("[bold]Full trend payloads:[/bold]")
        for t in trends:
            data = {
                "trend_id":        t.trend_id,
                "topic":           t.topic,
                "raw_title":       t.raw_title,
                "hook_type":       t.hook_type,
                "hook_count":      t.hook_count,
                "viral_score":     t.viral_score,
                "source":          t.source,
                "source_url":      t.source_url,
                "keywords":        t.keywords[:5],
                "component_scores": t.component_scores,
            }
            console.print_json(json.dumps(data, indent=2))

    except ImportError:
        # Graceful fallback if rich is not installed
        print("\n=== APEX SENTINEL — TREND REPORT ===")
        for i, t in enumerate(trends, 1):
            print(f"\n#{i} | score={t.viral_score:.3f} | {t.hook_type} | {t.source}")
            print(f"   Topic    : {t.topic}")
            print(f"   URL      : {t.source_url}")
            print(f"   Keywords : {', '.join(t.keywords[:5])}")
            print(f"   Scores   : {t.component_scores}")


# ── CLI entry point ───────────────────────────────────────────────────────────

async def _cli_main(dry_run: bool, limit: int) -> None:
    cfg = get_config()
    if limit > 0:
        cfg["sentinel"]["top_k_trends"] = limit
    sentinel = Sentinel(dry_run=dry_run)
    await sentinel.run_once()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apex Engine 2.0 — The Sentinel (Trend Intelligence Module)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Scan and print results; do not emit EventBus events.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Override top_k_trends for this run (0 = use settings.yaml).",
    )
    args = parser.parse_args()
    asyncio.run(_cli_main(dry_run=args.dry_run, limit=args.limit))
