"""
orchestrator/clip_orchestrator.py — cross-pillar wiring for stream-clipping mode
Apex Engine 2.0

Solves the vod_id -> (source_path, streamer_id) mapping gap flagged three
times now (in sentinel/stream_ingest.py, brain/clip_whisper_bridge.py, and
cutter/clip_renderer.py's docstrings): none of MOMENT_DETECTED or
CLIP_SCRIPT_READY carry source_path or streamer_id — only
STREAM_VOD_INGESTED does. This module retains that mapping and resolves
it for the two downstream hand-offs that need it:

    STREAM_VOD_INGESTED --> [registry[vod_id] = (source_path, streamer_id)]

    MOMENT_DETECTED --> look up registry --> clip_whisper_bridge.submit_moment(
                             moment, source_path=<looked up>)

    CLIP_SCRIPT_READY --> look up registry --> clip_renderer.submit_render(
                              clip, source_path=<looked up>, streamer_id=<looked up>)

Why this module exists at all, instead of extending an existing one
------------------------------------------------------------------------
Every other module in this pipeline scoped itself to ONE hop (detection
window -> transcript, transcript -> segment selection, selection ->
render) and explicitly deferred "the orchestrator" as future work,
because building it required committing to a cross-pillar data-retention
policy (this registry) that none of those single-hop modules had any
business owning. This module is that commitment, made explicit and
testable in one place instead of smeared across three.

Why `orchestrator/`, not `core/`, not inside a pillar package, not apex.py
------------------------------------------------------------------------
- **Not `core/`**: every existing `core/` module (`event_bus`,
  `job_queue`, `config_loader`, `logger`) is pillar-agnostic — none of
  them import a single pillar-specific type. This module imports
  `MomentDetectedPayload`, `ClipScriptReadyPayload`,
  `ClipWhisperBridge`, `ClipRenderer` — real business knowledge of the
  stream-clipping pipeline's shape. Putting it in `core/` would make it
  the first module there anyone importing `core/` for its "safe, generic
  infra" guarantee would have to reason about differently.
- **Not inside `brain/`, `sentinel/`, or `cutter/`**: it doesn't belong
  to any single pillar — it depends on Sentinel's output shape
  (MOMENT_DETECTED), Brain's output shape (CLIP_SCRIPT_READY, via
  clip_whisper_bridge), and Cutter's input shape (submit_render). Filing
  it under any one of those would misattribute ownership the same way
  filing a referee under one of the two teams would.
- **Not apex.py**: apex.py's job is booting/lifecycling pillar objects,
  not owning a stateful registry or event-handling logic. Growing that
  responsibility into apex.py would conflate "wire the app together" with
  "coordinate the stream-clipping data flow."
- **A new top-level package**, mirroring `brain/`/`sentinel/`/`cutter/`
  structurally, but explicitly NOT a fourth pillar — CLAUDE.md defines
  exactly three (Sentinel, Brain, Cutter). This package is the connective
  tissue for ONE of the two operating modes described at the top of
  CLAUDE.md ("Stream Clipping"), parallel to how the trend-to-script mode
  is Sentinel -> Brain -> Factory. Naming it `orchestrator/` rather than
  e.g. `pillar4/` is deliberate — it should never be mistaken for owning
  detection, selection, or rendering logic; it only wires already-built
  pieces together.

Registry: bounded, LRU by ACCESS (not just insert)
------------------------------------------------------
A plain dict keyed by vod_id would grow forever in a long-running
process — every VOD ever ingested stays resident. Bounded via
`collections.OrderedDict` acting as an LRU cache: `max_registry_size`
entries (default 200 — see reasoning below), oldest-by-recency evicted
first when a new STREAM_VOD_INGESTED would exceed it.

"Recency" means last ACCESS, not last insert — every successful
`_registry_get()` (a MOMENT_DETECTED or CLIP_SCRIPT_READY hit) moves that
entry to the end, same as a fresh insert. This matters concretely: a VOD
can produce multiple MOMENT_DETECTED candidates (up to the
tests/test_stream_sentinel_scale.py MAX_CANDIDATES=5 ceiling this whole
pipeline is already sized against), and each candidate's CLIP_SCRIPT_READY
can arrive minutes later, well after other VODs have been ingested in
between. If eviction only considered insert time, a VOD sitting mid-pipeline
could be evicted out from under itself by unrelated newer VODs even while
still actively referenced. Refreshing on access keeps anything still "in
flight" hot regardless of insertion order.

Default cap of 200: each entry is three short strings + a timestamp —
even 200 entries is trivial memory (nowhere near CLAUDE.md's RAM ceiling
concerns, which are about resident ML models, not a few KB of path
strings). The cap isn't a memory control, it's a leak guard — sized
generously above any realistic number of VODs actively mid-pipeline at
once (each full VOD-to-clips cycle takes minutes; 200 concurrent
in-flight VODs would imply either an enormous ingestion rate this
pipeline hasn't been sized for, or something stuck never completing —
both worth surfacing via the eviction WARNING log, not silently
tolerating via an unbounded dict).

Unknown vod_id: log and drop, never crash
----------------------------------------------
A MOMENT_DETECTED or CLIP_SCRIPT_READY for a vod_id with no registry
entry (never ingested through this orchestrator, or evicted) is logged at
WARNING and dropped — the handler returns cleanly. This is deliberate
defense-in-depth: core.event_bus.EventBus.publish() already isolates
handler exceptions from each other, but that's a safety net, not a
design — every other job body in this codebase (Brain, ClipWhisperBridge,
ClipRenderer) handles its own known failure modes explicitly rather than
leaning on the bus's catch-all, and this module follows the same
discipline.

Brain instance reuse (a wiring correctness note for whoever constructs this)
------------------------------------------------------------------------------
This module doesn't construct its own ClipWhisperBridge/ClipRenderer —
it's handed already-constructed instances. Whoever DOES construct them
(apex.py) must pass the SAME Brain instance already running for
trend-to-script mode into ClipWhisperBridge — CLAUDE.md's "JobQueue
ownership" section is explicit that Brain owns exactly ONE queue for
every LLM-call-shaped job, trend script generation AND clip segment
selection both. Constructing a second Brain for stream-clipping mode
would silently create a second queue, defeating that design. Not
enforced here (this module has no visibility into how bridge/renderer
were built) — enforced by apex.py's wiring, documented here so the
constraint travels with the code that depends on it.

Per-streamer target_duration_range override (settings.yaml
`stream_clipping.target_duration_ranges`)
------------------------------------------------------------------------
Why this lives HERE, not in Brain or Cutter's config: target_duration_range
is a Brain-consumed parameter (ClipScriptGenerator.generate(), per
CLAUDE.md's "Clip length" — never hardcoded, a job parameter the LLM's
selection and auto-repair both honor), but resolving it PER STREAMER needs
streamer_id, and streamer_id is only known here — exactly the same
vod_id -> streamer_id resolution gap this whole module exists to close for
Cutter's facecam bbox / crop_mode lookups. Brain's own queue/generator
never sees a streamer_id (MOMENT_DETECTED doesn't carry one, deliberately —
see this module's opening section), so it has no way to do this lookup
itself. This orchestrator already resolves `info.streamer_id` right where
it calls `bridge.submit_moment(..., target_duration_range=...)` — that's
the one and only place in the whole pipeline where "streamer_id" and
"about to pick a target_duration_range" are both in scope at once.

Config lives under `stream_clipping:` (not `cutter:`) because this
override is resolved and applied by THIS module (the stream-clipping
cross-pillar wiring), not by Cutter — `cutter:` settings are consumed
directly by ClipRenderer. No entry for a streamer_id -> the constructor's
`target_duration_range` default (12.0, 75.0) applies unchanged, same
"explicit entry overrides, no entry preserves prior behavior" pattern as
`cutter.crop_modes`. Each configured range is validated at construction —
a 2-element (min_sec, max_sec) with 0 < min_sec < max_sec — raising
ValueError immediately on anything else (not a literal enum like
crop_modes/caption_styles/visual_style, but the same "fail loudly at
construction, never silently misbehave at render time" discipline).

The override reaches ClipScriptGenerator.generate() through the existing,
unmodified call chain — this module's only change is WHICH value it
passes into submit_moment(): _on_moment_detected -> bridge.submit_moment(
target_duration_range=<resolved>) -> ClipWhisperBridge._transcribe_and_submit
-> brain.select_clip_script(..., target_duration_range, ...) ->
ClipScriptGenerator.generate(..., target_duration_range, ...). That chain
already threaded a target_duration_range parameter end to end before this
feature existed; this feature only makes the value vary per streamer_id
instead of being one constant for every VOD.

What this does NOT do
--------------------------
No auto-posting, no campaign/brief integration, no orchestrator-level
retry/backoff beyond what clip_whisper_bridge/clip_renderer already do
internally. This module's job ends at "the right downstream method got
called with the right arguments" — nothing about what happens to
CLIP_RENDER_READY after it's published.

One more gap this surfaces, not fixes: nothing in this repo currently
PUBLISHES STREAM_VOD_INGESTED in live operation (only
tools/manual_smoke_test_e2e.py constructs one, by hand, for testing).
Enabling this orchestrator wires up real subscribers ready to react the
moment something does — today, in live operation, that's inert plumbing
with nothing feeding it. Building that producer (a real VOD/chat
ingestion tool) is a separate future task.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from brain.clip_whisper_bridge import ClipWhisperBridge
from cutter.clip_renderer import ClipRenderer
from core.config_loader import get_config
from core.event_bus import (
    CLIP_SCRIPT_READY,
    MOMENT_DETECTED,
    STREAM_VOD_INGESTED,
    ClipScriptReadyPayload,
    EventBus,
    MomentDetectedPayload,
    StreamVodIngestedPayload,
    get_bus,
)
from core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MAX_REGISTRY_SIZE = 200
_DEFAULT_TARGET_DURATION_RANGE: tuple[float, float] = (12.0, 75.0)


def _validate_target_duration_range(streamer_id: str, value: Any) -> tuple[float, float]:
    """
    Validates one `stream_clipping.target_duration_ranges` entry: a
    2-element (min_sec, max_sec) with 0 < min_sec < max_sec. Raises
    ValueError immediately on anything else — same "fail loudly at
    construction" discipline as ClipRenderer's crop_modes/caption_styles/
    visual_style enum checks, applied here to a numeric range instead of
    an enum (see module docstring's "Per-streamer target_duration_range
    override").
    """
    try:
        lo_raw, hi_raw = value
        lo, hi = float(lo_raw), float(hi_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid target_duration_range {value!r} for streamer_id {streamer_id!r} — "
            "must be a 2-element [min_sec, max_sec] sequence of numbers."
        )
    if not (lo > 0 and hi > lo):
        raise ValueError(
            f"Invalid target_duration_range ({lo}, {hi}) for streamer_id {streamer_id!r} — "
            "min_sec must be > 0 and max_sec must be > min_sec."
        )
    return (lo, hi)


@dataclass(frozen=True)
class _VodInfo:
    source_path: str
    streamer_id: str
    ingested_at: datetime


class ClipOrchestrator:
    """
    Subscribes to STREAM_VOD_INGESTED (to populate the registry),
    MOMENT_DETECTED (to call clip_whisper_bridge.submit_moment with the
    resolved source_path), and CLIP_SCRIPT_READY (to call
    clip_renderer.submit_render with the resolved source_path/streamer_id).

    Takes already-constructed ClipWhisperBridge/ClipRenderer instances
    rather than building its own — see module docstring's "Brain instance
    reuse" note for why that matters for correctness, not just style.
    """

    def __init__(
        self,
        bridge:                  ClipWhisperBridge,
        renderer:                ClipRenderer,
        bus:                     EventBus | None = None,
        max_registry_size:       int = _DEFAULT_MAX_REGISTRY_SIZE,
        target_duration_range:   tuple[float, float] = _DEFAULT_TARGET_DURATION_RANGE,
        target_duration_ranges:  dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._bridge                = bridge
        self._renderer              = renderer
        self._bus                   = bus or get_bus()
        self._max_registry_size     = max_registry_size
        self._target_duration_range = target_duration_range

        if target_duration_ranges is not None:
            raw_ranges = target_duration_ranges
        else:
            cfg = get_config()
            raw_ranges = cfg.get("stream_clipping", {}).get("target_duration_ranges", {}) or {}
        # Strict validation at construction — see module docstring's
        # "Per-streamer target_duration_range override": a malformed entry
        # must fail loudly here, never silently misbehave at render time.
        self._target_duration_ranges = {
            sid: _validate_target_duration_range(sid, value) for sid, value in raw_ranges.items()
        }

        self._registry: "OrderedDict[str, _VodInfo]" = OrderedDict()
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to STREAM_VOD_INGESTED, MOMENT_DETECTED, CLIP_SCRIPT_READY."""
        if self._running:
            log.warning("[ClipOrchestrator] Already running — ignoring duplicate start().")
            return
        self._bus.subscribe_fn(STREAM_VOD_INGESTED, self._on_vod_ingested)
        self._bus.subscribe_fn(MOMENT_DETECTED, self._on_moment_detected)
        self._bus.subscribe_fn(CLIP_SCRIPT_READY, self._on_clip_script_ready)
        self._running = True
        log.info(
            f"[ClipOrchestrator] Online — subscribed to STREAM_VOD_INGESTED, "
            f"MOMENT_DETECTED, CLIP_SCRIPT_READY (registry cap={self._max_registry_size})."
        )

    async def stop(self) -> None:
        """Unsubscribe. Does not touch bridge/renderer — caller owns their lifecycle."""
        if not self._running:
            return
        self._bus.unsubscribe(STREAM_VOD_INGESTED, self._on_vod_ingested)
        self._bus.unsubscribe(MOMENT_DETECTED, self._on_moment_detected)
        self._bus.unsubscribe(CLIP_SCRIPT_READY, self._on_clip_script_ready)
        self._running = False
        log.info(f"[ClipOrchestrator] Stopped. {len(self._registry)} vod_id(s) still in registry at shutdown.")

    # ── Registry (bounded LRU, refreshed on access — see module docstring) ────

    def _registry_put(self, vod_id: str, source_path: str, streamer_id: str) -> None:
        if vod_id in self._registry:
            self._registry.move_to_end(vod_id)
        self._registry[vod_id] = _VodInfo(
            source_path=source_path,
            streamer_id=streamer_id,
            ingested_at=datetime.now(timezone.utc),
        )
        while len(self._registry) > self._max_registry_size:
            evicted_vod_id, evicted_info = self._registry.popitem(last=False)
            log.warning(
                f"[ClipOrchestrator] Registry at capacity ({self._max_registry_size}) — "
                f"evicted vod_id={evicted_vod_id} (ingested_at={evicted_info.ingested_at.isoformat()}). "
                f"Any MOMENT_DETECTED/CLIP_SCRIPT_READY for this vod arriving after this point "
                f"will be logged and dropped as unknown."
            )

    def _registry_get(self, vod_id: str) -> _VodInfo | None:
        info = self._registry.get(vod_id)
        if info is not None:
            self._registry.move_to_end(vod_id)  # still actively referenced — stays "hot"
        return info

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_vod_ingested(self, payload: StreamVodIngestedPayload) -> None:
        self._registry_put(payload.vod_id, payload.source_path, payload.streamer_id)
        log.info(
            f"[ClipOrchestrator] Registered vod_id={payload.vod_id} -> "
            f"source_path={payload.source_path}, streamer_id={payload.streamer_id} "
            f"(registry size={len(self._registry)}/{self._max_registry_size})"
        )

    async def _on_moment_detected(self, payload: MomentDetectedPayload) -> None:
        info = self._registry_get(payload.vod_id)
        if info is None:
            log.warning(
                f"[ClipOrchestrator] MOMENT_DETECTED for unknown vod_id={payload.vod_id} — "
                f"dropping. No STREAM_VOD_INGESTED was seen for this vod_id, or its registry "
                f"entry was evicted (registry cap={self._max_registry_size})."
            )
            return
        # Per-streamer override (settings.yaml stream_clipping.
        # target_duration_ranges) wins; no entry -> the constructor default
        # applies unchanged, same precedence pattern as ClipRenderer's
        # crop_modes (see module docstring).
        target_duration_range = self._target_duration_ranges.get(
            info.streamer_id, self._target_duration_range
        )
        await self._bridge.submit_moment(
            payload,
            source_path=info.source_path,
            target_duration_range=target_duration_range,
        )

    async def _on_clip_script_ready(self, payload: ClipScriptReadyPayload) -> None:
        info = self._registry_get(payload.vod_id)
        if info is None:
            log.warning(
                f"[ClipOrchestrator] CLIP_SCRIPT_READY for unknown vod_id={payload.vod_id} — "
                f"dropping. No STREAM_VOD_INGESTED was seen for this vod_id, or its registry "
                f"entry was evicted (registry cap={self._max_registry_size})."
            )
            return
        await self._renderer.submit_render(
            payload,
            source_path=info.source_path,
            streamer_id=info.streamer_id,
        )
