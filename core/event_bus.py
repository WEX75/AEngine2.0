"""
core/event_bus.py

Lightweight asyncio pub/sub event bus.
Modules publish events by name; subscribers receive them as dataclass payloads.

Usage:
    bus = EventBus()

    @bus.subscribe("TREND_FOUND")
    async def on_trend(payload: TrendFoundPayload):
        ...

    await bus.publish("TREND_FOUND", payload)
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from core.logger import get_logger

log = get_logger(__name__)

# ── Event names (constants to avoid typos) ──────────────────────────────────
TREND_FOUND = "TREND_FOUND"
SCRIPT_READY = "SCRIPT_READY"
RENDER_COMPLETE = "RENDER_COMPLETE"
SENTINEL_ERROR = "SENTINEL_ERROR"
BRAIN_ERROR = "BRAIN_ERROR"
FACTORY_ERROR = "FACTORY_ERROR"
STREAM_VOD_INGESTED = "STREAM_VOD_INGESTED"
MOMENT_DETECTED = "MOMENT_DETECTED"
CLIP_SCRIPT_READY = "CLIP_SCRIPT_READY"
CLIP_RENDER_READY = "CLIP_RENDER_READY"


# ── Payload base ─────────────────────────────────────────────────────────────
@dataclass
class BasePayload:
    event_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrendFoundPayload(BasePayload):
    """Emitted by The Sentinel when a qualifying trend is detected."""
    event_name: str = TREND_FOUND
    trend_id: str = ""                 # Unique hash of the trend topic
    topic: str = ""                    # e.g. "Scientists discovered X"
    hook_type: str = ""                # e.g. "top_5", "top_4", "did_you_know"
    hook_count: int = 5                # The N in "Top N"
    viral_score: float = 0.0           # 0.0 – 1.0 composite score
    source: str = ""                   # "reddit", "google_trends", "combined"
    source_url: str = ""               # Original post/link for fact-checking
    keywords: list[str] = field(default_factory=list)
    raw_title: str = ""                # Original headline / post title


@dataclass
class ScriptReadyPayload(BasePayload):
    """Emitted by The Brain when a video script is generated."""
    event_name: str = SCRIPT_READY
    trend_id: str = ""
    topic: str = ""
    hook: str = ""          # First 3-second hook line
    body: list[str] = field(default_factory=list)   # Numbered body points
    cta: str = ""           # Call-to-action / cliffhanger
    full_script: str = ""   # Concatenated for TTS


@dataclass
class RenderCompletePayload(BasePayload):
    """Emitted by The Factory when a video is rendered."""
    event_name: str = RENDER_COMPLETE
    trend_id: str = ""
    output_path: str = ""
    duration_seconds: float = 0.0


@dataclass
class StreamVodIngestedPayload(BasePayload):
    """
    Emitted once a stream VOD's chat log + audio track have been extracted
    and are ready for moment detection (Pillar 1, stream-clipping mode).
    Deliberately carries only paths to the small chat/audio artifacts, not
    the full VOD — Sentinel must never load the full VOD up front (CLAUDE.md
    Pillar 1). Field set matches the STREAM_VOD_INGESTED contract in
    CLAUDE.md exactly.
    """
    event_name: str = STREAM_VOD_INGESTED
    vod_id: str = ""
    streamer_id: str = ""
    source_path: str = ""
    duration_sec: float = 0.0
    chat_log_path: str = ""
    audio_track_path: str = ""
    subtitle_track_path: str = ""      # "" = no subtitle track provided (see
                                        # sentinel/stream_ingest.py's CRITICAL
                                        # PRODUCT DECISION on chat-less + subtitle-less intake)
    ingested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MomentDetectedPayload(BasePayload):
    """
    One merged high-score candidate window from StreamMomentDetector,
    ready for the Brain (Pillar 2) to select a segment range from. Field
    names mirror sentinel.stream_moment_detector.MomentCandidate exactly —
    this payload is a pure conversion, not a new model with extra fields.
    Field set matches the MOMENT_DETECTED contract in CLAUDE.md exactly.
    """
    event_name: str = MOMENT_DETECTED
    vod_id: str = ""
    window_start: float = 0.0
    window_end: float = 0.0
    composite_score: float = 0.0
    chat_score: float = 0.0
    audio_score: float = 0.0
    semantic_score: float = 0.0        # subtitle-keyword z at the peak bucket; 0.0 when no subtitle track
    top_spam_tokens: list[str] = field(default_factory=list)
    detection_method: str = "chat_audio_zscore_fusion"


@dataclass
class ClipScriptReadyPayload(BasePayload):
    """
    Emitted by the Brain (brain/clip_script_generator.py) once a
    MOMENT_DETECTED window has been narrowed to a concrete segment range by
    the LLM. start_segment_id/end_segment_id are indices into the Whisper
    indexed-segment array for that VOD — never raw timestamps (CLAUDE.md
    Pillar 2: local 8B-class models are unreliable at numeric precision).
    caption_segments is the actual {id, start, end, text} slice between
    those ids, computed by code (not the LLM) once the ids are finalized —
    this is what the Cutter (Pillar 3) will burn in as captions.
    Field set matches the CLIP_SCRIPT_READY contract in CLAUDE.md exactly.
    """
    event_name: str = CLIP_SCRIPT_READY
    vod_id: str = ""
    start_segment_id: int = 0
    end_segment_id: int = 0
    caption_segments: list[dict[str, Any]] = field(default_factory=list)
    hook_reason: str = ""


@dataclass
class ClipRenderReadyPayload(BasePayload):
    """
    Emitted by the Cutter (cutter/clip_renderer.py) once a CLIP_SCRIPT_READY
    selection has been rendered to a final vertical 9:16 video file — the
    last event in the stream-clipping pipeline (no further hand-off).
    duration_sec is the CLIP's own length (derived from caption_segments'
    real timestamps), not how long rendering took — that's render_seconds.
    Field set matches the CLIP_RENDER_READY contract in CLAUDE.md ("final
    output path + metadata").
    """
    event_name: str = CLIP_RENDER_READY
    vod_id: str = ""
    output_path: str = ""
    duration_sec: float = 0.0
    encoder_used: str = ""
    render_seconds: float = 0.0


# ── Bus implementation ────────────────────────────────────────────────────────
_Handler = Callable[[Any], Coroutine[Any, Any, None]]


class EventBus:
    """
    Simple asyncio event bus.
    All handlers are called concurrently via asyncio.gather().
    Exceptions in individual handlers are caught and logged — they do NOT
    propagate to the publisher, ensuring one bad handler can't crash the system.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[_Handler]] = defaultdict(list)

    def subscribe(self, event_name: str) -> Callable[[_Handler], _Handler]:
        """Decorator to register an async handler for a given event name."""
        def decorator(fn: _Handler) -> _Handler:
            self._handlers[event_name].append(fn)
            log.debug(f"[EventBus] '{fn.__qualname__}' subscribed to '{event_name}'")
            return fn
        return decorator

    def subscribe_fn(self, event_name: str, handler: _Handler) -> None:
        """Programmatic (non-decorator) handler registration."""
        self._handlers[event_name].append(handler)
        log.debug(f"[EventBus] '{handler.__qualname__}' subscribed to '{event_name}'")

    def unsubscribe(self, event_name: str, handler: _Handler) -> None:
        """Remove a previously registered handler. Safe to call even if not registered."""
        try:
            self._handlers[event_name].remove(handler)
            log.debug(f"[EventBus] '{handler.__qualname__}' unsubscribed from '{event_name}'")
        except ValueError:
            pass

    async def publish(self, event_name: str, payload: BasePayload) -> None:
        """
        Publish an event. All registered handlers are called concurrently.
        Safe — never raises even if handlers fail.
        """
        handlers = self._handlers.get(event_name, [])
        if not handlers:
            log.debug(f"[EventBus] No handlers for '{event_name}'")
            return

        log.info(f"[EventBus] Publishing '{event_name}' to {len(handlers)} handler(s)")

        async def _safe_call(handler: _Handler) -> None:
            try:
                await handler(payload)
            except Exception as exc:
                log.exception(
                    f"[EventBus] Handler '{handler.__qualname__}' "
                    f"raised an exception for event '{event_name}': {exc}"
                )

        await asyncio.gather(*(_safe_call(h) for h in handlers))


# ── Singleton bus shared across the whole application ────────────────────────
_bus: EventBus | None = None


def get_bus() -> EventBus:
    """Returns the application-level singleton EventBus."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
