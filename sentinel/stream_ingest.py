"""
sentinel/stream_ingest.py  —  Pillar 1: Stream Ingest → Detection integration
Apex Engine 2.0

This module is pure wiring. All detection logic lives in
sentinel.stream_moment_detector.StreamMomentDetector, which is done and
tested (tests/test_stream_sentinel.py, tests/test_stream_sentinel_scale.py).
StreamIngestHandler's only job is:

    STREAM_VOD_INGESTED  →  load chat/audio  →  detector.detect()  →  MOMENT_DETECTED (one per candidate)

Fixture / on-disk format
-------------------------
STREAM_VOD_INGESTED carries *paths*, not data (CLAUDE.md Pillar 1: never
load the full VOD up front). This module defines the on-disk format for the
two small artifacts it actually reads:

  chat_log_path     JSON Lines — one message per line:
                         {"timestamp": <float, seconds from VOD start>, "text": <str>}

  audio_track_path  One JSON object:
                         {"bucket_sec": <float, optional, default 1.0>,
                          "dbfs": [<float>, ...]}
                     bucket_sec is the audio track's own native resolution
                     and may differ from the detector's chat/output
                     bucket_sec (default 1.0) — StreamMomentDetector.detect's
                     audio_bucket_sec parameter (added for the granularity
                     fix in stream_moment_detector.py) aligns the two via
                     nearest-timestamp matching, so ingestion just has to
                     pass through whatever resolution the audio was
                     actually captured at.

  subtitle_track_path   Optional. A .vtt or .srt file (either extension is
                     auto-detected by content, not by filename) — see
                     load_subtitle_track()'s own docstring for the parsing
                     and rolling-auto-caption-dedup rules. "" (default,
                     matching StreamVodIngestedPayload's field default)
                     means no subtitle track was provided for this VOD.

CRITICAL PRODUCT DECISION — insufficient-signal rejection at intake
---------------------------------------------------------------------
Archetype-B-shaped content (no chat channel at all — interview/podcast VODs)
can have audio that never clears the composite threshold on its own. Now that
subtitles exist as a third fusion signal (sentinel/stream_moment_detector.py),
on_vod_ingested() enforces this explicit policy instead of leaving it to
silent inference:

    no chat_messages loaded AND no subtitle_cues loaded
        => raise InsufficientSignalError, publish nothing.

This is a deliberate, structural check — independent of how the detector's
weights happen to be configured — NOT an attempt to measure "is the audio
actually weak" at intake time (that would require running detection first).
The alternative (silently falling back to audio-only detection) would
reintroduce a known-inadequate signal set. Content with a real chat channel is entirely unaffected
by this rule regardless of whether subtitles are provided — the rule only
fires on the specific no-chat/no-subtitle combination.

core.event_bus.EventBus.publish() catches and logs handler exceptions
rather than propagating them (core/event_bus.py), so the end-to-end
observable behavior of this rejection is "zero MOMENT_DETECTED events
published, error logged" — not a crash of the surrounding process.

Queueing decision — inline, not a JobQueue
--------------------------------------------
Detection is NOT enqueued on JobQueue (existing or new). Reasoning:

  CLAUDE.md's separate-queue rule for clip-processing vs script-generation
  jobs exists because those two workloads have genuinely different resource
  shapes that contend for the same thermal/GPU budget — clip jobs are
  CPU+GPU+encode across ~3-4 minutes, script jobs are a single LLM call.
  JobQueue's max_size backpressure exists to keep a slow single worker from
  being buried by jobs arriving faster than it can drain them.

  Moment detection has neither property. Per the scale test
  (tests/test_stream_sentinel_scale.py), a full 10-hour VOD detects in a
  small fraction of a second — CPU-only, no GPU, no FFmpeg, no encode, no
  network I/O once chat/audio are loaded. There's no sustained resource
  hold to protect other jobs from, and no slow drain rate to backpressure
  against. Adding a queue+worker here would add real complexity (a second
  asyncio.Task, its own start/stop lifecycle, backpressure tuning) to guard
  against a resource-contention problem this workload doesn't have.

  It is still CPU-bound *synchronous* Python, though (nested loops over up
  to ~36k buckets for a 10h VOD) — running it directly inside the async
  STREAM_VOD_INGESTED handler would block the whole asyncio event loop for
  that duration, stalling unrelated coroutines (Brain's Ollama calls,
  JobQueue's own poll loop) for no good reason. So detect() (and the file
  loads) run via asyncio.to_thread — off the event loop, but still outside
  any queue. If concurrent VOD ingestion volume ever becomes high enough
  that unbounded concurrent detect() calls start competing for CPU with
  clip jobs, revisit this with real numbers; don't pre-build backpressure
  for a load pattern that hasn't been observed.
"""
from __future__ import annotations

import asyncio
import html
import json
import re
from pathlib import Path

from core.config_loader import get_config
from core.event_bus import (
    STREAM_VOD_INGESTED,
    MOMENT_DETECTED,
    EventBus,
    MomentDetectedPayload,
    StreamVodIngestedPayload,
)
from core.logger import get_logger
from sentinel.stream_moment_detector import (
    ChatMessage,
    StreamMomentDetector,
    SubtitleCue,
    redistribute_weights,
)

log = get_logger(__name__)


class InsufficientSignalError(RuntimeError):
    """
    Raised by on_vod_ingested() when a VOD has neither a chat channel nor a
    subtitle track — see the module docstring's CRITICAL PRODUCT DECISION
    section. Deliberately a distinct exception type (not a bare RuntimeError)
    so callers/tests can distinguish "rejected by policy" from any other
    failure mode.
    """


# ── Fixture loaders ───────────────────────────────────────────────────────────


def load_chat_log(path: str | Path) -> list[ChatMessage]:
    """Parse the JSON-Lines chat log format documented in the module docstring."""
    messages: list[ChatMessage] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages.append(ChatMessage(timestamp=float(row["timestamp"]), text=str(row["text"])))
    return messages


def load_audio_track(path: str | Path) -> tuple[list[float], float]:
    """
    Parse the JSON audio-track format documented in the module docstring.
    Returns (dbfs_values, audio_bucket_sec).
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    dbfs = [float(v) for v in data["dbfs"]]
    bucket_sec = float(data.get("bucket_sec", 1.0))
    return dbfs, bucket_sec


# ── Subtitle loader ───────────────────────────────────────────────────────────

# Matches a cue timing line in either .vtt (HH:MM:SS.mmm) or .srt
# (HH:MM:SS,mmm) form, with an optional hour field (some tools emit
# MM:SS.mmm for short cues) and ignoring any trailing cue-settings text
# (.vtt's "align:start position:0%" etc. — .search(), not .match()/
# .fullmatch(), so trailing junk after the timestamp is simply not captured).
_TIMESTAMP_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
# Strips both .vtt's <c>...</c> word-highlight tags and its inline
# per-word timing tags (<00:03:01.959>) — both are just `<...>` to a regex.
_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


def _timestamp_to_seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    h = int(hours) if hours else 0
    return h * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _clean_cue_text(raw_lines: list[str]) -> str:
    joined = " ".join(raw_lines)
    stripped = _TAG_RE.sub("", joined)
    unescaped = html.unescape(stripped)
    return _WHITESPACE_RE.sub(" ", unescaped).strip()


def _dedup_rolling_cues(
    raw_cues: list[tuple[float, float, str]]
) -> list[SubtitleCue]:
    """
    Collapses real-world auto-caption messiness down to one SubtitleCue per
    genuinely NEW chunk of text. YouTube's "roll-up" auto-captions render as
    a 2-line ticker (verified against a real downloaded auto-caption file —
    see the subtitle loader's unit tests):

      1. A "growing" cue's raw text is [previously-settled line] +
         [new line currently being typed word-by-word] — i.e. it always
         starts with whatever line most recently settled.
      2. Once that new line finishes, a near-zero-duration "settle" cue
         repeats ONLY that new line by itself (the old line has scrolled
         off) — this becomes the "previously-settled line" prefix for the
         *next* growing cue.

    So the running comparison must be against the single most-recently-
    settled LINE, not the full multi-line accumulated text — comparing
    against the full text would miss that the settle cue only re-sends the
    newest line, and every settle cue would look like "new" content instead
    of a duplicate. Concretely:

        previous_line = ""
        for each raw cue's text:
            if text == previous_line:            # exact-duplicate settle
                skip — no new content
            elif previous_line and text.startswith(previous_line):
                new_content = text with previous_line's prefix stripped
            else:
                new_content = text                # unrelated line (new
                                                    # sentence, or ordinary
                                                    # non-rolling .srt/.vtt)
            previous_line = new_content            # the settled tail is
                                                    # what the NEXT cue's
                                                    # prefix will match against
            emit new_content (if non-empty) as one SubtitleCue

    A cue that doesn't match the rolling pattern (ordinary, unrelated text —
    the common case for clean hand-authored .srt/.vtt) passes through
    unchanged, since "previous_line" then simply won't prefix-match it.
    """
    cues: list[SubtitleCue] = []
    previous_line = ""
    for start, end, text in raw_cues:
        if text == previous_line:
            continue  # exact-duplicate settle cue — no new content
        if previous_line and text.startswith(previous_line):
            new_content = text[len(previous_line):].strip()
        else:
            new_content = text
        previous_line = new_content
        if new_content:
            cues.append(SubtitleCue(start=start, end=end, text=new_content))
    return cues


def load_subtitle_track(path: str | Path) -> list[SubtitleCue]:
    """
    Parses a .vtt or .srt file into deduplicated SubtitleCue objects.
    Format is detected by content (presence of a timing line), not by file
    extension — both formats share the same "blocks separated by a blank
    line, each with one timing line" structure once header/index lines are
    skipped, so one parser handles both.

    Real auto-caption files are messy (see _dedup_rolling_cues' docstring
    for the rolling-caption patterns this handles) — validated against a
    real downloaded YouTube auto-caption file, not only hand-written mocks
    (tests/test_stream_subtitle_loader.py).
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    raw_cues: list[tuple[float, float, str]] = []
    i, n = 0, len(lines)
    while i < n:
        m = _TIMESTAMP_RE.search(lines[i])
        if not m:
            i += 1
            continue

        start = _timestamp_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _timestamp_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1

        text_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        text = _clean_cue_text(text_lines)
        if text:
            raw_cues.append((start, end, text))
        i += 1  # skip the blank block separator

    return _dedup_rolling_cues(raw_cues)


# ── Detector factory (settings.yaml wiring) ───────────────────────────────────

_FALLBACK_PROFILE_WEIGHTS = {"chat_weight": 0.6, "audio_weight": 0.4, "semantic_weight": 0.0}


def build_moment_detector_from_config(
    profile:        str = "gaming",
    *,
    has_chat:       bool = True,
    has_subtitles:  bool = True,
    config:         dict | None = None,
) -> StreamMomentDetector:
    """
    Builds a StreamMomentDetector from settings.yaml's
    sentinel.stream_moment_detection section: looks up the named profile's
    base fusion weights, applies redistribute_weights() for whichever
    signals are actually active on THIS VOD (has_chat/has_subtitles — audio
    is always active, it defines the timeline), and wires in the two-
    category keyword lists.

    Falls back to the original chat=0.6/audio=0.4/semantic=0.0 weights if
    the named profile — or the whole stream_moment_detection config section
    — is missing, so this works unmodified against a settings.yaml that
    predates this feature.

    Pass an explicit `config` dict (as CLAUDE.md's testing philosophy
    requires — no test should depend on the real settings.yaml file) to
    keep this hermetic in tests; defaults to the real get_config() otherwise.
    """
    cfg = config if config is not None else get_config()
    smd_cfg = cfg.get("sentinel", {}).get("stream_moment_detection", {})

    profiles = smd_cfg.get("profiles", {})
    profile_weights = profiles.get(profile, _FALLBACK_PROFILE_WEIGHTS)

    chat_w, audio_w, semantic_w = redistribute_weights(
        profile_weights.get("chat_weight", 0.0),
        profile_weights.get("audio_weight", 0.0),
        profile_weights.get("semantic_weight", 0.0),
        has_chat=has_chat, has_audio=True, has_semantic=has_subtitles,
    )

    keywords = smd_cfg.get("semantic_keywords", {})
    return StreamMomentDetector(
        chat_weight=chat_w,
        audio_weight=audio_w,
        semantic_weight=semantic_w,
        strong_keywords=keywords.get("strong", []),
        generic_keywords=keywords.get("generic", []),
        strong_keyword_weight=smd_cfg.get("strong_keyword_weight", 3.0),
        generic_keyword_weight=smd_cfg.get("generic_keyword_weight", 1.0),
        # Per-profile fusion rule, not a global: only profiles that declare
        # audio_z_floor (interview does, gaming doesn't) get rectified
        # audio — see stream_moment_detector.py's "Rectified audio" section.
        audio_z_floor=profile_weights.get("audio_z_floor"),
        # Per-profile detection threshold: different archetypes have
        # different achievable composite dynamic ranges (interview content
        # tops out well below gaming's chat-driven spikes), so a profile
        # may declare its own composite_threshold. Absent = the 2.5
        # default, so gaming and pre-existing configs are untouched.
        composite_threshold=profile_weights.get("composite_threshold", 2.5),
    )


# ── Handler ────────────────────────────────────────────────────────────────────


class StreamIngestHandler:
    """
    Subscribes to STREAM_VOD_INGESTED, runs StreamMomentDetector over the
    referenced chat/audio artifacts, and publishes one MOMENT_DETECTED per
    candidate. See module docstring for the on-disk fixture format and the
    inline-vs-queue reasoning.
    """

    def __init__(self, detector: StreamMomentDetector | None = None) -> None:
        self._detector = detector or StreamMomentDetector()
        self._bus: EventBus | None = None

    def register(self, bus: EventBus) -> None:
        """Subscribe to STREAM_VOD_INGESTED on the given bus (mirrors Brain/Factory's register(bus) convention in apex.py)."""
        self._bus = bus
        bus.subscribe_fn(STREAM_VOD_INGESTED, self.on_vod_ingested)
        log.debug("[StreamIngest] StreamIngestHandler registered for STREAM_VOD_INGESTED")

    async def on_vod_ingested(self, payload: StreamVodIngestedPayload) -> list[MomentDetectedPayload]:
        """
        Load chat/audio, run detection, publish MOMENT_DETECTED per
        candidate. Returns the published payloads (useful for tests and for
        callers that don't want to rely solely on the bus side-effect).
        """
        if self._bus is None:
            raise RuntimeError(
                "StreamIngestHandler.register(bus) must be called before "
                "handling STREAM_VOD_INGESTED events"
            )

        chat_messages = await asyncio.to_thread(load_chat_log, payload.chat_log_path)
        audio_dbfs, audio_bucket_sec = await asyncio.to_thread(load_audio_track, payload.audio_track_path)

        subtitle_cues: list[SubtitleCue] = []
        if payload.subtitle_track_path:
            subtitle_cues = await asyncio.to_thread(load_subtitle_track, payload.subtitle_track_path)

        if not chat_messages and not subtitle_cues:
            raise InsufficientSignalError(
                f"vod_id={payload.vod_id}: no chat log and no subtitle track — "
                "audio-only detection is known-inadequate for this content shape "
                "(it may never clear the composite threshold). Provide a subtitle_track_path "
                "or a chat_log_path before ingesting this VOD."
            )

        candidates = await asyncio.to_thread(
            self._detector.detect,
            payload.vod_id,
            chat_messages,
            audio_dbfs,
            audio_bucket_sec,
            subtitle_cues,
        )

        published: list[MomentDetectedPayload] = []
        for c in candidates:
            moment_payload = MomentDetectedPayload(
                vod_id=c.vod_id,
                window_start=c.window_start,
                window_end=c.window_end,
                composite_score=c.composite_score,
                chat_score=c.chat_score,
                audio_score=c.audio_score,
                semantic_score=c.semantic_score,
                top_spam_tokens=list(c.top_spam_tokens),
                detection_method=c.detection_method,
            )
            await self._bus.publish(MOMENT_DETECTED, moment_payload)
            published.append(moment_payload)

        log.info(
            f"[StreamIngest] vod_id={payload.vod_id}: "
            f"{len(published)} moment(s) published"
        )
        return published
