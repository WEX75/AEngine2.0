"""
cutter/clip_renderer.py — Pillar 3: CLIP_SCRIPT_READY -> rendered vertical clip
Apex Engine 2.0

The last stage of the stream-clipping pipeline. Owns its own bounded
JobQueue for the heavy render shape (CLAUDE.md "JobQueue ownership" —
render is a genuinely different resource shape from Brain's light
LLM-call queue and clip_whisper_bridge's heavy transcription queue; see
"Why Cutter owns its own queue" below):

    CLIP_SCRIPT_READY + source video path + streamer_id
        --> [THIS MODULE'S OWN QUEUE]
            map segment ids -> real timestamps (via caption_segments)
            --> ONE seek-only ffmpeg call: crop+stack (or center-crop
                fallback) + burn in .ass captions, no full-file re-encode
        --> CLIP_RENDER_READY (published by this module)

Why source_path AND streamer_id are explicit parameters
------------------------------------------------------------
Same gap as clip_whisper_bridge.py, for the same reason: CLIP_SCRIPT_READY's
contract in CLAUDE.md is `{vod_id, start_segment_id, end_segment_id,
caption_segments, hook_reason}` — no source_path, no streamer_id. Adding
either would violate "never overload an existing model with unrelated
optional fields." The facecam bbox lookup needs streamer_id and the
render needs source_path, so both come in as explicit submit_render()
arguments, same pattern as clip_whisper_bridge.submit_moment()'s
source_path. `orchestrator/clip_orchestrator.py` is the vod_id ->
(source_path, streamer_id) mapping this used to flag as missing — it
resolves both from its registry before calling submit_render() here.
This module still never subscribes to the EventBus itself; the
orchestrator subscribes and calls in.

One ffmpeg call, not three
-----------------------------
The task's render stages (seek-extract, split-screen crop, caption
burn-in) are assembled into a SINGLE ffmpeg subprocess invocation via one
filter_complex graph, not three sequential ffmpeg passes. Three passes
would mean two extra unnecessary decode/encode cycles on top of the one
that actually matters — directly against CLAUDE.md's "no full-file
re-encode" principle applied to the extracted clip itself, not just the
source VOD. `-ss`/`-t` before `-i` still does the fast input-side seek
so only the flagged window is ever decoded from the source file at all.

Why Cutter owns its own queue, not Brain's or clip_whisper_bridge's
------------------------------------------------------------------------
Per CLAUDE.md's "JobQueue ownership" section, this is the third named
resource shape (light LLM call / heavy CPU transcription / heavy render)
and per that section it's no longer provisional for Cutter specifically —
Cutter's job is real now: seek-extract + vertical-crop + caption render,
FFmpeg only. Sharing Brain's or the transcription queue would
head-of-line-block short jobs behind a multi-second-to-minute render,
exactly what the separate-queues rule exists to prevent.

CRITICAL: every ffmpeg call in the job body goes through asyncio.to_thread
------------------------------------------------------------------------
This is the FIRST heavy-render queue built in this repo, which makes it
the first real test of CLAUDE.md's caveat #1 ("queue separation alone
does not deliver concurrency... synchronous work run directly inside one
queue's worker blocks the SHARED asyncio event loop for every other queue
too"). Every subprocess call here — the encoder probe and the actual
render — is dispatched via asyncio.to_thread, never called inline.
tests/test_clip_renderer.py proves this empirically (not just by code
inspection) with a heartbeat task running concurrently with a
deliberately slow mocked render, same technique clip_whisper_bridge used
to prove queue serialization.

Queue sizing — worker count
------------------------------
ONE worker, same reasoning shape as clip_whisper_bridge's: on the CPU
(libx264) fallback path FFmpeg encode is CPU-heavy (20-23s per CLAUDE.md's
hardware table), and on the GPU paths (h264_vaapi, h264_amf) it's
GPU-touching — the same GPU Ollama's Vulkan backend uses. Concurrent
renders would contend for the same CPU cores (libx264) or the same GPU
encode block (VAAPI/AMF) with no throughput benefit, just contention.
Brief overlap with Brain's GPU-touching Ollama call across DIFFERENT
queues is fine per the Thermal & resource rules ("across jobs... desired
for throughput, not a violation, as long as sustained full-load overlap
isn't happening") — render is short (verified 10.2x realtime via
h264_vaapi on this hardware) so that overlap window is brief by
construction, not sustained.

Queue sizing — max_size
----------------------------
Same upstream-arrival-rate reasoning as clip_whisper_bridge: render jobs
arrive 1:1 with CLIP_SCRIPT_READY events, which arrive 1:1 with
MOMENT_DETECTED candidates — ultimately bounded by the same
tests/test_stream_sentinel_scale.py MAX_CANDIDATES=5 ceiling this whole
pipeline is already sized against. max_size=5 here too. Note render's own
per-job cost (20-23s) is roughly an order of magnitude cheaper than
Whisper's (154-185s), so a full backlog here drains in ~2 minutes worst
case, not ~15 — this queue is less likely to actually reach capacity in
practice than clip_whisper_bridge's, but the sizing principle (bound the
backlog, don't let it grow unbounded) is the same regardless.

Encoder selection
--------------------
Probed ONCE, in start() (not __init__ — probing is a subprocess call and
CLAUDE.md's GPU section says "GPU offload must be empirically confirmed,
never assumed from install logs"; this applies exactly as well to encoder
availability). Explicit `encoder=` constructor arg skips the probe
entirely — useful both for tests and for anyone who's already confirmed
which encoder they want.

Probe order: h264_vaapi -> h264_amf -> libx264. VA-API is preferred over
AMF here specifically because it's the one CONFIRMED working on the
target hardware: a real-hardware smoke test on the AMD Radeon 780M
(gfx1103) verified h264_vaapi actually encodes, at 10.2x realtime — this
mirrors CLAUDE.md's GPU acceleration reality section, where Vulkan (not
ROCm) turned out to be Ollama's actual working GPU path on this same
chip; VAAPI (not AMF) is the FFmpeg equivalent finding. h264_amf stays in
the probe order as a second-tier fallback for other AMD hardware where
VAAPI isn't available but AMF is — it is NOT verified on this specific
chip, kept only for portability to other machines. libx264 (CPU,
universally available) is the final fallback if neither GPU path probes
available. Never CUDA (CLAUDE.md: no NVIDIA anywhere in this stack).

h264_vaapi additionally needs a real DRM render node device — read from
settings.yaml's `cutter.vaapi_device` (default `/dev/dri/renderD128`),
NOT hardcoded in this module, since the device path can differ across
machines/GPUs. The probe checks BOTH that h264_vaapi is listed in
`ffmpeg -encoders` AND that the configured device path actually exists on
disk — encoder-listed-but-no-device is a real failure mode (e.g. wrong
device path in config, or a container/VM without the render node passed
through), and "listed" alone doesn't meet the "empirically confirmed,
never assumed" bar this codebase holds GPU claims to everywhere else.

VA-API filter chain ordering: ass burn-in BEFORE format=nv12,hwupload
------------------------------------------------------------------------
Subtitle rendering (the `ass` filter) is a CPU-side filter — it cannot
run on frames already resident in GPU memory. So when h264_vaapi is
selected, the filter_complex graph runs crop/stack/ass entirely on the
CPU first, and only THEN appends `format=nv12,hwupload` as the last step
before handing frames to the VAAPI encoder. Reverse that order (hwupload
before ass) and ffmpeg fails outright — a CPU filter can't read
GPU-resident frames. For libx264/h264_amf, no hwupload step exists at
all; the filter graph ends right after ass burn-in, same as before VAAPI
support was added.

Rendering modes & precedence (crop_modes)
------------------------------------------
Three modes, selected per streamer via settings.yaml `cutter.crop_modes`:

  * "split"        — facecam bbox on top, center-cropped content below
                     (requires a configured facecam bbox — explicit split
                     with no bbox is a construction-time error, there is
                     nothing to split with).
  * "center_crop"  — plain center-crop of the full frame to 9:16.
  * "letterbox"    — the full source frame preserved (campaign briefs
                     with a mandatory stream overlay that must never be
                     cropped out): scale to fit entirely inside the
                     target canvas at the source's REAL aspect ratio
                     (probed via ffprobe, never assumed 16:9), padding
                     the leftover dimension with solid bars. ONE general
                     fit-and-pad expression handles every case — wider
                     than 9:16 -> top/bottom bars, narrower -> left/right
                     bars, exact 9:16 -> no bars. Bar color comes from
                     `cutter.letterbox_bar_color` (default black),
                     validated as a 6-digit hex value at construction.
                     Compositing an external overlay graphic onto the
                     footage is a SEPARATE feature, deliberately not part
                     of this padding logic.

PRECEDENCE RULE: an explicit `crop_modes` entry for a streamer ALWAYS
wins over the implicit bbox-presence selection below. A streamer with NO
`crop_modes` entry behaves exactly as before this mode existed: bbox
configured -> split, no bbox -> center-crop fallback. `crop_modes` values
are validated STRICTLY against the three-mode enum at construction — an
unknown/typo value raises ValueError immediately, it never silently falls
back to a default.

Letterbox captions: rendered INSIDE the bottom bar (vertically centered
in it), computed from the probed source aspect ratio — not the full-bleed
safe-zone fraction. If the bottom bar is too small to properly hold the
caption style (thin bar from a near-9:16 source, zero bar from an exact
or narrower-than-target source), placement falls back to the footage
safe-zone with a logged warning — captions are never forced into a bar
that can't hold them.

Facecam bbox fallback
--------------------------
No general facecam detector (CLAUDE.md: "don't build a general
facecam-detector until the pipeline is proven on 1-2 real sources" —
explicitly out of scope, not deferred by accident). A streamer_id with no
configured bbox (settings.yaml `cutter.facecam_bboxes`) — and no explicit
`crop_modes` entry — falls back to a plain center-crop of the full frame
to the target 9:16 — no split-screen, not an error. This is a documented,
deliberate degradation, not a silent guess at where a facecam might be.

Word-level "pop" captions (word_timestamps=True path)
------------------------------------------------------------------------
The gap this section used to flag is closed: clip_whisper_bridge.py's
_WhisperTranscriber now calls transcribe() with word_timestamps=True, and
caption_segments carry a "words" list ({word, start, end}, VOD-absolute)
alongside each segment. _build_ass_file renders those as pop-style beats —
1-{max words} words on screen at a time, timed from the real word
timestamps, each scaling in via an ASS \t transform — instead of
full-sentence blocks. Every visual/timing choice is a named config value
at the top of the ASS section (font/size/colours/safe-zone
margin/animation strength/beat pacing), not magic numbers. Segments
without word data (older payloads) still render as the original
unanimated segment-level blocks — fallback, not an error.

Pacing across archetypes (gaming vs interview)
------------------------------------------------------------------------
One config set serves both content archetypes because beat pacing is
partly speech-driven, not purely word-count-driven: beats break at
_MAX_WORDS_PER_BEAT words OR at a real speech pause
(_BEAT_SPLIT_ON_PAUSE_SEC), and only hold across gaps of at most
_BEAT_HOLD_MAX_GAP_SEC. Fast gaming speech fills beats to the cap
(~0.5-0.8s per beat); slower interview delivery produces longer-held
beats and breaks on its own natural pauses instead of freezing a stale
caption through silence. If a campaign brief ever wants
per-archetype pacing (e.g. 2-word beats for very fast streamers),
_MAX_WORDS_PER_BEAT is the single lever — but it is deliberately one
global value until a real brief asks otherwise.

Caption styles, per streamer (`cutter.caption_styles`)
------------------------------------------------------------------------
Same per-streamer + strict-enum-validation pattern as `crop_modes`: "pop"
(default, unchanged — animated word beats when word data exists, block
captions otherwise) | "static" (plain unanimated block captions always,
even when word data IS present — reuses _build_ass_file's existing
word-less fallback path via force_static=True, just exposed as an
explicit choice) | "none" (skip caption/.ass burn-in entirely — some
source content already has captions baked in and must not get more
added; the configured crop_mode still applies, only the ass filter step
is skipped). A typo raises ValueError at construction, same as
crop_modes — never a silent default.

Visual style + slow zoom, per streamer (`cutter.visual_styles`,
`cutter.slow_zoom`)
------------------------------------------------------------------------
"vintage_grain" (subtle film grain via ffmpeg's `noise` filter + slight
desaturation + a gentle contrast lift via `eq`, all named constants —
see _VINTAGE_GRAIN_NOISE_STRENGTH / _VINTAGE_DESATURATION /
_VINTAGE_CONTRAST) and `slow_zoom` (bool, a gentle continuous Ken-Burns
zoom via `zoompan`, `d=1` so it drives real video motion rather than
freezing on a still — see _SLOW_ZOOM_MAX_SCALE / _SLOW_ZOOM_STEP_PER_FRAME
/ _SLOW_ZOOM_OUTPUT_FPS) are independent per-streamer settings, both
applied in ONE combined filter stage BEFORE caption burn-in — so grain/
color/zoom affect only the footage, and captions render fixed on top of
the already-styled frame rather than zooming or graining themselves.
`visual_styles` values are strictly validated at construction like
crop_modes/caption_styles; no entry -> "default" (unchanged output),
`slow_zoom` no entry -> False. These constants are a first-pass
"professional vintage, not degraded" judgment call, not yet eyeballed
against real rendered output — worth a visual review pass before a real
campaign relies on them.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from core.config_loader import get_config
from core.event_bus import (
    CLIP_RENDER_READY,
    ClipRenderReadyPayload,
    ClipScriptReadyPayload,
    EventBus,
    get_bus,
)
from core.job_queue import JobQueue
from core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_TARGET_WIDTH  = 1080
_DEFAULT_TARGET_HEIGHT = 1920

_ENCODER_VAAPI    = "h264_vaapi"    # Preferred — CONFIRMED working on the target AMD 780M (10.2x realtime)
_ENCODER_AMF      = "h264_amf"      # Second-tier fallback — NOT verified on this chip, kept for other AMD hardware
_ENCODER_FALLBACK = "libx264"       # Always available, CPU-only

_DEFAULT_VAAPI_DEVICE = "/dev/dri/renderD128"   # settings.yaml default only — never hardcoded elsewhere in this module

# Rendering modes (settings.yaml `cutter.crop_modes`) — see "Rendering modes
# & precedence" in the module docstring. Validated STRICTLY against this set
# at construction; an unknown value raises, never silently defaults.
_MODE_SPLIT       = "split"
_MODE_CENTER_CROP = "center_crop"
_MODE_LETTERBOX   = "letterbox"
_VALID_CROP_MODES = frozenset({_MODE_SPLIT, _MODE_CENTER_CROP, _MODE_LETTERBOX})

_DEFAULT_BAR_COLOR = "#000000"                  # letterbox bar color default — overridable, validated hex
_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

# Caption styles (settings.yaml `cutter.caption_styles`), per streamer_id —
# same precedence/validation pattern as crop_modes: a typo raises at
# construction, never silently defaults. No entry for a streamer -> "pop"
# (today's exact behavior, unchanged).
#   * "pop"    — word-level animated beats when word data exists, unanimated
#                segment blocks otherwise (existing behavior).
#   * "static" — plain, unanimated segment-block captions ALWAYS, even when
#                word data is present — reuses _build_ass_file's existing
#                word-less fallback path (force_static=True), just exposed
#                as an explicit choice rather than an implicit fallback.
#   * "none"   — no caption/.ass burn-in at all. Some source content already
#                has captions baked in and must not get more added. The
#                configured crop_mode still applies — only the ass filter
#                step is skipped.
_CAPTION_STYLE_POP    = "pop"
_CAPTION_STYLE_STATIC = "static"
_CAPTION_STYLE_NONE   = "none"
_VALID_CAPTION_STYLES = frozenset({_CAPTION_STYLE_POP, _CAPTION_STYLE_STATIC, _CAPTION_STYLE_NONE})

# Visual styles (settings.yaml `cutter.visual_styles`), per streamer_id —
# same strict-enum pattern. No entry -> "default" (unchanged output).
_VISUAL_STYLE_DEFAULT       = "default"
_VISUAL_STYLE_VINTAGE_GRAIN = "vintage_grain"
_VALID_VISUAL_STYLES = frozenset({_VISUAL_STYLE_DEFAULT, _VISUAL_STYLE_VINTAGE_GRAIN})

# "vintage_grain" look: subtle film grain + slight desaturation + a gentle
# contrast lift, applied BEFORE caption burn-in (grain/color affects the
# footage, never the caption text on top of it). Named constants, not magic
# numbers, per this module's existing style-constant convention (see the
# caption style block below). FLAG FOR VISUAL REVIEW: these values are a
# first-pass "professional vintage, not degraded" judgment call, chosen for
# subtlety over a real rendered-output comparison — worth eyeballing on
# real footage before a campaign relies on them.
_VINTAGE_GRAIN_NOISE_STRENGTH = 8        # ffmpeg `noise` filter alls= (0-100 scale); low end for a
                                          # subtle film-grain texture, not visible degradation
_VINTAGE_DESATURATION         = 0.88     # ffmpeg `eq` filter saturation multiplier; 1.0 = unchanged
_VINTAGE_CONTRAST             = 1.08     # ffmpeg `eq` filter contrast multiplier; a subtle lift,
                                          # deliberately short of crushed blacks

# "slow_zoom" (settings.yaml `cutter.slow_zoom`, bool per streamer_id,
# default false): a gentle continuous Ken-Burns-style zoom-in via ffmpeg's
# `zoompan` filter. `d=1` (one output frame per input frame) is what makes
# zoompan drive real video motion instead of freezing on stills. Applied in
# the same pre-caption stage as vintage_grain, so captions render fixed on
# top of the already-zoomed frame rather than zooming themselves.
_SLOW_ZOOM_MAX_SCALE      = 1.08         # caps the zoom at +8% over the clip — gentle, not disorienting
_SLOW_ZOOM_STEP_PER_FRAME = 0.0006       # per-output-frame zoom increment (zoompan's `z` expression)
_SLOW_ZOOM_OUTPUT_FPS     = 30           # zoompan requires an explicit output fps; assumes this
                                          # pipeline's standard vertical-clip delivery frame rate —
                                          # revisit if a real source's native fps differs materially

# See "Queue sizing" in the module docstring for the reasoning behind both.
_DEFAULT_QUEUE_MAX_SIZE = 5

Runner = Callable[[list[str]], None]
Prober = Callable[[str], tuple[int, int]]       # source_path -> (width, height), via ffprobe by default


class ClipRenderError(RuntimeError):
    """Raised when a render fails — bad input, ffmpeg failure, or missing captions."""


# ── ASS subtitle generation ───────────────────────────────────────────────────────

# Caption style — every visual/timing choice is a NAMED value here (same
# easy-to-find-and-edit pattern the original single style-block constant
# used), never a magic number buried in the template string below. A
# campaign brief that wants a different look edits these, nothing else.
#
# "Pop" style: one short word-group ("beat") on screen at a time, timed
# from real Whisper word timestamps, each beat scaling in with a \t
# transform. Segment-level blocks remain only as the fallback for caption
# segments that carry no word data (see _build_ass_file).
_CAPTION_FONT_NAME       = "Anton"        # open font (SIL OFL) — heavy display face built for
                                          # exactly this large-caption use; NOT a proprietary font.
                                          # Must be installed (fontconfig) on the render machine.
_CAPTION_FONT_SIZE       = 112            # at PlayResY=1920; large enough to read on a phone feed
_CAPTION_PRIMARY_COLOUR  = "&H00FFFFFF"   # white fill        (ASS AABBGGRR hex)
_CAPTION_SECONDARY_COLOUR = "&H000000FF"  # karaoke secondary (unused by pop style, required field)
_CAPTION_OUTLINE_COLOUR  = "&H00000000"   # black outline — the high-contrast edge
_CAPTION_BACK_COLOUR     = "&H00000000"   # shadow colour (shadow depth 0 below, so invisible)
_CAPTION_OUTLINE_WIDTH   = 7              # thick enough to hold contrast over bright footage
_CAPTION_SHADOW_DEPTH    = 0              # outline does the contrast work; drop shadow off
_CAPTION_BOLD            = 0              # Anton is already ultra-heavy — synthetic bolding would distort it
_CAPTION_ALIGNMENT       = 2              # numpad 2 = bottom-center anchor
_CAPTION_MARGIN_H        = 60             # px side margins at PlayResX=1080 — keeps long beats off the edges
_CAPTION_SAFE_ZONE_BOTTOM_FRAC = 0.36     # caption anchor sits this fraction of frame height above the
                                          # bottom edge: ~64% down a 9:16 frame — below a split-screen
                                          # seam, above platform UI (progress bar / captions / buttons)
_MAX_WORDS_PER_BEAT      = 3              # 1-3 words visible at a time; the pacing lever (see
                                          # "Pacing across archetypes" in the module docstring)
_BEAT_SPLIT_ON_PAUSE_SEC = 0.8            # a real speech pause longer than this breaks a beat even
                                          # under the word cap — pacing adapts to speech rhythm
_BEAT_HOLD_MAX_GAP_SEC   = 0.6            # hold a beat on screen until the next one starts when the
                                          # gap is at most this — no flicker between beats; longer
                                          # gaps are real silences and the caption clears
_POP_SCALE_START_PCT     = 62             # animation strength: each beat scales in from this % ...
_POP_SCALE_IN_MS         = 120            # ... to 100% over this many ms (subtle, not bouncy)

# Letterbox-mode caption placement (see "Rendering modes & precedence" in the
# module docstring): the smallest bottom bar that can properly hold one
# caption line — the line itself, its outline on both edges, and breathing
# room above and below. A thinner (or absent) bar falls back to the footage
# safe-zone placement, with a logged warning.
_LETTERBOX_BAR_CAPTION_PAD_PX = 20
_LETTERBOX_MIN_CAPTION_BAR_PX = (
    _CAPTION_FONT_SIZE
    + 2 * _CAPTION_OUTLINE_WIDTH
    + 2 * _LETTERBOX_BAR_CAPTION_PAD_PX
)

_ASS_STYLE_BLOCK = """[Script Info]
Title: Apex Engine Clip Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{primary},{secondary},{outline_colour},{back},{bold},0,0,0,100,100,0,0,1,{outline_w},{shadow},{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# The per-beat scale-in override, prefixed to every animated Dialogue line.
_POP_OVERRIDE_TAG = (
    f"{{\\fscx{_POP_SCALE_START_PCT}\\fscy{_POP_SCALE_START_PCT}"
    f"\\t(0,{_POP_SCALE_IN_MS},\\fscx100\\fscy100)}}"
)


def _ass_timestamp(seconds: float) -> str:
    """Formats seconds as ASS's H:MM:SS.CS (centisecond) timestamp format."""
    seconds = max(0.0, seconds)
    total_cs = round(seconds * 100)
    cs, total_s = total_cs % 100, total_cs // 100
    s, total_m = total_s % 60, total_s // 60
    m, h = total_m % 60, total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Neutralize characters that would corrupt ASS parsing/override tags."""
    return text.replace("\n", "\\N").replace("{", "(").replace("}", ")")


def _group_words_into_beats(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Groups one segment's word timings into display beats: a new beat starts
    when the current one is full (_MAX_WORDS_PER_BEAT) or when a real
    speech pause (> _BEAT_SPLIT_ON_PAUSE_SEC between the previous word's
    end and this word's start) breaks the flow. The pause rule is what
    makes one set of config values pace sensibly across archetypes — fast
    gaming speech fills beats to the word cap, slow interview delivery
    breaks on its own natural pauses instead of freezing a stale caption
    through silence.
    """
    beats: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for w in words:
        pause = (w["start"] - current[-1]["end"]) if current else 0.0
        if current and (len(current) >= _MAX_WORDS_PER_BEAT or pause > _BEAT_SPLIT_ON_PAUSE_SEC):
            beats.append(current)
            current = [w]
        else:
            current.append(w)
    if current:
        beats.append(current)
    return beats


def _build_ass_file(
    caption_segments: list[dict[str, Any]],
    clip_start:       float,
    width:            int,
    height:           int,
    margin_v:         int | None = None,
    force_static:     bool = False,
) -> str:
    """
    Builds the full .ass file content from CLIP_SCRIPT_READY's
    caption_segments. Timestamps (segment and word alike) are VOD-absolute
    (see clip_whisper_bridge.py) — shifted back to clip-relative here
    (minus clip_start), since the RENDERED clip's own timeline starts at 0,
    not at wherever the window fell in the original VOD.

    Two rendering modes, decided per segment:

      * words present ("words" key non-empty — the word_timestamps=True
        path) AND force_static is False: "pop"-style beats. Words are
        grouped via _group_words_into_beats(); each beat becomes one
        Dialogue line running from its first word's real start to its last
        word's real end — extended to the next beat's start when the gap is
        at most _BEAT_HOLD_MAX_GAP_SEC, so back-to-back beats never
        flicker, while real silences still clear the screen. Every beat
        line carries the _POP_OVERRIDE_TAG scale-in.
      * no words (older payloads, or a transcriber run without word
        timestamps), OR force_static=True: the original
        one-Dialogue-per-segment block, unanimated — unchanged legacy
        behavior, not an error. `force_static` is caption_styles="static"
        (see ClipRenderer) choosing this path explicitly even when word
        data IS available, rather than it only ever being an implicit
        fallback for word-less payloads.

    `margin_v` overrides the caption's bottom margin (letterbox mode
    places captions inside the bottom bar); None keeps the default
    footage safe-zone placement.
    """
    if margin_v is None:
        margin_v = round(height * _CAPTION_SAFE_ZONE_BOTTOM_FRAC)
    lines = [_ASS_STYLE_BLOCK.format(
        width=width, height=height,
        font=_CAPTION_FONT_NAME, font_size=_CAPTION_FONT_SIZE,
        primary=_CAPTION_PRIMARY_COLOUR, secondary=_CAPTION_SECONDARY_COLOUR,
        outline_colour=_CAPTION_OUTLINE_COLOUR, back=_CAPTION_BACK_COLOUR,
        bold=_CAPTION_BOLD, outline_w=_CAPTION_OUTLINE_WIDTH,
        shadow=_CAPTION_SHADOW_DEPTH, alignment=_CAPTION_ALIGNMENT,
        margin_h=_CAPTION_MARGIN_H, margin_v=margin_v,
    )]

    # Collect (start, end, text, animated) tuples first so beat hold can
    # look across segment boundaries at the true next beat.
    beat_events: list[tuple[float, float, str, bool]] = []
    for seg in caption_segments:
        words = [] if force_static else (seg.get("words") or [])
        if words:
            for beat in _group_words_into_beats(list(words)):
                text = _escape_ass_text(" ".join(str(w["word"]) for w in beat))
                beat_events.append((beat[0]["start"], beat[-1]["end"], text, True))
        else:
            beat_events.append(
                (seg["start"], seg["end"], _escape_ass_text(str(seg["text"])), False)
            )

    for i, (start, end, text, animated) in enumerate(beat_events):
        if animated and i + 1 < len(beat_events):
            next_start = beat_events[i + 1][0]
            if 0.0 <= next_start - end <= _BEAT_HOLD_MAX_GAP_SEC:
                end = next_start
        prefix = _POP_OVERRIDE_TAG if animated else ""
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(start - clip_start)},"
            f"{_ass_timestamp(end - clip_start)},Default,,0,0,0,,{prefix}{text}"
        )
    return "\n".join(lines) + "\n"


# ── ffmpeg filter graph construction ──────────────────────────────────────────────


def _crop_to_aspect_expr(target_w: int, target_h: int) -> str:
    """
    ffmpeg crop filter that takes the largest centered region matching
    target_w:target_h from the input, using iw/ih expressions (no need to
    know the source resolution ahead of time). Caller appends scale=W:H
    to fit exactly. Used for the bottom "content" crop and the no-bbox
    fallback crop.
    """
    ratio = f"{target_w}/{target_h}"
    return (
        f"crop='if(gt(iw/ih,{ratio}),ih*{ratio},iw)':"
        f"'if(gt(iw/ih,{ratio}),ih,iw*{target_h}/{target_w})'"
    )


def _letterbox_layout(src_w: int, src_h: int, target_w: int, target_h: int) -> tuple[int, int]:
    """
    The letterbox fit geometry, mirroring ffmpeg's
    scale=W:H:force_original_aspect_ratio=decrease + centered pad: returns
    (side_bar_width, top_bottom_bar_height) — per side, so a 16:9 source in
    a 9:16 target gives (0, 656). Exactly one of the two is non-zero unless
    the source is already at the target ratio (then both are 0 — no bars).
    """
    scale = min(target_w / src_w, target_h / src_h)
    scaled_w = min(target_w, round(src_w * scale))
    scaled_h = min(target_h, round(src_h * scale))
    return (target_w - scaled_w) // 2, (target_h - scaled_h) // 2


def _letterbox_caption_margin_v(bottom_bar_h: int) -> int | None:
    """
    Bottom caption margin that vertically centers one caption line inside
    the bottom letterbox bar — or None when the bar is too small to
    properly hold the caption style (below _LETTERBOX_MIN_CAPTION_BAR_PX,
    including the zero-height case of side-bar/exact-ratio sources), in
    which case the caller falls back to the footage safe-zone placement.
    """
    if bottom_bar_h < _LETTERBOX_MIN_CAPTION_BAR_PX:
        return None
    return (bottom_bar_h - _CAPTION_FONT_SIZE) // 2


def _normalize_bar_color(color: str) -> str:
    """
    Validates a letterbox bar color as a 6-digit hex value ('#RRGGBB' or
    'RRGGBB') and normalizes it to ffmpeg's 0xRRGGBB form. Raises
    ValueError on anything else — never silently falls back to black.
    """
    match = _HEX_COLOR_RE.match(str(color).strip())
    if not match:
        raise ValueError(
            f"Invalid letterbox_bar_color {color!r} — must be a 6-digit hex value like '#000000'."
        )
    return "0x" + match.group(1).upper()


def _vintage_grain_filter() -> str:
    """
    The "vintage_grain" visual_style: subtle film grain (temporal `noise`)
    + slight desaturation + a gentle contrast lift (`eq`), from the named
    constants above. One filter-chain fragment, applied before caption
    burn-in so grain/color affects only the footage.
    """
    return (
        f"noise=alls={_VINTAGE_GRAIN_NOISE_STRENGTH}:allf=t,"
        f"eq=saturation={_VINTAGE_DESATURATION}:contrast={_VINTAGE_CONTRAST}"
    )


def _slow_zoom_filter(width: int, height: int) -> str:
    """
    The "slow_zoom" Ken-Burns-style continuous zoom-in, from the named
    constants above. `d=1` makes zoompan advance one output frame per input
    frame (real video motion), not freeze on a still.
    """
    return (
        f"zoompan=z='min(zoom+{_SLOW_ZOOM_STEP_PER_FRAME},{_SLOW_ZOOM_MAX_SCALE})':"
        f"d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={_SLOW_ZOOM_OUTPUT_FPS}"
    )


def _build_filter_complex(
    mode:          str,
    facecam_bbox:  dict[str, int] | None,
    target_width:  int,
    target_height: int,
    ass_path:      str | None,
    encoder:       str,
    bar_color:     str = "0x000000",
    visual_style:  str = _VISUAL_STYLE_DEFAULT,
    slow_zoom:     bool = False,
) -> tuple[str, str]:
    """
    Returns (filter_complex, output_pad_label). `mode` is one of
    _VALID_CROP_MODES, already resolved via the precedence rule (see
    _resolve_crop_mode / module docstring).

    "split" (facecam_bbox required): top half = exact bbox crop (facecam),
    bottom half = center-crop of the full frame to fill the remaining
    aspect (content/gameplay) — vstacked to target_width x target_height,
    then captions burned in.

    "center_crop": single center-crop of the full frame straight to
    target_width x target_height, no split, then captions burned in.

    "letterbox": ONE general fit-and-pad — scale to fit entirely inside
    the target at the source's own aspect ratio, then pad the remainder
    with `bar_color` bars, centered. The same expression produces
    top/bottom bars (wider-than-target source), side bars (narrower), or
    no bars (exact ratio) — never hardcoded to one bar orientation.

    `ass_path=None` means caption_styles="none" for this streamer (see
    ClipRenderer._resolve_caption_style): the ass burn-in step is skipped
    entirely, and the crop/stack/pad output (after any visual_style/
    slow_zoom stage) feeds straight into hwupload (VAAPI) or becomes the
    final output pad directly (everything else).

    `visual_style`/`slow_zoom` (vintage_grain / Ken-Burns zoom) run in one
    combined filter stage BEFORE caption burn-in — grain/color/zoom affect
    only the footage, never the caption text drawn on top of it afterward.

    When encoder == h264_vaapi: appends format=nv12,hwupload as the LAST
    step, after ass burn-in — subtitle rendering is a CPU filter and must
    run before frames are handed to the GPU (see module docstring's "VA-API
    filter chain ordering"). No hwupload step for any other encoder.
    """
    half_h = target_height // 2
    has_captions = ass_path is not None
    escaped_ass_path = None
    if has_captions:
        escaped_ass_path = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    if mode == _MODE_SPLIT:
        top = (
            f"[0:v]crop={facecam_bbox['w']}:{facecam_bbox['h']}:"
            f"{facecam_bbox['x']}:{facecam_bbox['y']},scale={target_width}:{half_h}[top]"
        )
        bottom = f"[0:v]{_crop_to_aspect_expr(target_width, half_h)},scale={target_width}:{half_h}[bottom]"
        stack = "[top][bottom]vstack=inputs=2[stacked]"
        parts = [top, bottom, stack]
        source_label = "stacked"
    elif mode == _MODE_LETTERBOX:
        fit_and_pad = (
            f"[0:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color={bar_color}[padded]"
        )
        parts = [fit_and_pad]
        source_label = "padded"
    else:
        crop = f"[0:v]{_crop_to_aspect_expr(target_width, target_height)},scale={target_width}:{target_height}[cropped]"
        parts = [crop]
        source_label = "cropped"

    current_label = source_label
    style_filters: list[str] = []
    if visual_style == _VISUAL_STYLE_VINTAGE_GRAIN:
        style_filters.append(_vintage_grain_filter())
    if slow_zoom:
        style_filters.append(_slow_zoom_filter(target_width, target_height))
    if style_filters:
        parts.append(f"[{current_label}]{','.join(style_filters)}[styled]")
        current_label = "styled"

    if has_captions:
        caption_out = "captioned" if encoder == _ENCODER_VAAPI else "vout"
        parts.append(f"[{current_label}]ass='{escaped_ass_path}'[{caption_out}]")
        current_label = caption_out

    if encoder == _ENCODER_VAAPI:
        parts.append(f"[{current_label}]format=nv12,hwupload[vout]")
        current_label = "vout"

    return ";".join(parts), f"[{current_label}]"


def _build_render_command(
    source_path:   str,
    clip_start:    float,
    duration:      float,
    ass_path:      str | None,
    output_path:   str,
    encoder:       str,
    mode:          str,
    facecam_bbox:  dict[str, int] | None,
    target_width:  int,
    target_height: int,
    bar_color:     str = "0x000000",
    vaapi_device:  str | None = None,
    ffmpeg_bin:    str = "ffmpeg",
    visual_style:  str = _VISUAL_STYLE_DEFAULT,
    slow_zoom:     bool = False,
) -> list[str]:
    """
    Assembles the single ffmpeg command: seek + crop/stack (or fit-and-pad
    in letterbox mode) + optional visual_style/slow_zoom + caption burn-in
    (`ass_path=None` skips it — caption_styles="none") + hwupload for
    h264_vaapi. `-vaapi_device` is a global option and must come before
    `-i` — it sets up the hardware device context `hwupload`/`h264_vaapi`
    both rely on.
    """
    filter_complex, video_label = _build_filter_complex(
        mode, facecam_bbox, target_width, target_height, ass_path, encoder, bar_color,
        visual_style, slow_zoom,
    )

    cmd = [ffmpeg_bin, "-y"]
    if encoder == _ENCODER_VAAPI:
        cmd += ["-vaapi_device", str(vaapi_device)]
    cmd += [
        "-ss", f"{clip_start}",
        "-t", f"{duration}",
        "-i", source_path,
        "-filter_complex", filter_complex,
        "-map", video_label,
        "-map", "0:a?",           # optional — don't fail if the source has no audio stream
        "-c:v", encoder,
        "-c:a", "aac",
        output_path,
    ]
    return cmd


# ── Default runner: real ffmpeg subprocess ────────────────────────────────────────


def _default_run_ffmpeg(cmd: list[str]) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ClipRenderError(f"ffmpeg not found on PATH (tried '{cmd[0]}'). Install ffmpeg to render clips.") from exc
    if result.returncode != 0:
        raise ClipRenderError(f"ffmpeg render failed (exit {result.returncode}): {result.stderr[-2000:]}")


def _list_ffmpeg_encoders(ffmpeg_bin: str = "ffmpeg") -> str:
    """
    Real encoder probe: raw stdout of `ffmpeg -encoders`. One subprocess
    call, checked against by name for both h264_vaapi and h264_amf —
    empirical, never assumed (CLAUDE.md's GPU standard applies to FFmpeg
    encoders exactly as much as it does to Ollama's GPU offload).
    """
    try:
        result = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return result.stdout


def _probe_source_dimensions(source_path: str, ffprobe_bin: str = "ffprobe") -> tuple[int, int]:
    """
    Real source-dimension probe via ffprobe — letterbox mode needs the
    source's ACTUAL aspect ratio (bar geometry drives caption placement),
    never an assumed 16:9. One subprocess call on the first video stream;
    injectable via ClipRenderer's `source_prober` so tests never run it.
    """
    cmd = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
        source_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ClipRenderError(
            f"ffprobe not found on PATH (tried '{ffprobe_bin}') — required for letterbox mode."
        ) from exc
    if result.returncode != 0:
        raise ClipRenderError(
            f"ffprobe failed on '{source_path}' (exit {result.returncode}): {result.stderr[-500:]}"
        )
    try:
        w, h = result.stdout.strip().splitlines()[0].split("x")
        return int(w), int(h)
    except (ValueError, IndexError) as exc:
        raise ClipRenderError(
            f"ffprobe returned unparseable dimensions for '{source_path}': {result.stdout!r}"
        ) from exc


# ── Renderer ───────────────────────────────────────────────────────────────────────


class ClipRenderer:
    """
    Owns a bounded JobQueue for rendering (see module docstring for
    sizing). submit_render() enqueues a job whose body maps segment ids
    to real timestamps via caption_segments, runs one ffmpeg call
    (seek + crop/stack + caption burn-in), and publishes CLIP_RENDER_READY.

    `runner` is injectable so tests never need a real ffmpeg binary — same
    DI pattern as clip_whisper_bridge's extractor/transcriber. `encoder`
    explicit constructor arg skips the probe (h264_vaapi -> h264_amf ->
    libx264) entirely. `encoder_lister` is injectable too, so tests can
    exercise the probe's OWN 3-way decision logic without a real ffmpeg
    binary or a real /dev/dri device.
    """

    def __init__(
        self,
        bus:             EventBus | None = None,
        runner:          Runner   | None = None,
        encoder:         str      | None = None,
        encoder_lister:  Callable[[str], str] | None = None,
        vaapi_device:    str      | None = None,
        facecam_bboxes:  dict[str, dict[str, int]] | None = None,
        crop_modes:      dict[str, str] | None = None,
        letterbox_bar_color: str | None = None,
        caption_styles:  dict[str, str] | None = None,
        visual_styles:   dict[str, str] | None = None,
        slow_zoom:       dict[str, bool] | None = None,
        source_prober:   Prober   | None = None,
        output_dir:      str | Path | None = None,
        queue_max_size:  int = _DEFAULT_QUEUE_MAX_SIZE,
        ffmpeg_bin:      str = "ffmpeg",
    ) -> None:
        self._bus            = bus or get_bus()
        self._runner         = runner or _default_run_ffmpeg
        self._encoder_lister = encoder_lister or _list_ffmpeg_encoders
        self._source_prober  = source_prober or _probe_source_dimensions
        self._ffmpeg_bin     = ffmpeg_bin

        # None = not yet probed; probing happens once in start(), never __init__
        # (probing is a subprocess call — see module docstring's "Encoder
        # selection" section for why that can't happen at construction time).
        self._encoder          = encoder
        self._encoder_explicit = encoder is not None

        if vaapi_device is not None:
            self._vaapi_device = vaapi_device
        else:
            cfg = get_config()
            self._vaapi_device = cfg.get("cutter", {}).get("vaapi_device", _DEFAULT_VAAPI_DEVICE)

        if facecam_bboxes is not None:
            self._facecam_bboxes = facecam_bboxes
        else:
            cfg = get_config()
            self._facecam_bboxes = cfg.get("cutter", {}).get("facecam_bboxes", {}) or {}

        if crop_modes is not None:
            self._crop_modes = crop_modes
        else:
            cfg = get_config()
            self._crop_modes = cfg.get("cutter", {}).get("crop_modes", {}) or {}

        # Strict enum validation, at construction — a typo'd crop_mode must
        # fail loudly and immediately, never silently fall back to a default
        # (module docstring, "Rendering modes & precedence").
        for sid, mode in self._crop_modes.items():
            if mode not in _VALID_CROP_MODES:
                raise ValueError(
                    f"Invalid crop_mode {mode!r} for streamer {sid!r} — must be one of "
                    f"{sorted(_VALID_CROP_MODES)}. Refusing to silently fall back to a default."
                )
            if mode == _MODE_SPLIT and sid not in self._facecam_bboxes:
                raise ValueError(
                    f"crop_mode 'split' explicitly configured for streamer {sid!r}, but no facecam "
                    f"bbox is configured for it (cutter.facecam_bboxes) — there is nothing to split with."
                )

        if letterbox_bar_color is None:
            cfg = get_config()
            letterbox_bar_color = cfg.get("cutter", {}).get("letterbox_bar_color", _DEFAULT_BAR_COLOR)
        self._bar_color = _normalize_bar_color(letterbox_bar_color)   # raises ValueError on non-hex input

        if caption_styles is not None:
            self._caption_styles = caption_styles
        else:
            cfg = get_config()
            self._caption_styles = cfg.get("cutter", {}).get("caption_styles", {}) or {}

        # Strict enum validation, at construction — same pattern as
        # crop_modes above: a typo'd caption_style must fail loudly and
        # immediately, never silently fall back to a default.
        for sid, style in self._caption_styles.items():
            if style not in _VALID_CAPTION_STYLES:
                raise ValueError(
                    f"Invalid caption_style {style!r} for streamer {sid!r} — must be one of "
                    f"{sorted(_VALID_CAPTION_STYLES)}. Refusing to silently fall back to a default."
                )

        if visual_styles is not None:
            self._visual_styles = visual_styles
        else:
            cfg = get_config()
            self._visual_styles = cfg.get("cutter", {}).get("visual_styles", {}) or {}

        for sid, style in self._visual_styles.items():
            if style not in _VALID_VISUAL_STYLES:
                raise ValueError(
                    f"Invalid visual_style {style!r} for streamer {sid!r} — must be one of "
                    f"{sorted(_VALID_VISUAL_STYLES)}. Refusing to silently fall back to a default."
                )

        if slow_zoom is not None:
            self._slow_zoom = slow_zoom
        else:
            cfg = get_config()
            self._slow_zoom = cfg.get("cutter", {}).get("slow_zoom", {}) or {}

        for sid, enabled in self._slow_zoom.items():
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"Invalid slow_zoom value {enabled!r} for streamer {sid!r} — must be a bool "
                    "(true/false), not a string or number."
                )

        if output_dir is not None:
            self._output_dir = Path(output_dir)
        else:
            cfg = get_config()
            self._output_dir = Path(cfg.get("cutter", {}).get("output_dir", "output/clips"))

        self._queue_max_size = queue_max_size
        self._queue           = JobQueue(max_size=queue_max_size)
        self._running          = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Probe the encoder once (unless explicitly set) and start the
        render queue worker. Probe order: h264_vaapi -> h264_amf ->
        libx264 — see module docstring's "Encoder selection" for why
        VAAPI is preferred (confirmed working on the target hardware,
        unlike AMF) and why the VAAPI device path is checked, not just
        the encoder's presence in `ffmpeg -encoders`.
        """
        if self._running:
            log.warning("[ClipRenderer] Already running — ignoring duplicate start().")
            return

        if self._encoder_explicit:
            log.info(f"[ClipRenderer] Encoder explicitly set to '{self._encoder}' — skipping probe.")
        else:
            encoders_output = await asyncio.to_thread(self._encoder_lister, self._ffmpeg_bin)
            vaapi_listed = _ENCODER_VAAPI in encoders_output
            vaapi_device_present = await asyncio.to_thread(os.path.exists, self._vaapi_device)
            vaapi_available = vaapi_listed and vaapi_device_present
            amf_available = _ENCODER_AMF in encoders_output

            if vaapi_available:
                self._encoder = _ENCODER_VAAPI
            elif amf_available:
                self._encoder = _ENCODER_AMF
            else:
                self._encoder = _ENCODER_FALLBACK

            log.info(
                f"[ClipRenderer] Encoder probe: h264_vaapi="
                f"{'available' if vaapi_available else f'NOT available (listed={vaapi_listed}, device_exists={vaapi_device_present} at {self._vaapi_device})'}, "
                f"h264_amf={'available' if amf_available else 'NOT available'} "
                f"— selected '{self._encoder}'."
            )

        await self._queue.start()
        self._running = True
        log.info("[ClipRenderer] Online — render queue started.")

    async def stop(self) -> None:
        """
        Stop this renderer's queue worker. Does NOT wait for queued/running
        renders to finish — call `await renderer.drain()` first if that
        matters (see `drain()`).
        """
        if not self._running:
            return
        await self._queue.stop()
        self._running = False
        log.info("[ClipRenderer] Stopped.")

    async def drain(self) -> None:
        """
        Block until every submit_render() call made so far has finished
        (CLIP_RENDER_READY published, or the failure logged) — this is the
        last stage, so draining this queue means that clip's render work is
        truly done, not just handed off further.
        """
        await self._queue.join()

    # ── Public API ────────────────────────────────────────────────────────────

    async def submit_render(
        self,
        clip:           ClipScriptReadyPayload,
        source_path:    str | Path,
        streamer_id:    str,
        target_width:   int = _DEFAULT_TARGET_WIDTH,
        target_height:  int = _DEFAULT_TARGET_HEIGHT,
    ) -> None:
        """
        Enqueue a render for one CLIP_SCRIPT_READY selection. Returns once
        the job is queued, not once it completes — CLIP_RENDER_READY is
        published later, from inside this queue's single worker, same
        fire-and-forget shape as clip_whisper_bridge.submit_moment() and
        Brain.submit_clip_script_job().

        WHEN THE QUEUE IS FULL: blocks the caller (never raises, never
        silently drops) — same inherited core.job_queue.JobQueue /
        asyncio.Queue.put() semantics documented and empirically confirmed
        in clip_whisper_bridge.py. A WARNING is logged before a call that
        will block.

        Raises
        ------
        ValueError  if caption_segments is empty — checked here,
        synchronously, before the job ever occupies a queue slot; there's
        nothing to render without at least one caption segment.
        """
        if not clip.caption_segments:
            raise ValueError(
                f"CLIP_SCRIPT_READY for vod_id={clip.vod_id} has no caption_segments — nothing to render"
            )

        queue_depth = self._queue.qsize()
        log.info(
            f"[ClipRenderer] Queuing render — vod_id={clip.vod_id} "
            f"segments=[{clip.start_segment_id}, {clip.end_segment_id}] "
            f"streamer_id={streamer_id} (queue depth={queue_depth})"
        )
        if queue_depth >= self._queue_max_size:
            log.warning(
                f"[ClipRenderer] Queue at capacity ({queue_depth}/{self._queue_max_size}) — "
                f"submit_render() for vod_id={clip.vod_id} will BLOCK until a running job "
                f"frees a slot. The render is NOT lost, just waiting."
            )
        await self._queue.enqueue(
            self._render_and_emit, clip, source_path, streamer_id, target_width, target_height
        )

    # ── Job worker ────────────────────────────────────────────────────────────

    async def _render_and_emit(
        self,
        clip:           ClipScriptReadyPayload,
        source_path:    str | Path,
        streamer_id:    str,
        target_width:   int,
        target_height:  int,
    ) -> None:
        """
        Runs inside this renderer's own JobQueue single worker. Renders,
        then emits CLIP_RENDER_READY or logs the failure — same
        error-isolation pattern as every other job body in this codebase:
        one bad render must never take down the worker or block jobs
        queued behind it.
        """
        t0 = time.monotonic()
        try:
            output_path, clip_duration = await self._render(
                clip, source_path, streamer_id, target_width, target_height
            )
        except ClipRenderError as exc:
            log.error(f"[ClipRenderer] vod_id={clip.vod_id}: {exc}")
            return
        except Exception as exc:
            log.exception(f"[ClipRenderer] Unexpected error rendering vod_id={clip.vod_id}: {exc}")
            return

        render_seconds = time.monotonic() - t0
        payload = ClipRenderReadyPayload(
            vod_id=clip.vod_id,
            output_path=str(output_path),
            duration_sec=clip_duration,
            encoder_used=self._encoder,
            render_seconds=render_seconds,
        )
        log.info(
            f"[ClipRenderer] >> Emitting CLIP_RENDER_READY | vod_id={clip.vod_id} | "
            f"output={output_path} | encoder={self._encoder} | render_seconds={render_seconds:.1f}s"
        )
        await self._bus.publish(CLIP_RENDER_READY, payload)

    def _resolve_crop_mode(self, streamer_id: str) -> str:
        """
        The precedence rule (module docstring, "Rendering modes &
        precedence"): an explicit `crop_modes` entry always wins; with no
        entry, the legacy implicit selection applies unchanged — bbox
        configured -> split, no bbox -> center-crop fallback. Explicit
        values were already validated at construction.
        """
        explicit = self._crop_modes.get(streamer_id)
        if explicit is not None:
            return explicit
        return _MODE_SPLIT if streamer_id in self._facecam_bboxes else _MODE_CENTER_CROP

    def _resolve_caption_style(self, streamer_id: str) -> str:
        """No entry -> "pop" (today's exact behavior, unchanged)."""
        return self._caption_styles.get(streamer_id, _CAPTION_STYLE_POP)

    def _resolve_visual_style(self, streamer_id: str) -> str:
        """No entry -> "default" (today's exact behavior, unchanged)."""
        return self._visual_styles.get(streamer_id, _VISUAL_STYLE_DEFAULT)

    def _resolve_slow_zoom(self, streamer_id: str) -> bool:
        """No entry -> False (today's exact behavior, unchanged)."""
        return self._slow_zoom.get(streamer_id, False)

    async def _render(
        self,
        clip:           ClipScriptReadyPayload,
        source_path:    str | Path,
        streamer_id:    str,
        target_width:   int,
        target_height:  int,
    ) -> tuple[Path, float]:
        """
        Builds the .ass file, assembles the ffmpeg command, and runs it via
        asyncio.to_thread (never inline — see module docstring's CRITICAL
        note; the letterbox ffprobe call goes through to_thread for the
        same reason). Returns (output_path, clip_duration_sec).
        """
        segments = sorted(clip.caption_segments, key=lambda s: s["id"])
        clip_start = segments[0]["start"]
        clip_end   = segments[-1]["end"]
        duration   = clip_end - clip_start
        if duration <= 0:
            raise ClipRenderError(
                f"invalid clip duration for vod_id={clip.vod_id}: start={clip_start} end={clip_end}"
            )

        mode = self._resolve_crop_mode(streamer_id)
        facecam_bbox = self._facecam_bboxes.get(streamer_id) if mode == _MODE_SPLIT else None
        caption_style = self._resolve_caption_style(streamer_id)
        visual_style = self._resolve_visual_style(streamer_id)
        slow_zoom = self._resolve_slow_zoom(streamer_id)

        # Letterbox caption-bar geometry is only meaningful when captions
        # are actually being burned in — caption_styles="none" skips it
        # entirely (nothing to place, and the "falls back to safe-zone"
        # warning below would be misleading with no captions in the output).
        margin_v: int | None = None
        if mode == _MODE_LETTERBOX and caption_style != _CAPTION_STYLE_NONE:
            src_w, src_h = await asyncio.to_thread(self._source_prober, str(source_path))
            side_bar_w, bottom_bar_h = _letterbox_layout(src_w, src_h, target_width, target_height)
            margin_v = _letterbox_caption_margin_v(bottom_bar_h)
            if margin_v is None:
                log.warning(
                    f"[ClipRenderer] vod_id={clip.vod_id}: letterbox bottom bar is {bottom_bar_h}px "
                    f"(source {src_w}x{src_h} -> side bars {side_bar_w}px), below the "
                    f"{_LETTERBOX_MIN_CAPTION_BAR_PX}px the caption style needs — falling back to "
                    f"the footage safe-zone caption placement instead of forcing captions into the bar."
                )

        mode_detail = {
            _MODE_SPLIT:       f"split-screen, facecam bbox={facecam_bbox}",
            _MODE_CENTER_CROP: "center-crop",
            _MODE_LETTERBOX:   f"letterbox, bar color={self._bar_color}",
        }[mode]
        log.info(
            f"[ClipRenderer] vod_id={clip.vod_id}: rendering [{clip_start:.1f}s, {clip_end:.1f}s) "
            f"({mode_detail}, caption_style={caption_style}, visual_style={visual_style}, "
            f"slow_zoom={slow_zoom}) via {self._encoder}"
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / f"{clip.vod_id}_{int(clip_start)}-{int(clip_end)}.mp4"

        ass_path: str | None = None
        try:
            if caption_style != _CAPTION_STYLE_NONE:
                fd, ass_path = tempfile.mkstemp(suffix=".ass", prefix="apex_clip_captions_")
                os.close(fd)
                with open(ass_path, "w", encoding="utf-8") as f:
                    f.write(_build_ass_file(
                        segments, clip_start, target_width, target_height, margin_v,
                        force_static=(caption_style == _CAPTION_STYLE_STATIC),
                    ))

            cmd = _build_render_command(
                str(source_path), clip_start, duration, ass_path, str(output_path),
                self._encoder, mode, facecam_bbox, target_width, target_height,
                self._bar_color, self._vaapi_device, self._ffmpeg_bin,
                visual_style, slow_zoom,
            )
            await asyncio.to_thread(self._runner, cmd)
            return output_path, duration
        finally:
            if ass_path is not None:
                try:
                    os.remove(ass_path)
                except OSError:
                    pass
