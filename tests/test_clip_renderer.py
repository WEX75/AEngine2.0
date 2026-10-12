"""
tests/test_clip_renderer.py  —  Pillar 3 (Cutter) render queue test
Apex Engine 2.0

Fully-mocked, offline, deterministic — no real ffmpeg binary invoked. The
ffmpeg execution layer is a fully injected fake runner, same DI pattern
as clip_whisper_bridge's extractor/transcriber. The encoder probe is
injectable too (`encoder_lister`), so the h264_vaapi -> h264_amf ->
libx264 decision logic is tested without a real ffmpeg binary or a real
/dev/dri device — only the VAAPI device-existence check touches the real
filesystem, and only via ordinary tempfile paths, never an actual GPU.

Eight things need proving:
  1. Split-screen ffmpeg args assembled correctly from a configured
     facecam bbox.
  2. No-bbox fallback assembles a plain center-crop, no vstack.
  3. .ass file content generated correctly from caption_segments
     (clip-relative timestamps, correct style block, escaped text).
  4. CLIP_RENDER_READY published with correct fields on success.
  5. A failed render doesn't crash the worker or block the job queued
     behind it.
  6. The event loop stays responsive while a (mocked) long render runs —
     the first real test of CLAUDE.md's caveat #1 for a heavy-render
     queue (queue separation alone doesn't deliver concurrency unless the
     job body actually offloads via asyncio.to_thread).
  7. The encoder probe order (h264_vaapi -> h264_amf -> libx264) picks
     correctly in all four combinations of "listed in ffmpeg -encoders"
     x "VA-API device path exists" — VAAPI being LISTED is not enough on
     its own, matching CLAUDE.md's "empirically confirmed, never assumed"
     standard.
  8. The VA-API filter chain applies .ass burn-in BEFORE
     format=nv12,hwupload (subtitle rendering is a CPU filter and cannot
     run on GPU-resident frames), and non-VAAPI encoders get no hwupload
     step at all.

A ninth area (added with the letterbox mode):
  9. The "letterbox" crop_mode: general fit-and-pad from the source's REAL
     probed aspect ratio (wider than 9:16 -> top/bottom bars, narrower ->
     side bars, exact 9:16 -> no bars — one implementation, never
     hardcoded to one bar orientation), captions placed INSIDE the bottom
     bar when it can hold them and falling back to the footage safe-zone
     (with a logged warning) when it can't, strict enum validation of
     crop_mode (typo = loud failure, never a silent default), explicit
     crop_mode taking precedence over the implicit bbox-presence
     selection, and hex validation of the configurable bar color.

Run:
    python tests/test_clip_renderer.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import CLIP_RENDER_READY, ClipScriptReadyPayload, EventBus
from cutter.clip_renderer import (
    _BEAT_HOLD_MAX_GAP_SEC,
    _BEAT_SPLIT_ON_PAUSE_SEC,
    _CAPTION_FONT_NAME,
    _CAPTION_FONT_SIZE,
    _CAPTION_OUTLINE_COLOUR,
    _CAPTION_PRIMARY_COLOUR,
    _CAPTION_SAFE_ZONE_BOTTOM_FRAC,
    _LETTERBOX_MIN_CAPTION_BAR_PX,
    _MAX_WORDS_PER_BEAT,
    _POP_SCALE_IN_MS,
    _POP_SCALE_START_PCT,
    _SLOW_ZOOM_MAX_SCALE,
    _SLOW_ZOOM_OUTPUT_FPS,
    _SLOW_ZOOM_STEP_PER_FRAME,
    _VINTAGE_CONTRAST,
    _VINTAGE_DESATURATION,
    _VINTAGE_GRAIN_NOISE_STRENGTH,
    ClipRenderer,
    _ass_timestamp,
    _build_ass_file,
    _letterbox_layout,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

CAPTION_SEGMENTS = [
    {"id": 0, "start": 100.0, "end": 105.0, "text": "segment zero"},
    {"id": 1, "start": 105.0, "end": 111.0, "text": "segment one"},
    {"id": 2, "start": 111.0, "end": 118.0, "text": "segment two"},
]

# Word-level caption segments, VOD-absolute timestamps — the shape
# clip_whisper_bridge produces with word_timestamps=True and
# ClipScriptGenerator passes through as caption_segments["words"].
WORD_CAPTION_SEGMENTS = [
    {"id": 0, "start": 100.0, "end": 103.0, "text": "this is absolutely insane",
     "words": [
         {"word": "this",       "start": 100.0,  "end": 100.3},
         {"word": "is",         "start": 100.35, "end": 100.5},
         {"word": "absolutely", "start": 100.55, "end": 101.1},
         {"word": "insane",     "start": 101.15, "end": 101.8},
     ]},
    {"id": 1, "start": 103.0, "end": 105.0, "text": "totally unreal",
     "words": [
         {"word": "totally", "start": 103.2, "end": 103.6},
         {"word": "unreal",  "start": 103.7, "end": 104.4},
     ]},
]

# The exact per-beat pop override prefix — built from the NAMED config
# values, not re-hardcoded numbers, so a config change can't silently
# desync tests from production output.
_POP_TAG = (
    f"{{\\fscx{_POP_SCALE_START_PCT}\\fscy{_POP_SCALE_START_PCT}"
    f"\\t(0,{_POP_SCALE_IN_MS},\\fscx100\\fscy100)}}"
)


def _make_clip(vod_id: str = "vod_render_001", segments: list[dict] | None = None) -> ClipScriptReadyPayload:
    segments = segments if segments is not None else CAPTION_SEGMENTS
    return ClipScriptReadyPayload(
        vod_id=vod_id,
        start_segment_id=segments[0]["id"],
        end_segment_id=segments[-1]["id"],
        caption_segments=list(segments),
        hook_reason="test hook reason",
    )


class _FakeRunner:
    """Records every ffmpeg command it's called with; can fail for specific output paths."""

    def __init__(self, should_fail_for: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._should_fail_for = should_fail_for or set()

    def __call__(self, cmd: list[str]) -> None:
        self.calls.append(cmd)
        output_path = cmd[-1]
        if output_path in self._should_fail_for:
            raise RuntimeError(f"simulated ffmpeg failure for {output_path}")


class _SleepingRunner:
    """
    Blocks the calling THREAD for `delay` seconds via a real time.sleep().
    If the renderer correctly dispatches this via asyncio.to_thread, the
    event loop keeps running everything else during that sleep. If it
    doesn't (a synchronous inline call), the whole loop freezes for
    `delay` seconds instead — exactly what test 6 below is built to catch.
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> None:
        self.calls.append(cmd)
        time.sleep(self._delay)


def _make_spy() -> tuple[list[Any], Any]:
    received: list[Any] = []

    async def _spy(payload: Any) -> None:
        received.append(payload)

    return received, _spy


def _fake_lister(encoders_stdout: str):
    """Returns a fake encoder_lister that ignores ffmpeg_bin and always returns the given stdout."""
    def _lister(ffmpeg_bin: str) -> str:
        return encoders_stdout
    return _lister


async def _wait_until(predicate, timeout_iters: int = 200, interval: float = 0.02) -> None:
    for _ in range(timeout_iters):
        if predicate():
            return
        await asyncio.sleep(interval)


# ── 1. Split-screen ffmpeg args assembled correctly from a facecam bbox ──────

def test_split_screen_ffmpeg_args_assembled_correctly() -> None:
    bus = EventBus()
    runner = _FakeRunner()
    bbox = {"x": 1500, "y": 100, "w": 400, "h": 400}

    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264",
            facecam_bboxes={"streamer_a": bbox}, output_dir=tmp_dir,
        )
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_a")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    assert len(runner.calls) == 1
    cmd = runner.calls[0]

    assert cmd[cmd.index("-ss") + 1] == "100.0"     # first caption segment's start
    assert cmd[cmd.index("-t") + 1] == "18.0"       # 118.0 - 100.0
    assert "/videos/src.mp4" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"

    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    assert "crop=400:400:1500:100" in filter_complex, "exact facecam bbox crop must appear verbatim"
    assert "vstack=inputs=2" in filter_complex, "split-screen stack must be present"
    assert "scale=1080:960" in filter_complex, "each half must scale to half the target height"
    assert "ass=" in filter_complex
    print("  [PASS] split-screen ffmpeg args correctly assembled from configured facecam bbox\n")


# ── 2. No configured bbox -> plain center-crop fallback, no split ────────────

def test_no_bbox_fallback_uses_center_crop_no_split() -> None:
    bus = EventBus()
    runner = _FakeRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={}, output_dir=tmp_dir,
        )
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="unknown_streamer")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    assert len(runner.calls) == 1
    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")

    assert "vstack" not in filter_complex, "fallback must not split-screen"
    assert "crop='if(" in filter_complex, "aspect-fit crop expression must be present"
    assert "crop=400" not in filter_complex, "no exact-bbox crop when nothing is configured"
    assert "scale=1080:1920" in filter_complex, "fallback scales straight to the FULL target resolution, not a half"
    print("  [PASS] no configured facecam bbox -> plain center-crop fallback, no split-screen\n")


# ── 3. .ass content generated correctly from caption_segments ────────────────

def test_ass_timestamp_formatting() -> None:
    assert _ass_timestamp(-5.0) == "0:00:00.00"    # clamped, never negative
    assert _ass_timestamp(0.0) == "0:00:00.00"
    assert _ass_timestamp(65.5) == "0:01:05.50"
    assert _ass_timestamp(3661.23) == "1:01:01.23"
    print("  [PASS] ASS timestamp formatting correct, including negative-clamp\n")


def test_ass_file_content_generated_correctly() -> None:
    ass = _build_ass_file(CAPTION_SEGMENTS, clip_start=100.0, width=1080, height=1920)
    print(ass)

    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "[V4+ Styles]" in ass
    assert "Style: Default," in ass

    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == 3
    # Absolute 100.0/105.0/111.0/118.0 shifted to clip-relative (minus clip_start=100.0)
    assert dialogue_lines[0] == "Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,segment zero"
    assert dialogue_lines[1] == "Dialogue: 0,0:00:05.00,0:00:11.00,Default,,0,0,0,,segment one"
    assert dialogue_lines[2] == "Dialogue: 0,0:00:11.00,0:00:18.00,Default,,0,0,0,,segment two"
    print("  [PASS] .ass file content correct: style block + clip-relative Dialogue timestamps\n")


def test_ass_file_escapes_braces_and_newlines() -> None:
    segs = [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello {world}\nline two"}]
    ass = _build_ass_file(segs, clip_start=0.0, width=1080, height=1920)
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")][0]
    print(f"  {dialogue}")
    assert "{world}" not in dialogue, "literal braces would corrupt ASS override-tag parsing"
    assert "(world)" in dialogue
    assert "\\N" in dialogue, "newlines must become ASS's explicit line-break tag"
    print("  [PASS] ASS text escaping: braces neutralized, newlines converted to \\N\n")


# ── 3b. Word-level "pop" captions (word_timestamps=True path) ────────────────
# One Dialogue per word-BEAT (up to _MAX_WORDS_PER_BEAT words, split early on
# a real speech pause), each with a \t scale-in pop, timed from real Whisper
# word timestamps — never full-sentence blocks when word data exists.

def test_word_level_captions_emit_pop_beats_not_sentence_blocks() -> None:
    ass = _build_ass_file(WORD_CAPTION_SEGMENTS, clip_start=100.0, width=1080, height=1920)
    print(ass)
    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]

    # 4 words at max 3/beat -> 2 beats for segment 0; 2 words -> 1 beat for
    # segment 1. NOT one line per segment.
    assert len(dialogue_lines) == 3, (
        f"expected 3 word-beats, got {len(dialogue_lines)} lines — "
        "word-level data must not fall back to sentence blocks"
    )

    # Beat 1: words 1-3, starts at the FIRST word's real timestamp; holds
    # until the next beat starts (gap 0.05s <= hold limit) to avoid flicker.
    assert dialogue_lines[0] == (
        f"Dialogue: 0,0:00:00.00,0:00:01.15,Default,,0,0,0,,{_POP_TAG}this is absolutely"
    )
    # Beat 2: 4th word alone; next beat is 1.4s away (> hold limit), so it
    # ends at its own word's real end, not stretched across the pause.
    assert dialogue_lines[1] == (
        f"Dialogue: 0,0:00:01.15,0:00:01.80,Default,,0,0,0,,{_POP_TAG}insane"
    )
    # Beat 3: segment 1's two words, timed from ITS words' real timestamps
    # (103.2, not the segment's own 103.0 start).
    assert dialogue_lines[2] == (
        f"Dialogue: 0,0:00:03.20,0:00:04.40,Default,,0,0,0,,{_POP_TAG}totally unreal"
    )
    print("  [PASS] word-level captions: per-beat pop Dialogue lines with real word timing\n")


def test_word_beats_split_on_speech_pause() -> None:
    # Two words, well under the per-beat word cap, but separated by a pause
    # longer than _BEAT_SPLIT_ON_PAUSE_SEC — must become two beats, so a
    # caption never sits frozen on screen through a real silence.
    pause = _BEAT_SPLIT_ON_PAUSE_SEC + 0.4
    segs = [{"id": 0, "start": 0.0, "end": 5.0, "text": "hello there",
             "words": [
                 {"word": "hello", "start": 0.0, "end": 0.4},
                 {"word": "there", "start": 0.4 + pause, "end": 0.8 + pause},
             ]}]
    ass = _build_ass_file(segs, clip_start=0.0, width=1080, height=1920)
    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == 2, "a pause longer than the split threshold must break the beat"
    assert "hello" in dialogue_lines[0] and "there" not in dialogue_lines[0]
    assert "there" in dialogue_lines[1]
    print("  [PASS] word beats split on real speech pauses, not only on the word cap\n")


def test_segments_without_words_fall_back_to_block_captions() -> None:
    # Older payloads (or a Whisper run without word timestamps) carry no
    # "words" key — those must keep the original one-Dialogue-per-segment
    # behavior, with no pop animation tag.
    ass = _build_ass_file(CAPTION_SEGMENTS, clip_start=100.0, width=1080, height=1920)
    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == 3
    for line in dialogue_lines:
        assert "\\t(" not in line, "segment-level fallback must not carry the pop animation tag"
    assert dialogue_lines[0].endswith(",segment zero")
    print("  [PASS] word-less segments keep the segment-level block fallback, unanimated\n")


def test_caption_style_uses_named_config_values() -> None:
    ass = _build_ass_file(WORD_CAPTION_SEGMENTS, clip_start=100.0, width=1080, height=1920)
    style_line = [l for l in ass.splitlines() if l.startswith("Style: Default,")][0]
    print(f"  {style_line}")

    assert f",{_CAPTION_FONT_NAME}," in style_line, "style must use the configured open font, not Arial"
    assert _CAPTION_PRIMARY_COLOUR in style_line, "white fill must come from the named config value"
    assert _CAPTION_OUTLINE_COLOUR in style_line, "black outline must come from the named config value"
    # Safe-zone vertical margin derives from frame height via the named
    # fraction — never a fixed pixel count that breaks at other resolutions.
    expected_margin_v = round(1920 * _CAPTION_SAFE_ZONE_BOTTOM_FRAC)
    assert f",{expected_margin_v},1" in style_line, (
        f"MarginV must be height * {_CAPTION_SAFE_ZONE_BOTTOM_FRAC} = {expected_margin_v}"
    )
    assert _BEAT_HOLD_MAX_GAP_SEC > 0 and _MAX_WORDS_PER_BEAT >= 1  # named config values exist and are sane
    print("  [PASS] caption style built from named config values (font/colors/safe-zone margin)\n")


def test_word_text_is_escaped_in_beats() -> None:
    segs = [{"id": 0, "start": 0.0, "end": 2.0, "text": "brace {attack}",
             "words": [{"word": "{attack}", "start": 0.0, "end": 0.5}]}]
    ass = _build_ass_file(segs, clip_start=0.0, width=1080, height=1920)
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")][0]
    assert "{attack}" not in dialogue, "literal braces in word text would corrupt the pop override tag"
    assert "(attack)" in dialogue
    print("  [PASS] word text braces neutralized inside animated beats\n")


# ── 4. CLIP_RENDER_READY published with correct fields ───────────────────────

def test_clip_render_ready_published_with_correct_fields() -> None:
    bus = EventBus()
    received, spy = _make_spy()
    bus.subscribe_fn(CLIP_RENDER_READY, spy)
    runner = _FakeRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(bus=bus, runner=runner, encoder="libx264", facecam_bboxes={}, output_dir=tmp_dir)
        clip = _make_clip(vod_id="vod_fields_001")

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(received))
            finally:
                await renderer.stop()

        asyncio.run(run())

    assert len(received) == 1
    payload = received[0]
    print(f"  vod_id={payload.vod_id} output_path={payload.output_path} "
          f"duration_sec={payload.duration_sec} encoder_used={payload.encoder_used} "
          f"render_seconds={payload.render_seconds:.3f}")

    assert payload.vod_id == "vod_fields_001"
    assert payload.encoder_used == "libx264"
    assert payload.duration_sec == 18.0   # 118.0 - 100.0, the CLIP's own length
    assert payload.render_seconds >= 0.0
    assert payload.output_path.endswith(".mp4")
    assert "vod_fields_001" in payload.output_path
    print("  [PASS] CLIP_RENDER_READY published with correct vod_id/output_path/duration/encoder\n")


# ── 5. A failed render must not block the job queued behind it ───────────────

def test_failed_render_does_not_block_next_queued_job() -> None:
    bus = EventBus()
    received, spy = _make_spy()
    bus.subscribe_fn(CLIP_RENDER_READY, spy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        broken_clip = _make_clip(vod_id="vod_broken")
        healthy_clip = _make_clip(vod_id="vod_healthy")
        broken_output = str(Path(tmp_dir) / "vod_broken_100-118.mp4")
        runner = _FakeRunner(should_fail_for={broken_output})
        renderer = ClipRenderer(bus=bus, runner=runner, encoder="libx264", facecam_bboxes={}, output_dir=tmp_dir)

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(broken_clip, source_path="/videos/a.mp4", streamer_id="s1")
                await renderer.submit_render(healthy_clip, source_path="/videos/b.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(received))
            finally:
                await renderer.stop()

        asyncio.run(run())

    assert len(runner.calls) == 2, f"both jobs must have been attempted, got {len(runner.calls)}"
    assert len(received) == 1, f"expected only the healthy job's CLIP_RENDER_READY, got {len(received)}"
    assert received[0].vod_id == "vod_healthy"
    print("  [PASS] failed render isolated to its own job — queued job behind it still completes\n")


# ── 6. Event loop stays responsive during a long (mocked) render ─────────────

def test_event_loop_stays_responsive_during_render() -> None:
    bus = EventBus()
    runner = _SleepingRunner(delay=0.3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(bus=bus, runner=runner, encoder="libx264", facecam_bboxes={}, output_dir=tmp_dir)
        clip = _make_clip()
        heartbeats: list[float] = []

        async def _heartbeat() -> None:
            for _ in range(25):
                heartbeats.append(time.monotonic())
                await asyncio.sleep(0.02)

        async def run() -> None:
            await renderer.start()
            try:
                heartbeat_task = asyncio.create_task(_heartbeat())
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await heartbeat_task
            finally:
                await renderer.stop()

        asyncio.run(run())

    assert len(runner.calls) == 1, "the render must have actually run during the heartbeat window"
    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    max_gap = max(gaps)
    print(f"  heartbeat count: {len(heartbeats)}, max gap between beats: {max_gap * 1000:.1f}ms (target ~20ms)")

    assert max_gap < 0.15, (
        f"event loop appears to have stalled during the render — max gap between heartbeats "
        f"was {max_gap:.3f}s, expected ~0.02s. A synchronous (non-to_thread) ffmpeg call inside "
        f"the queue worker would block the loop exactly like this — this is the first real test "
        f"of CLAUDE.md's caveat #1 for a heavy-render queue."
    )
    print("  [PASS] event loop stayed responsive (heartbeat kept ~20ms cadence) during a 0.3s mocked render\n")


# ── 7. Encoder probe order: h264_vaapi -> h264_amf -> libx264 ────────────────
# VAAPI must be CONFIRMED, not just listed — both "in ffmpeg -encoders" AND
# "device path exists" have to be true, matching CLAUDE.md's "empirically
# confirmed, never assumed" standard (the same one that caught ROCm not
# actually working for Ollama on this chip).

def test_probe_prefers_vaapi_when_listed_and_device_exists() -> None:
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp_dir, tempfile.NamedTemporaryFile() as fake_device:
        renderer = ClipRenderer(
            bus=bus,
            encoder_lister=_fake_lister("... h264_vaapi ... h264_amf ..."),
            vaapi_device=fake_device.name,
            output_dir=tmp_dir,
        )

        async def run() -> None:
            await renderer.start()
            await renderer.stop()

        asyncio.run(run())
        assert renderer._encoder == "h264_vaapi", "VA-API must be preferred when listed AND its device exists"
    print("  [PASS] both h264_vaapi and h264_amf listed, device exists -> h264_vaapi selected\n")


def test_probe_falls_back_to_amf_when_vaapi_device_missing() -> None:
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus,
            encoder_lister=_fake_lister("... h264_vaapi ... h264_amf ..."),
            vaapi_device="/nonexistent/render/device/path",
            output_dir=tmp_dir,
        )

        async def run() -> None:
            await renderer.start()
            await renderer.stop()

        asyncio.run(run())
        assert renderer._encoder == "h264_amf", (
            "h264_vaapi is LISTED but its configured device path doesn't exist — "
            "listed alone must not be enough to select it"
        )
    print("  [PASS] h264_vaapi listed but device missing -> falls back to h264_amf\n")


def test_probe_falls_back_to_libx264_when_neither_gpu_encoder_listed() -> None:
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus,
            encoder_lister=_fake_lister("... mpeg4 ... aac ..."),   # neither GPU encoder present
            vaapi_device="/dev/dri/renderD128",
            output_dir=tmp_dir,
        )

        async def run() -> None:
            await renderer.start()
            await renderer.stop()

        asyncio.run(run())
        assert renderer._encoder == "libx264"
    print("  [PASS] neither h264_vaapi nor h264_amf listed -> falls back to libx264\n")


def test_probe_selects_amf_when_vaapi_not_listed_at_all() -> None:
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus,
            encoder_lister=_fake_lister("... h264_amf ..."),   # vaapi genuinely absent, not just device-less
            vaapi_device="/dev/dri/renderD128",
            output_dir=tmp_dir,
        )

        async def run() -> None:
            await renderer.start()
            await renderer.stop()

        asyncio.run(run())
        assert renderer._encoder == "h264_amf"
    print("  [PASS] h264_vaapi not listed at all -> h264_amf selected (second-tier fallback)\n")


# ── 8. VA-API filter chain: ass burn-in BEFORE format=nv12,hwupload ──────────

def test_vaapi_filter_chain_applies_ass_before_hwupload() -> None:
    bus = EventBus()
    runner = _FakeRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="h264_vaapi", vaapi_device="/dev/dri/renderD128",
            facecam_bboxes={}, output_dir=tmp_dir,
        )
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    assert len(runner.calls) == 1
    cmd = runner.calls[0]

    assert "-vaapi_device" in cmd
    assert cmd[cmd.index("-vaapi_device") + 1] == "/dev/dri/renderD128"
    assert cmd.index("-vaapi_device") < cmd.index("-i"), "-vaapi_device is a global option, must precede -i"
    assert cmd[cmd.index("-c:v") + 1] == "h264_vaapi"

    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    assert "ass=" in filter_complex
    assert "format=nv12,hwupload" in filter_complex
    assert filter_complex.index("ass=") < filter_complex.index("format=nv12,hwupload"), (
        "subtitle burn-in is a CPU filter and MUST run before frames are handed to the GPU "
        "(format=nv12,hwupload) — reversed order would make ffmpeg fail outright"
    )
    print("  [PASS] VA-API filter chain applies .ass burn-in BEFORE format=nv12,hwupload\n")


def test_non_vaapi_filter_chain_has_no_hwupload_step() -> None:
    bus = EventBus()
    runner = _FakeRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(bus=bus, runner=runner, encoder="libx264", facecam_bboxes={}, output_dir=tmp_dir)
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    cmd = runner.calls[0]
    assert "-vaapi_device" not in cmd, "libx264 must never add -vaapi_device"
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "hwupload" not in filter_complex
    assert "format=nv12" not in filter_complex
    print("  [PASS] non-VAAPI encoder (libx264) has no hwupload step in the filter chain — unchanged behavior\n")


# ── 9. Letterbox mode: general fit-and-pad, bar captions, strict config ──────

class _AssCapturingRunner(_FakeRunner):
    """
    A _FakeRunner that also reads the generated .ass file DURING the ffmpeg
    call (it's deleted right after the render), so tests can assert on the
    caption placement the renderer actually produced.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ass_contents: list[str] = []

    def __call__(self, cmd: list[str]) -> None:
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        match = re.search(r"ass='([^']*)'", filter_complex)
        assert match, "every render must burn in captions via the ass filter"
        ass_path = match.group(1).replace("\\:", ":").replace("\\\\", "\\")
        self.ass_contents.append(Path(ass_path).read_text(encoding="utf-8"))
        super().__call__(cmd)


class _LogCapture(logging.Handler):
    """Collects log records from the renderer's logger — standalone-runnable, no pytest caplog."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _fake_prober(width: int, height: int):
    """A source_prober stand-in returning fixed dimensions — no real ffprobe."""
    def _prober(source_path: str) -> tuple[int, int]:
        return width, height
    return _prober


def _margin_v_from_style(ass: str) -> int:
    style_line = [l for l in ass.splitlines() if l.startswith("Style: Default,")][0]
    # MarginV is the third field from the end (MarginV, Encoding is last, MarginR before)
    return int(style_line.split(",")[-2])


def _run_letterbox_render(
    src_w: int, src_h: int, vod_id: str = "vod_lb_001"
) -> tuple[list[str], str, list[logging.LogRecord]]:
    """
    Renders one clip in letterbox mode with a fake prober reporting
    src_w x src_h. Returns (ffmpeg cmd, generated .ass content, log records).
    """
    bus = EventBus()
    runner = _AssCapturingRunner()
    capture = _LogCapture()
    renderer_log = logging.getLogger("cutter.clip_renderer")
    renderer_log.addHandler(capture)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            renderer = ClipRenderer(
                bus=bus, runner=runner, encoder="libx264",
                # bbox deliberately configured for the same streamer: explicit
                # crop_mode must take precedence over the implicit
                # bbox-presence split selection.
                facecam_bboxes={"streamer_lb": {"x": 0, "y": 0, "w": 400, "h": 400}},
                crop_modes={"streamer_lb": "letterbox"},
                source_prober=_fake_prober(src_w, src_h),
                output_dir=tmp_dir,
            )
            clip = _make_clip(vod_id=vod_id)

            async def run() -> None:
                await renderer.start()
                try:
                    await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_lb")
                    await _wait_until(lambda: bool(runner.calls))
                finally:
                    await renderer.stop()

            asyncio.run(run())
    finally:
        renderer_log.removeHandler(capture)

    assert len(runner.calls) == 1
    return runner.calls[0], runner.ass_contents[0], capture.records


def test_letterbox_wide_source_pads_and_captions_render_in_bottom_bar() -> None:
    # 16:9 source (wider than the 9:16 target) -> top/bottom bars. Also the
    # precedence check: streamer HAS a facecam bbox configured, but the
    # explicit crop_mode=letterbox must win over the implicit split.
    cmd, ass, _ = _run_letterbox_render(1920, 1080)
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")

    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in filter_complex
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x000000" in filter_complex
    assert "vstack" not in filter_complex, "explicit letterbox must override the implicit bbox-presence split"
    assert "crop" not in filter_complex, "letterbox preserves the full frame — no cropping of any kind"

    # 1920x1080 fit into 1080x1920 -> scaled 1080x608 -> 656px bars top and
    # bottom. The caption must center in the bottom BAR, not use the
    # full-bleed safe-zone fraction.
    _, bar_h = _letterbox_layout(1920, 1080, 1080, 1920)
    assert bar_h == 656
    expected_margin = (bar_h - _CAPTION_FONT_SIZE) // 2
    margin_v = _margin_v_from_style(ass)
    print(f"  bottom bar: {bar_h}px, caption MarginV: {margin_v} (safe-zone would be "
          f"{round(1920 * _CAPTION_SAFE_ZONE_BOTTOM_FRAC)})")
    assert margin_v == expected_margin
    assert margin_v != round(1920 * _CAPTION_SAFE_ZONE_BOTTOM_FRAC), (
        "letterbox caption placement must come from the bar geometry, not the safe-zone fraction"
    )
    print("  [PASS] wide source: generic fit-and-pad (top/bottom bars), captions centered in the bottom bar\n")


def test_letterbox_narrow_source_side_bars_fall_back_to_safe_zone() -> None:
    # Source NARROWER than 9:16 (e.g. 800x1920) -> side bars, footage fills
    # the full height. There is no bottom bar to hold captions, so placement
    # must fall back to the footage safe-zone, with a warning.
    cmd, ass, records = _run_letterbox_render(800, 1920, vod_id="vod_lb_narrow")
    filter_complex = cmd[cmd.index("-filter_complex") + 1]

    # SAME generic fit-and-pad expression — no separate hardcoded side-bar path.
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in filter_complex
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x000000" in filter_complex

    side_bar_w, bottom_bar_h = _letterbox_layout(800, 1920, 1080, 1920)
    print(f"  side bars: {side_bar_w}px each, bottom bar: {bottom_bar_h}px")
    assert side_bar_w == 140 and bottom_bar_h == 0

    margin_v = _margin_v_from_style(ass)
    assert margin_v == round(1920 * _CAPTION_SAFE_ZONE_BOTTOM_FRAC), (
        "no bottom bar -> captions must use the footage safe-zone placement"
    )
    warnings = [r for r in records if r.levelno == logging.WARNING and "safe-zone" in r.getMessage()]
    assert warnings, "the bar-too-small fallback must be loudly logged, not silent"
    print(f"  warning logged: {warnings[0].getMessage()}")
    print("  [PASS] narrow source: side bars via the same fit-and-pad, captions fall back to safe-zone with warning\n")


def test_letterbox_exact_ratio_source_gets_no_bars() -> None:
    # Source already at 9:16 -> pad is a no-op, no bars at all.
    cmd, ass, _ = _run_letterbox_render(1080, 1920, vod_id="vod_lb_exact")
    filter_complex = cmd[cmd.index("-filter_complex") + 1]

    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2" in filter_complex, (
        "exact-ratio sources go through the SAME general fit-and-pad (a no-op pad), not a special case"
    )
    side_bar_w, bottom_bar_h = _letterbox_layout(1080, 1920, 1080, 1920)
    assert side_bar_w == 0 and bottom_bar_h == 0, "an exact 9:16 source must produce zero bars"
    assert _margin_v_from_style(ass) == round(1920 * _CAPTION_SAFE_ZONE_BOTTOM_FRAC)
    print("  [PASS] exact 9:16 source: no bars, captions on the footage safe-zone as usual\n")


def test_letterbox_thin_bar_falls_back_to_safe_zone_with_warning() -> None:
    # Source aspect only slightly wider than 9:16 (1080x1800) -> a 60px bar,
    # far below what the caption style needs — captions must NOT be forced
    # into it.
    _, bottom_bar_h = _letterbox_layout(1080, 1800, 1080, 1920)
    print(f"  bottom bar: {bottom_bar_h}px, minimum needed: {_LETTERBOX_MIN_CAPTION_BAR_PX}px")
    assert 0 < bottom_bar_h < _LETTERBOX_MIN_CAPTION_BAR_PX, "fixture must produce a real-but-too-thin bar"

    _, ass, records = _run_letterbox_render(1080, 1800, vod_id="vod_lb_thin")
    margin_v = _margin_v_from_style(ass)
    assert margin_v == round(1920 * _CAPTION_SAFE_ZONE_BOTTOM_FRAC), (
        "a bar too thin for the caption style must fall back to the footage safe-zone placement"
    )
    warnings = [r for r in records if r.levelno == logging.WARNING and "safe-zone" in r.getMessage()]
    assert warnings, "the thin-bar fallback must log a clear warning"
    print(f"  warning logged: {warnings[0].getMessage()}")
    print("  [PASS] too-thin bar: safe-zone fallback with a clear logged warning\n")


def test_unknown_crop_mode_fails_loudly_at_construction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ClipRenderer(
                bus=EventBus(), encoder="libx264", facecam_bboxes={},
                crop_modes={"streamer_x": "letterbocks"},   # typo — must never silently default
                output_dir=tmp_dir,
            )
        except ValueError as exc:
            message = str(exc)
            print(f"  raised: {message}")
            assert "letterbocks" in message and "streamer_x" in message
            assert "letterbox" in message, "the error must list the valid modes"
        else:
            raise AssertionError("an unknown crop_mode must raise ValueError, never fall back to a default")
    print("  [PASS] unknown/typo crop_mode raises ValueError naming the streamer and the valid modes\n")


def test_crop_mode_split_without_bbox_fails_loudly_at_construction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ClipRenderer(
                bus=EventBus(), encoder="libx264", facecam_bboxes={},
                crop_modes={"streamer_x": "split"},   # explicit split demands a bbox to split WITH
                output_dir=tmp_dir,
            )
        except ValueError as exc:
            print(f"  raised: {exc}")
            assert "streamer_x" in str(exc) and "facecam" in str(exc).lower()
        else:
            raise AssertionError("explicit crop_mode=split with no facecam bbox must raise, not degrade silently")
    print("  [PASS] explicit crop_mode=split with no configured bbox raises at construction\n")


def test_explicit_center_crop_overrides_configured_bbox() -> None:
    # Precedence in the other direction: a bbox IS configured, but explicit
    # crop_mode=center_crop must suppress the implicit split selection.
    bus = EventBus()
    runner = _FakeRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264",
            facecam_bboxes={"s1": {"x": 10, "y": 10, "w": 300, "h": 300}},
            crop_modes={"s1": "center_crop"},
            output_dir=tmp_dir,
        )
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    assert "vstack" not in filter_complex, "explicit center_crop must override the implicit bbox-presence split"
    assert "crop=300" not in filter_complex, "the configured bbox must not be used under explicit center_crop"
    assert "scale=1080:1920" in filter_complex
    print("  [PASS] explicit crop_mode=center_crop takes precedence over a configured facecam bbox\n")


def test_letterbox_bar_color_configurable_and_validated() -> None:
    # Custom hex color flows through to the pad filter (normalized to 0xRRGGBB).
    cmd_holder: list[list[str]] = []

    bus = EventBus()
    runner = _FakeRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={},
            crop_modes={"s1": "letterbox"}, letterbox_bar_color="#1a2B3c",
            source_prober=_fake_prober(1920, 1080), output_dir=tmp_dir,
        )
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())
        cmd_holder = runner.calls

    filter_complex = cmd_holder[0][cmd_holder[0].index("-filter_complex") + 1]
    assert "color=0x1A2B3C" in filter_complex, "the configured bar color must reach the pad filter, normalized"
    print("  [PASS] configured bar color '#1a2B3c' reaches the pad filter as 0x1A2B3C\n")

    # Invalid values fail loudly at construction, never fall back to black.
    for bad in ("not_a_color", "#12345", "#GGGGGG"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                ClipRenderer(
                    bus=EventBus(), encoder="libx264", facecam_bboxes={},
                    letterbox_bar_color=bad, output_dir=tmp_dir,
                )
            except ValueError as exc:
                print(f"  '{bad}' raised: {exc}")
            else:
                raise AssertionError(f"invalid bar color {bad!r} must raise ValueError")
    print("  [PASS] invalid bar colors ('not_a_color', '#12345', '#GGGGGG') all raise ValueError\n")


# ── 10. caption_styles / visual_style / slow_zoom (per-streamer settings) ────
# Same per-streamer + strict-enum-validation pattern as crop_modes. Eight
# things need proving:
#   a. Unset defaults preserve today's exact behavior (pop captions, no
#      extra visual filters) — regression parity, not just new-feature checks.
#   b. caption_styles="static" forces block captions even when word data
#      IS present (reuses the word-less fallback path explicitly).
#   c. caption_styles="none" removes the ass filter step from the chain
#      entirely — crop_mode still applies.
#   d. visual_style="vintage_grain" filters (noise/eq) appear BEFORE the
#      ass burn-in step.
#   e. slow_zoom's zoompan filter also appears BEFORE the ass burn-in step.
#   f. caption_styles="static" + visual_style="vintage_grain" compose
#      correctly together on the same clip (independent filter-chain
#      concerns: content of the caption vs. presence of the style stage).
#   g. caption_styles="none" still hwuploads correctly under h264_vaapi
#      (no ass step to order around, hwupload still the last step).
#   h. Unknown/typo caption_style, visual_style, and non-bool slow_zoom
#      all raise ValueError at construction, never silently default.


def test_unset_caption_and_visual_settings_preserve_default_behavior() -> None:
    # No caption_styles/visual_styles/slow_zoom configured at all — must be
    # bit-for-bit the pre-existing behavior: pop captions (word data
    # present), no grain/zoom filters.
    bus = EventBus()
    runner = _AssCapturingRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(bus=bus, runner=runner, encoder="libx264", facecam_bboxes={}, output_dir=tmp_dir)
        clip = _make_clip(vod_id="vod_default_settings", segments=WORD_CAPTION_SEGMENTS)

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="unconfigured_streamer")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    assert "ass=" in filter_complex, "captions must still burn in by default (caption_styles unset -> 'pop')"
    assert "noise=" not in filter_complex and "eq=" not in filter_complex, "no vintage_grain filter unless configured"
    assert "zoompan=" not in filter_complex, "no slow_zoom filter unless configured"
    assert _POP_TAG in runner.ass_contents[0], "unset caption_styles must still produce animated pop beats"
    print("  [PASS] unset caption_styles/visual_style/slow_zoom preserve today's exact default output\n")


def test_caption_style_static_forces_block_captions_with_word_data() -> None:
    bus = EventBus()
    runner = _AssCapturingRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={},
            caption_styles={"streamer_static": "static"}, output_dir=tmp_dir,
        )
        clip = _make_clip(vod_id="vod_static", segments=WORD_CAPTION_SEGMENTS)

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_static")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    ass = runner.ass_contents[0]
    print(ass)
    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == len(WORD_CAPTION_SEGMENTS), (
        "static must produce one block per SEGMENT, not per word-beat, even though word data exists"
    )
    for line in dialogue_lines:
        assert _POP_TAG not in line, "static must never carry the pop animation tag"
    assert dialogue_lines[0].endswith(",this is absolutely insane")
    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    assert "ass=" in filter_complex, "static still burns captions in — only the content/animation changes"
    print("  [PASS] caption_styles='static' forces plain block captions even with word-level data present\n")


def test_caption_style_none_removes_ass_step_from_chain() -> None:
    bus = EventBus()
    runner = _FakeRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={},
            caption_styles={"streamer_none": "none"}, output_dir=tmp_dir,
        )
        clip = _make_clip(vod_id="vod_no_captions")

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_none")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    cmd = runner.calls[0]
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    assert "ass=" not in filter_complex, "caption_styles='none' must remove the ass burn-in step entirely"
    # crop_mode still applies — no bbox configured for this streamer, so the
    # usual center-crop fallback must still be present.
    assert "crop='if(" in filter_complex, "crop_mode must still apply when captions are skipped"
    assert "scale=1080:1920" in filter_complex
    print("  [PASS] caption_styles='none' removes the ass filter step; crop_mode still applies\n")


def test_caption_style_none_with_vaapi_still_hwuploads_without_ass() -> None:
    bus = EventBus()
    runner = _FakeRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="h264_vaapi", vaapi_device="/dev/dri/renderD128",
            facecam_bboxes={}, caption_styles={"s1": "none"}, output_dir=tmp_dir,
        )
        clip = _make_clip()

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="s1")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    assert "ass=" not in filter_complex
    assert "format=nv12,hwupload" in filter_complex, "hwupload must still run even with no ass step to order around"
    print("  [PASS] caption_styles='none' + h264_vaapi: hwupload still applied, no ass step\n")


def test_vintage_grain_filters_ordered_before_caption_burn_in() -> None:
    bus = EventBus()
    runner = _FakeRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={},
            visual_styles={"streamer_vg": "vintage_grain"}, output_dir=tmp_dir,
        )
        clip = _make_clip(vod_id="vod_vintage")

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_vg")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    noise_expr = f"noise=alls={_VINTAGE_GRAIN_NOISE_STRENGTH}:allf=t"
    eq_expr = f"eq=saturation={_VINTAGE_DESATURATION}:contrast={_VINTAGE_CONTRAST}"
    assert noise_expr in filter_complex, "vintage_grain must apply the named-constant noise filter"
    assert eq_expr in filter_complex, "vintage_grain must apply the named-constant desaturation/contrast filter"
    assert filter_complex.index(noise_expr) < filter_complex.index("ass="), (
        "grain/color must be applied to the footage BEFORE captions are burned in"
    )
    print("  [PASS] visual_style='vintage_grain' filters ordered before caption burn-in\n")


def test_slow_zoom_filter_ordered_before_caption_burn_in() -> None:
    bus = EventBus()
    runner = _FakeRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={},
            slow_zoom={"streamer_sz": True}, output_dir=tmp_dir,
        )
        clip = _make_clip(vod_id="vod_slow_zoom")

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_sz")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    print(f"  filter_complex: {filter_complex}")
    zoom_expr = f"min(zoom+{_SLOW_ZOOM_STEP_PER_FRAME},{_SLOW_ZOOM_MAX_SCALE})"
    assert zoom_expr in filter_complex, "slow_zoom must apply the named-constant zoompan expression"
    assert f"fps={_SLOW_ZOOM_OUTPUT_FPS}" in filter_complex
    assert filter_complex.index("zoompan=") < filter_complex.index("ass="), (
        "the Ken-Burns zoom must be applied to the footage BEFORE captions are burned in"
    )
    print("  [PASS] slow_zoom's zoompan filter ordered before caption burn-in\n")


def test_static_captions_and_vintage_grain_compose_together() -> None:
    # Two independent per-streamer settings on the SAME clip: caption
    # CONTENT (static, no pop tag) and the pre-caption visual filter stage
    # (vintage_grain) must both take effect without interfering with each
    # other.
    bus = EventBus()
    runner = _AssCapturingRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        renderer = ClipRenderer(
            bus=bus, runner=runner, encoder="libx264", facecam_bboxes={},
            caption_styles={"streamer_combo": "static"},
            visual_styles={"streamer_combo": "vintage_grain"},
            output_dir=tmp_dir,
        )
        clip = _make_clip(vod_id="vod_combo", segments=WORD_CAPTION_SEGMENTS)

        async def run() -> None:
            await renderer.start()
            try:
                await renderer.submit_render(clip, source_path="/videos/src.mp4", streamer_id="streamer_combo")
                await _wait_until(lambda: bool(runner.calls))
            finally:
                await renderer.stop()

        asyncio.run(run())

    filter_complex = runner.calls[0][runner.calls[0].index("-filter_complex") + 1]
    ass = runner.ass_contents[0]
    print(f"  filter_complex: {filter_complex}")

    noise_expr = f"noise=alls={_VINTAGE_GRAIN_NOISE_STRENGTH}:allf=t"
    assert noise_expr in filter_complex, "vintage_grain filter must still apply alongside caption_styles=static"
    assert filter_complex.index(noise_expr) < filter_complex.index("ass="), "grain must precede caption burn-in"

    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == len(WORD_CAPTION_SEGMENTS), "static caption content must apply alongside vintage_grain"
    for line in dialogue_lines:
        assert _POP_TAG not in line, "static must still suppress the pop animation tag when combined with vintage_grain"
    print("  [PASS] caption_styles='static' + visual_style='vintage_grain' compose correctly on the same clip\n")


def test_unknown_caption_style_fails_loudly_at_construction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ClipRenderer(
                bus=EventBus(), encoder="libx264", facecam_bboxes={},
                caption_styles={"streamer_x": "poppp"},   # typo
                output_dir=tmp_dir,
            )
        except ValueError as exc:
            message = str(exc)
            print(f"  raised: {message}")
            assert "poppp" in message and "streamer_x" in message
            assert "pop" in message and "static" in message and "none" in message
        else:
            raise AssertionError("an unknown caption_style must raise ValueError, never fall back to a default")
    print("  [PASS] unknown/typo caption_style raises ValueError naming the streamer and the valid styles\n")


def test_unknown_visual_style_fails_loudly_at_construction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ClipRenderer(
                bus=EventBus(), encoder="libx264", facecam_bboxes={},
                visual_styles={"streamer_x": "vintage_graiiin"},   # typo
                output_dir=tmp_dir,
            )
        except ValueError as exc:
            message = str(exc)
            print(f"  raised: {message}")
            assert "vintage_graiiin" in message and "streamer_x" in message
            assert "default" in message and "vintage_grain" in message
        else:
            raise AssertionError("an unknown visual_style must raise ValueError, never fall back to a default")
    print("  [PASS] unknown/typo visual_style raises ValueError naming the streamer and the valid styles\n")


def test_non_bool_slow_zoom_fails_loudly_at_construction() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ClipRenderer(
                bus=EventBus(), encoder="libx264", facecam_bboxes={},
                slow_zoom={"streamer_x": "yes"},   # must be a real bool, not a string
                output_dir=tmp_dir,
            )
        except ValueError as exc:
            print(f"  raised: {exc}")
            assert "streamer_x" in str(exc)
        else:
            raise AssertionError("a non-bool slow_zoom value must raise ValueError, never be silently coerced")
    print("  [PASS] non-bool slow_zoom value raises ValueError\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX CUTTER — Pillar 3 Render Queue Test")
    print("=" * 60 + "\n")
    test_split_screen_ffmpeg_args_assembled_correctly()
    test_no_bbox_fallback_uses_center_crop_no_split()
    test_ass_timestamp_formatting()
    test_ass_file_content_generated_correctly()
    test_ass_file_escapes_braces_and_newlines()
    test_word_level_captions_emit_pop_beats_not_sentence_blocks()
    test_word_beats_split_on_speech_pause()
    test_segments_without_words_fall_back_to_block_captions()
    test_caption_style_uses_named_config_values()
    test_word_text_is_escaped_in_beats()
    test_clip_render_ready_published_with_correct_fields()
    test_failed_render_does_not_block_next_queued_job()
    test_event_loop_stays_responsive_during_render()
    test_probe_prefers_vaapi_when_listed_and_device_exists()
    test_probe_falls_back_to_amf_when_vaapi_device_missing()
    test_probe_falls_back_to_libx264_when_neither_gpu_encoder_listed()
    test_probe_selects_amf_when_vaapi_not_listed_at_all()
    test_vaapi_filter_chain_applies_ass_before_hwupload()
    test_non_vaapi_filter_chain_has_no_hwupload_step()
    test_letterbox_wide_source_pads_and_captions_render_in_bottom_bar()
    test_letterbox_narrow_source_side_bars_fall_back_to_safe_zone()
    test_letterbox_exact_ratio_source_gets_no_bars()
    test_letterbox_thin_bar_falls_back_to_safe_zone_with_warning()
    test_unknown_crop_mode_fails_loudly_at_construction()
    test_crop_mode_split_without_bbox_fails_loudly_at_construction()
    test_explicit_center_crop_overrides_configured_bbox()
    test_letterbox_bar_color_configurable_and_validated()
    test_unset_caption_and_visual_settings_preserve_default_behavior()
    test_caption_style_static_forces_block_captions_with_word_data()
    test_caption_style_none_removes_ass_step_from_chain()
    test_caption_style_none_with_vaapi_still_hwuploads_without_ass()
    test_vintage_grain_filters_ordered_before_caption_burn_in()
    test_slow_zoom_filter_ordered_before_caption_burn_in()
    test_static_captions_and_vintage_grain_compose_together()
    test_unknown_caption_style_fails_loudly_at_construction()
    test_unknown_visual_style_fails_loudly_at_construction()
    test_non_bool_slow_zoom_fails_loudly_at_construction()
    print("  All Pillar 3 Cutter render queue tests passed.")
    print("=" * 60 + "\n")
