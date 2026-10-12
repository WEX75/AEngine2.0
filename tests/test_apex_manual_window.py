"""
tests/test_apex_manual_window.py — apex.py's --render-manual-window CLI, offline/mock-first
Apex Engine 2.0

Fully-mocked, offline, deterministic — no real ffmpeg binary, no real
Whisper model load, no live Ollama call (there IS no Ollama call on this
path — that's the point). Unlike run_ingest_vod() (see test_apex_ingest.py's
own docstring: deliberately NOT deep-mocked, a wiring function),
run_render_manual_window() is fully DI'd (extractor/transcriber/renderer
all injectable), so it IS exercised end-to-end here with fakes standing in
for ffmpeg/Whisper/ClipRenderer.

This is a genuinely separate code path from --ingest-vod: no
StreamIngestHandler, no StreamMomentDetector, no Brain, no
ClipWhisperBridge, no ClipOrchestrator registry. Seven things need proving:

  1. _resolve_manual_video_path() is a dedicated helper, not a reuse of
     --ingest-vod's own _resolve_ingest_paths() (whose error text
     hardcodes "--ingest-vod" — reusing it here would misname the flag in
     the error message).
  2. _resolve_manual_window() — the pure clamp/validation helper: default
     range preserved when unset, per-streamer override applied, malformed
     override surfaces its ValueError, too-short raises, too-long
     hard-clamps (with a logged warning), nonsensical window raises.
  3. run_render_manual_window() builds the ClipScriptReadyPayload directly
     from real (fake, injected) transcript segments — no Brain/LLM call
     anywhere in this path — and submits it to the real ClipRenderer
     interface via submit_render(), then drains.
  4. Zero transcribed segments (silent window) renders with an empty
     caption, not an error.
  5. The window clamp happens BEFORE extraction — a per-streamer
     target_duration_range override changes what actually gets extracted
     and rendered, not just what's logged.
  6. A window that fails validation never touches the extractor/
     transcriber/renderer at all (fails fast, before any I/O).
  7. A stuck render surfaces as a timeout (return code 1), not a hang.

Run:
    python tests/test_apex_manual_window.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import apex
from brain.clip_script_generator import TranscriptSegment, WordTiming

# ── Fakes ──────────────────────────────────────────────────────────────────────


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, source_path: str, window_start: float, window_end: float, output_path: str) -> None:
        self.calls.append((source_path, window_start, window_end, output_path))


class _FakeTranscriber:
    def __init__(self, segments: list[TranscriptSegment] | None = None) -> None:
        self._segments = segments if segments is not None else [
            TranscriptSegment(id=0, start=0.0, end=4.0, text="clip relative zero",
                               words=(WordTiming(word="clip", start=0.0, end=1.0),
                                      WordTiming(word="relative", start=1.0, end=2.5),
                                      WordTiming(word="zero", start=2.5, end=4.0))),
            TranscriptSegment(id=1, start=4.0, end=9.0, text="clip relative one"),
        ]
        self.calls: list[str] = []

    def __call__(self, audio_path: str) -> list[TranscriptSegment]:
        self.calls.append(audio_path)
        return list(self._segments)


class _FakeRenderer:
    def __init__(self, drain_delay: float = 0.0) -> None:
        self.started = False
        self.stopped = False
        self.calls: list[tuple[Any, str, str]] = []
        self._drain_delay = drain_delay

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def submit_render(self, clip, source_path, streamer_id, **kwargs) -> None:
        self.calls.append((clip, source_path, streamer_id))

    async def drain(self) -> None:
        if self._drain_delay:
            await asyncio.sleep(self._drain_delay)


# ── 0. _resolve_manual_video_path ─────────────────────────────────────────────
# A dedicated helper, deliberately NOT a reuse of --ingest-vod's own
# _resolve_ingest_paths() — that function's error text hardcodes
# "--ingest-vod", which would be a misleading message under this flag.

def test_resolve_manual_video_path_valid_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video = Path(tmp_dir) / "clip.mp4"
        video.write_bytes(b"fake")
        resolved = apex._resolve_manual_video_path(str(video))
        assert resolved == video.resolve()
    print("  [PASS] a valid video path resolves correctly\n")


def test_resolve_manual_video_path_missing_raises_with_correct_flag_name() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        missing = Path(tmp_dir) / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError, match="--render-manual-window"):
            apex._resolve_manual_video_path(str(missing))
    print("  [PASS] a missing path raises FileNotFoundError naming --render-manual-window, not --ingest-vod\n")


# ── 1. _resolve_manual_window ─────────────────────────────────────────────────


def test_resolve_manual_window_default_range_when_unset() -> None:
    start, end = apex._resolve_manual_window("streamer_a", 10.0, 30.0, config={})
    assert (start, end) == (10.0, 30.0), "20s window inside the (12, 75) default must pass through unchanged"
    print("  [PASS] no target_duration_ranges configured -> the (12.0, 75.0) default applies, window untouched\n")


def test_resolve_manual_window_invalid_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="end .* must be greater than start"):
        apex._resolve_manual_window("streamer_a", 30.0, 10.0, config={})
    print("  [PASS] end <= start raises ValueError before any target_duration_range check\n")


def test_resolve_manual_window_too_short_raises() -> None:
    # 3s window, default minimum is 12s.
    with pytest.raises(ValueError, match="shorter than streamer"):
        apex._resolve_manual_window("streamer_a", 10.0, 13.0, config={})
    print("  [PASS] a window shorter than target_duration_range's minimum raises, never silently extended\n")


def test_resolve_manual_window_too_long_hard_clamps() -> None:
    # 100s window, default max is 75s -> must clamp end to start + 75.
    start, end = apex._resolve_manual_window("streamer_a", 10.0, 110.0, config={})
    assert start == 10.0
    assert end == 85.0, f"expected end clamped to start(10)+max(75)=85, got {end}"
    print(f"  [PASS] overlong window hard-clamped: (10.0, 110.0) -> (10.0, {end})\n")


def test_resolve_manual_window_per_streamer_override_applied() -> None:
    cfg = {"stream_clipping": {"target_duration_ranges": {"streamer_tight": [10.0, 20.0]}}}

    # Configured streamer: its own (10, 20) range applies — 25s is too long, clamps to 30.
    start, end = apex._resolve_manual_window("streamer_tight", 10.0, 35.0, config=cfg)
    assert (start, end) == (10.0, 30.0), f"expected clamp to streamer_tight's own max (20s), got ({start}, {end})"

    # A DIFFERENT, unconfigured streamer in the same config still gets the global default —
    # a 60s window is fine under (12, 75) but would have been clamped under streamer_tight's (10, 20).
    start2, end2 = apex._resolve_manual_window("streamer_unconfigured", 10.0, 70.0, config=cfg)
    assert (start2, end2) == (10.0, 70.0), "unconfigured streamer must use the global default, not another streamer's override"
    print("  [PASS] per-streamer target_duration_range override resolved correctly; other streamers keep the default\n")


def test_resolve_manual_window_malformed_override_raises() -> None:
    cfg = {"stream_clipping": {"target_duration_ranges": {"streamer_bad": [75.0, 12.0]}}}  # min > max
    with pytest.raises(ValueError, match="streamer_bad"):
        apex._resolve_manual_window("streamer_bad", 10.0, 30.0, config=cfg)
    print("  [PASS] a malformed per-streamer target_duration_range entry surfaces its ValueError\n")


# ── 2/3/4/5. run_render_manual_window() end to end, fully DI'd ───────────────


def test_manual_window_builds_payload_from_transcript_with_no_brain_or_selection() -> None:
    extractor = _FakeExtractor()
    transcriber = _FakeTranscriber()
    renderer = _FakeRenderer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "src.mp4"
        video_path.write_bytes(b"fake")

        exit_code = asyncio.run(apex.run_render_manual_window(
            video_path=video_path, streamer_id="streamer_a", start=100.0, end=120.0,
            vod_id="vod_manual", extractor=extractor, transcriber=transcriber, renderer=renderer,
        ))

    assert exit_code == 0
    assert extractor.calls == [(str(video_path), 100.0, 120.0, extractor.calls[0][3])], (
        "extractor must be called with EXACTLY the given window — no transcribe margin on this path"
    )
    assert len(transcriber.calls) == 1
    assert renderer.started and renderer.stopped
    assert len(renderer.calls) == 1

    clip, source_path, streamer_id = renderer.calls[0]
    assert source_path == str(video_path)
    assert streamer_id == "streamer_a"
    assert clip.vod_id == "vod_manual"
    assert "Manual window" in clip.hook_reason
    assert clip.start_segment_id == 0 and clip.end_segment_id == 1, "must span every recovered segment — no LLM id selection"

    # Whisper's own timestamps are clip-relative (start at 0) — must be
    # shifted by +window_start (100.0) to VOD-absolute, same convention as
    # clip_whisper_bridge.py, segment AND word timestamps alike.
    assert clip.caption_segments[0] == {
        "id": 0, "start": 100.0, "end": 104.0, "text": "clip relative zero",
        "words": [
            {"word": "clip",     "start": 100.0, "end": 101.0},
            {"word": "relative", "start": 101.0, "end": 102.5},
            {"word": "zero",     "start": 102.5, "end": 104.0},
        ],
    }
    assert clip.caption_segments[1]["start"] == 104.0 and clip.caption_segments[1]["end"] == 109.0
    print("  [PASS] real transcript (fake extractor/transcriber) -> correct payload, no Brain/LLM selection involved\n")


def test_manual_window_zero_segments_renders_with_no_captions() -> None:
    extractor = _FakeExtractor()
    transcriber = _FakeTranscriber(segments=[])   # silent window
    renderer = _FakeRenderer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "silent.mp4"
        video_path.write_bytes(b"fake")

        exit_code = asyncio.run(apex.run_render_manual_window(
            video_path=video_path, streamer_id="streamer_a", start=50.0, end=70.0,
            vod_id="vod_silent", extractor=extractor, transcriber=transcriber, renderer=renderer,
        ))

    assert exit_code == 0, "a silent window must still render — not an error"
    assert len(renderer.calls) == 1
    clip = renderer.calls[0][0]
    assert clip.caption_segments == [{"id": 0, "start": 50.0, "end": 70.0, "text": "", "words": []}]
    print("  [PASS] zero transcribed segments -> renders with one empty caption spanning the window, no error\n")


def test_manual_window_clamp_reaches_extraction_and_render() -> None:
    extractor = _FakeExtractor()
    transcriber = _FakeTranscriber()
    renderer = _FakeRenderer()
    cfg = {"stream_clipping": {"target_duration_ranges": {"streamer_tight": [10.0, 20.0]}}}

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "src.mp4"
        video_path.write_bytes(b"fake")

        # Ask for a 50s window; streamer_tight's max is 20s.
        exit_code = asyncio.run(apex.run_render_manual_window(
            video_path=video_path, streamer_id="streamer_tight", start=100.0, end=150.0,
            vod_id="vod_clamped", extractor=extractor, transcriber=transcriber, renderer=renderer,
            config=cfg,
        ))

    assert exit_code == 0
    assert len(extractor.calls) == 1
    _src, ext_start, ext_end, _out = extractor.calls[0]
    assert (ext_start, ext_end) == (100.0, 120.0), (
        f"the clamp must apply BEFORE extraction — expected (100.0, 120.0), got ({ext_start}, {ext_end})"
    )
    print("  [PASS] the per-streamer target_duration_range clamp reaches extraction/render, not just logging\n")


def test_manual_window_invalid_window_never_touches_extractor_or_renderer() -> None:
    extractor = _FakeExtractor()
    transcriber = _FakeTranscriber()
    renderer = _FakeRenderer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "src.mp4"
        video_path.write_bytes(b"fake")

        async def run() -> None:
            with pytest.raises(ValueError, match="shorter than streamer"):
                await apex.run_render_manual_window(
                    video_path=video_path, streamer_id="streamer_a", start=100.0, end=102.0,   # 2s, too short
                    vod_id="vod_bad", extractor=extractor, transcriber=transcriber, renderer=renderer,
                )

        asyncio.run(run())

    assert extractor.calls == [], "an invalid/too-short window must fail before ANY extraction is attempted"
    assert transcriber.calls == [], "must fail before transcription too"
    assert renderer.calls == [] and not renderer.started, "must fail before the renderer is ever touched"
    print("  [PASS] a too-short window fails fast — no extractor/transcriber/renderer calls made\n")


def test_manual_window_stuck_render_times_out_and_returns_1() -> None:
    extractor = _FakeExtractor()
    transcriber = _FakeTranscriber()
    renderer = _FakeRenderer(drain_delay=5.0)   # far longer than the drain_timeout below

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "src.mp4"
        video_path.write_bytes(b"fake")

        exit_code = asyncio.run(apex.run_render_manual_window(
            video_path=video_path, streamer_id="streamer_a", start=100.0, end=120.0,
            vod_id="vod_stuck", extractor=extractor, transcriber=transcriber, renderer=renderer,
            drain_timeout=0.1,
        ))

    assert exit_code == 1, "a stuck render must surface as return code 1, not hang or raise"
    assert renderer.stopped, "the renderer must still be stopped in the finally block after a timeout"
    print("  [PASS] a stuck render times out cleanly and returns exit code 1\n")


# ── 6. Confirm total independence from --ingest-vod ──────────────────────────


def test_manual_window_path_does_not_reuse_or_mutate_ingest_vod_state() -> None:
    """
    run_render_manual_window and run_ingest_vod share no module-level
    mutable state and neither calls the other — confirmed here by running
    a manual-window render first and then confirming _resolve_ingest_paths
    (the --ingest-vod path's own pure helper) still behaves exactly as its
    own dedicated test file expects, completely unaffected.
    """
    extractor = _FakeExtractor()
    transcriber = _FakeTranscriber()
    renderer = _FakeRenderer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / "src.mp4"
        video_path.write_bytes(b"fake")

        asyncio.run(apex.run_render_manual_window(
            video_path=video_path, streamer_id="streamer_a", start=100.0, end=120.0,
            vod_id="vod_independent", extractor=extractor, transcriber=transcriber, renderer=renderer,
        ))

        # _resolve_ingest_paths must still work normally afterward — no shared state leaked.
        resolved_video, chat_log, subtitle = apex._resolve_ingest_paths(str(video_path), None, None)
        assert resolved_video == video_path.resolve()
        assert chat_log is None and subtitle is None
    print("  [PASS] --render-manual-window shares no state with --ingest-vod's own pure helpers\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX — apex.py --render-manual-window CLI test")
    print("=" * 60 + "\n")
    test_resolve_manual_video_path_valid_file()
    test_resolve_manual_video_path_missing_raises_with_correct_flag_name()
    test_resolve_manual_window_default_range_when_unset()
    test_resolve_manual_window_invalid_end_before_start_raises()
    test_resolve_manual_window_too_short_raises()
    test_resolve_manual_window_too_long_hard_clamps()
    test_resolve_manual_window_per_streamer_override_applied()
    test_resolve_manual_window_malformed_override_raises()
    test_manual_window_builds_payload_from_transcript_with_no_brain_or_selection()
    test_manual_window_zero_segments_renders_with_no_captions()
    test_manual_window_clamp_reaches_extraction_and_render()
    test_manual_window_invalid_window_never_touches_extractor_or_renderer()
    test_manual_window_stuck_render_times_out_and_returns_1()
    test_manual_window_path_does_not_reuse_or_mutate_ingest_vod_state()
    print("  All --render-manual-window tests passed.")
    print("=" * 60 + "\n")
