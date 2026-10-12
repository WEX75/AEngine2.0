"""
tests/test_apex_ingest.py — apex.py's --ingest-vod CLI, offline/mock-first
Apex Engine 2.0

apex.py's run_ingest_vod() itself is deliberately NOT deep-mocked here —
it's a wiring function (constructs real Brain/ClipWhisperBridge/
ClipRenderer/ClipOrchestrator/StreamIngestHandler and publishes a real
event), same category as main() itself, which this codebase has never
unit-tested directly either (see tests/test_clip_orchestrator.py's own
note: only the pure _stream_clipping_enabled() helper is tested, not
main()). Consistent with that established pattern, this file tests the
pieces that were actually extracted to be pure/testable:

  1. _resolve_ingest_paths()      — path validation, no side effects beyond
                                     existence checks.
  2. _build_ingest_detector()     — chat-log-present vs audio-only weight
                                     selection.
  3. _drain_ingest_pipeline()     — the load-bearing bridge -> brain ->
                                     renderer drain ORDER, which is exactly
                                     the new behavior this task added.
                                     Fully testable via DI with fakes,
                                     no mocking of apex.py's internal lazy
                                     imports required.

Each pillar class's OWN behavior (Brain, ClipWhisperBridge, ClipRenderer,
ClipOrchestrator, StreamIngestHandler, sentinel.audio_extractor) already
has its own dedicated mock-first test file — not re-proven here.

Run:
    python tests/test_apex_ingest.py
(also collectable by pytest)
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import apex

# ── 1. _resolve_ingest_paths ──────────────────────────────────────────────────
# Uses tempfile.TemporaryDirectory directly (not pytest's tmp_path fixture) so
# every test in this file runs identically under plain `python tests/test_apex_ingest.py`
# and under pytest — same convention as the rest of this test suite.


def test_resolve_ingest_paths_valid_video_and_chat_log() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video bytes")
        chat = tmp_path / "chat.jsonl"
        chat.write_text('{"timestamp": 1.0, "text": "hi"}\n')

        video_path, chat_log_path, subtitle_track_path = apex._resolve_ingest_paths(str(video), str(chat))

        assert video_path == video.resolve()
        assert chat_log_path == chat.resolve()
        assert subtitle_track_path is None
    print("  [PASS] valid video + chat log paths resolve correctly\n")


def test_resolve_ingest_paths_no_chat_log_is_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video bytes")

        video_path, chat_log_path, subtitle_track_path = apex._resolve_ingest_paths(str(video), None)

        assert video_path == video.resolve()
        assert chat_log_path is None
    print("  [PASS] omitted --chat-log resolves to None, not an error\n")


def test_resolve_ingest_paths_missing_video_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with pytest.raises(FileNotFoundError, match="--ingest-vod"):
            apex._resolve_ingest_paths(str(tmp_path / "does_not_exist.mp4"), None)
    print("  [PASS] missing --ingest-vod video path raises FileNotFoundError\n")


def test_resolve_ingest_paths_missing_chat_log_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video bytes")
        with pytest.raises(FileNotFoundError, match="--chat-log"):
            apex._resolve_ingest_paths(str(video), str(tmp_path / "does_not_exist.jsonl"))
    print("  [PASS] missing --chat-log path raises FileNotFoundError\n")


def test_resolve_ingest_paths_valid_subtitle_track() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video bytes")
        subs = tmp_path / "captions.vtt"
        subs.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello\n")

        video_path, chat_log_path, subtitle_track_path = apex._resolve_ingest_paths(
            str(video), None, str(subs)
        )

        assert video_path == video.resolve()
        assert chat_log_path is None
        assert subtitle_track_path == subs.resolve()
    print("  [PASS] valid --subtitle-track path resolves correctly\n")


def test_resolve_ingest_paths_no_subtitle_track_is_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video bytes")

        video_path, chat_log_path, subtitle_track_path = apex._resolve_ingest_paths(str(video), None)

        assert subtitle_track_path is None
    print("  [PASS] omitted --subtitle-track resolves to None, not an error\n")


def test_resolve_ingest_paths_missing_subtitle_track_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video bytes")
        with pytest.raises(FileNotFoundError, match="--subtitle-track"):
            apex._resolve_ingest_paths(str(video), None, str(tmp_path / "does_not_exist.vtt"))
    print("  [PASS] missing --subtitle-track path raises FileNotFoundError\n")


# ── 2. _build_ingest_detector ─────────────────────────────────────────────────


def test_build_ingest_detector_with_chat_log_uses_default_weights() -> None:
    detector = apex._build_ingest_detector(has_chat_log=True)
    assert detector._chat_weight == 0.6
    assert detector._audio_weight == 0.4
    assert detector._semantic_weight == 0.0
    print("  [PASS] --chat-log present -> default 0.6/0.4 chat/audio weights, unaffected by subtitle support\n")


def test_build_ingest_detector_without_chat_log_is_audio_only() -> None:
    detector = apex._build_ingest_detector(has_chat_log=False)
    assert detector._chat_weight == 0.0
    assert detector._audio_weight == 1.0
    print("  [PASS] no --chat-log, no --subtitle-track -> audio-only weights "
          "(moot in practice: StreamIngestHandler rejects this combination at intake)\n")


_FAKE_SMD_CONFIG = {
    "sentinel": {
        "stream_moment_detection": {
            "semantic_keywords": {
                "strong": ["headshot"],
                "generic": ["wow", "incredible"],
            },
            "profiles": {
                "interview": {"chat_weight": 0.0, "audio_weight": 0.3,
                              "semantic_weight": 0.7, "audio_z_floor": 0.0},
            },
        }
    }
}


def test_build_ingest_detector_no_chat_but_subtitle_track_uses_semantic_signal() -> None:
    """
    The actual fix for the flagged regression: a chat-less VOD with a
    subtitle track must not silently fall back to the known-inadequate
    audio-only weights just because the detector-construction code didn't
    know subtitles existed — it must actually USE the semantic signal.
    """
    detector = apex._build_ingest_detector(
        has_chat_log=False, has_subtitle_track=True, config=_FAKE_SMD_CONFIG
    )
    assert detector._chat_weight == 0.0
    assert detector._audio_weight == 0.3
    assert detector._semantic_weight == 0.7
    print("  [PASS] no --chat-log but --subtitle-track present -> audio+semantic fusion (not audio-only)\n")


def test_build_ingest_detector_subtitle_path_wires_keywords_and_audio_z_floor() -> None:
    """
    Guards a real wiring bug found on 2026-07-31: apex.py used to construct
    the no-chat+subtitles detector directly with StreamMomentDetector(...)
    and NO keyword lists at all — so the real CLI's semantic_z was
    identically zero (every cue scored 0.0 against empty keyword sets)
    while the fusion weights said 0.7 semantic. The detector must come from
    build_moment_detector_from_config (interview profile) so settings.yaml's
    keyword vocabulary AND the profile's audio_z_floor actually reach the
    real detection path, not just the diagnostics that hardcoded them.
    """
    detector = apex._build_ingest_detector(
        has_chat_log=False, has_subtitle_track=True, config=_FAKE_SMD_CONFIG
    )
    assert detector._strong_word_set == {"headshot"}, (
        "strong keywords from config must reach the CLI's detector — empty "
        "keyword sets make the semantic signal silently inert"
    )
    assert detector._generic_word_set == {"wow", "incredible"}
    assert detector._audio_z_floor == 0.0, (
        "the interview profile's rectified-audio rule must apply to the "
        "real CLI path, not only to detectors built in tests/diagnostics"
    )
    print("  [PASS] no-chat+subtitles detector carries config keywords and audio_z_floor "
          "(semantic signal actually live in the CLI path)\n")


# ── 3. _drain_ingest_pipeline — the load-bearing stage ORDER ─────────────────


class _RecordingStage:
    """Fake with an async drain() that appends its own name to a shared,
    shared-across-stages call-order list — proves ORDER, not just that
    each drain() was called."""

    def __init__(self, name: str, order: list[str], delay: float = 0.0) -> None:
        self._name = name
        self._order = order
        self._delay = delay

    async def drain(self) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)
        self._order.append(self._name)


def test_drain_ingest_pipeline_runs_bridge_then_brain_then_renderer() -> None:
    async def run() -> None:
        order: list[str] = []
        # Give the earlier stages in the WRONG order a head start if drained
        # concurrently, so a real ordering bug (e.g. asyncio.gather instead
        # of sequential awaits) would show up as a shuffled `order` list.
        bridge = _RecordingStage("bridge", order, delay=0.03)
        brain = _RecordingStage("brain", order, delay=0.02)
        renderer = _RecordingStage("renderer", order, delay=0.01)

        await apex._drain_ingest_pipeline(bridge, brain, renderer)

        assert order == ["bridge", "brain", "renderer"], (
            f"expected strict bridge -> brain -> renderer order, got {order}"
        )

    asyncio.run(run())
    print("  [PASS] _drain_ingest_pipeline drains bridge -> brain -> renderer, in order\n")


def test_drain_ingest_pipeline_propagates_a_stuck_stage_to_the_caller() -> None:
    """
    _drain_ingest_pipeline itself has no timeout — run_ingest_vod wraps the
    whole call in asyncio.wait_for(). Proves a hang in one stage isn't
    silently swallowed here; the caller's timeout is what's responsible for
    bounding it (see run_ingest_vod's docstring).
    """
    async def run() -> None:
        order: list[str] = []
        bridge = _RecordingStage("bridge", order)

        class _StuckBrain:
            async def drain(self) -> None:
                await asyncio.Event().wait()  # never completes

        renderer = _RecordingStage("renderer", order)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                apex._drain_ingest_pipeline(bridge, _StuckBrain(), renderer),
                timeout=0.1,
            )
        # bridge drained before the stuck stage; renderer never got there.
        assert order == ["bridge"], f"expected only bridge to have drained, got {order}"

    asyncio.run(run())
    print("  [PASS] a stuck stage propagates as a timeout at the caller, not a silent hang\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  APEX — apex.py --ingest-vod CLI test")
    print("=" * 60 + "\n")
    test_resolve_ingest_paths_valid_video_and_chat_log()
    test_resolve_ingest_paths_no_chat_log_is_none()
    test_resolve_ingest_paths_missing_video_raises()
    test_resolve_ingest_paths_missing_chat_log_raises()
    test_resolve_ingest_paths_valid_subtitle_track()
    test_resolve_ingest_paths_no_subtitle_track_is_none()
    test_resolve_ingest_paths_missing_subtitle_track_raises()
    test_build_ingest_detector_with_chat_log_uses_default_weights()
    test_build_ingest_detector_without_chat_log_is_audio_only()
    test_build_ingest_detector_no_chat_but_subtitle_track_uses_semantic_signal()
    test_build_ingest_detector_subtitle_path_wires_keywords_and_audio_z_floor()
    test_drain_ingest_pipeline_runs_bridge_then_brain_then_renderer()
    test_drain_ingest_pipeline_propagates_a_stuck_stage_to_the_caller()
    print("  All apex.py --ingest-vod tests passed.")
    print("=" * 60 + "\n")
