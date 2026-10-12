"""
sentinel/audio_extractor.py — Pillar 1: real video -> STREAM_VOD_INGESTED's audio_track_path
Apex Engine 2.0

Produces the per-bucket dBFS array that sentinel/stream_ingest.py's
documented audio_track_path format expects (`{"bucket_sec":..., "dbfs":
[...]}`), from an actual video file. This was flagged as a gap three
times over (stream_ingest.py's own docstring, and the ad-hoc copy that
lived in tools/manual_smoke_test_e2e.py before this module existed):
nothing in this repo derived that array from raw media — the smoke test
grew its own copy just to exercise the real pipeline once. This module is
that logic, moved to a real, importable, pillar-owned location; the
smoke tool now imports it instead of carrying its own copy.

Why sentinel/, not core/ or a new package
----------------------------------------------
This function's ENTIRE reason to exist is producing one of Sentinel's own
two required inputs (chat log + audio track — CLAUDE.md Pillar 1: "Ingest
chat log + audio track *only*"). It's not generic infra (core/ is
pillar-agnostic, same reasoning that kept orchestrator/clip_orchestrator.py
out of core/) and it's not cross-pillar wiring (it doesn't touch Brain or
Cutter at all, unlike orchestrator/). It belongs exactly where
stream_moment_detector.py (the consumer of this exact bucketed array) and
stream_ingest.py (the module whose documented on-disk format this
function's output matches) already live.

Real ffmpeg + a plain RMS/dBFS pass, not a dependency
------------------------------------------------------
Pulls the full audio track as mono 16kHz PCM WAV via one ffmpeg
subprocess call, then computes per-bucket RMS dBFS with numpy (already a
pinned dependency). No new third-party audio library — this is a small,
well-understood computation, not worth a dependency for.

CPU-only, always. Never touches the full VOD beyond ffmpeg's own
`-vn -ac 1 -ar 16000` audio-only extraction (no video decode).
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

_FLOOR_DB = -90.0   # Silence floor — matches the "quiet but not literally -inf" convention used throughout this codebase


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg fails to extract audio, or the result is unusable (empty, wrong format)."""


def extract_audio_dbfs(
    video_path: str | Path,
    bucket_sec: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[list[float], float]:
    """
    Real ffmpeg subprocess call: pull the full audio track as mono 16kHz
    PCM WAV, then compute per-bucket RMS dBFS in Python.

    Parameters
    ----------
    video_path : path to a real video (or audio) file, any format ffmpeg
                 can demux.
    bucket_sec : bucket resolution — passed straight through to
                 StreamMomentDetector's audio_bucket_sec parameter
                 downstream, so it doesn't have to match chat's own
                 bucket_sec (the granularity fix in
                 stream_moment_detector.py aligns them via
                 nearest-timestamp matching).
    ffmpeg_bin : override for testing / non-standard installs.

    Returns
    -------
    (dbfs_values, duration_sec)

    Raises
    ------
    AudioExtractionError  on ffmpeg failure, missing/wrong-format output,
    or an empty audio track.
    """
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="apex_audio_extract_")
    os.close(fd)
    try:
        cmd = [ffmpeg_bin, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", tmp_wav]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise AudioExtractionError(
                f"ffmpeg not found on PATH (tried '{ffmpeg_bin}'). Install ffmpeg to derive audio tracks."
            ) from exc
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg failed extracting the audio track from '{video_path}' "
                f"(exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        with wave.open(tmp_wav, "rb") as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        if sample_width != 2:
            raise AudioExtractionError(f"expected 16-bit PCM from ffmpeg, got sample_width={sample_width}")
        if n_frames == 0:
            raise AudioExtractionError(f"ffmpeg produced an empty audio track for '{video_path}' — is it a valid media file?")

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        duration_sec = n_frames / sample_rate
        samples_per_bucket = max(1, int(round(bucket_sec * sample_rate)))
        n_buckets = max(1, math.ceil(len(samples) / samples_per_bucket))

        dbfs: list[float] = []
        for i in range(n_buckets):
            chunk = samples[i * samples_per_bucket: (i + 1) * samples_per_bucket]
            if len(chunk) == 0:
                dbfs.append(_FLOOR_DB)
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            dbfs.append(_FLOOR_DB if rms <= 0 else max(_FLOOR_DB, 20.0 * math.log10(rms / 32768.0)))

        return dbfs, duration_sec
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
