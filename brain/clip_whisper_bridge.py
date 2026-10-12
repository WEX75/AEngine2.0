"""
brain/clip_whisper_bridge.py — Pillar 2: MOMENT_DETECTED -> Whisper -> Brain's queue
Apex Engine 2.0

The missing link between Sentinel's MOMENT_DETECTED output and
ClipScriptGenerator's expected input:

    MOMENT_DETECTED + source video path
        --> [THIS MODULE'S OWN QUEUE]
            seek-only ffmpeg audio extraction (just the flagged window)
            --> faster-whisper transcription (CPU-only, large-v3, INT8 —
                per CLAUDE.md's verified hardware config; never CUDA)
            --> indexed TranscriptSegment array, VOD-absolute timestamps
        --> brain.submit_clip_script_job(moment, segments, ...)
            --> [BRAIN'S QUEUE] ClipScriptGenerator.generate()
            --> CLIP_SCRIPT_READY (published by ClipScriptGenerator)

Why this module owns its OWN queue, not Brain's
--------------------------------------------------
Per CLAUDE.md's "JobQueue ownership" section, queues are keyed by resource
SHAPE, not by pillar. Extraction + Whisper transcription is a heavy,
minutes-scale, CPU-saturating job (154-185s per the hardware table) —
nothing like Brain's queue, which exists specifically for short (10-30s),
light, single-Ollama-call jobs. Two ways this could have gone wrong
instead:

  * Brain calls this module's extraction/transcription inline before
    enqueueing just the Ollama step. Doesn't block the event loop (the
    heavy calls are already asyncio.to_thread-wrapped below) — but "not
    blocking the loop" is not the same as "governed." Every concurrent
    call would spawn an ungoverned, unbounded-concurrency Whisper pass
    with no backpressure, silently smuggling a heavy resource shape into
    the one queue that's supposed to only ever hold light jobs.
  * This module holds a callback into Brain instead of a queue of its own.
    Same problem from the other side — nothing bounds how many
    transcriptions run concurrently if several MOMENT_DETECTED events
    arrive in a burst.

So: this module owns its own bounded JobQueue. Its job body is exactly the
existing to_thread-wrapped extraction+transcription; on completion, the
job body calls brain.submit_clip_script_job(...) to hand off to Brain's
queue for just the Ollama-shaped step. Two queues, two shapes, each
correctly governed on its own terms.

Queue sizing — worker count
------------------------------
ONE worker, deliberately, not by omission (core.job_queue.JobQueue only
supports one worker today, but the reasoning holds even if it grew
multi-worker support later): a single large-v3/int8 Whisper pass already
dominates a per-clip pipeline (~75-80% of total time, 154-185s, per
CLAUDE.md's hardware table) and is CPU-only by design. Running N concurrent
passes would mean N Whisper processes contending for the same 16 threads
this hardware has — RAM tolerates it (see below) but the THERMAL budget
does not: CLAUDE.md is explicit that "CPU and GPU share one die/cooling
budget... sustained 100/100 simultaneous load = throttling." A second
concurrent Whisper pass doesn't parallelize useful work here, it just
pushes sustained near-100% CPU utilization and likely makes BOTH jobs
slower via contention/throttling than running them back to back. One
worker is the deliberate ceiling, not an inherited default.

Queue sizing — max_size (RAM check)
--------------------------------------
Each RUNNING job holds one warm Whisper model (large-v3, int8) at ~1.5GB
resident. Since there's only ever 1 job running (one worker), the running
footprint is a flat ~1.5GB, not N x 1.5GB — comfortably inside the
~20-22GB total RAM ceiling even alongside Brain's queue (Ollama's own
llama3:8b Q4_K_M process, resident separately, ~4-5GB) running at the same
time; that's the "different resource shapes overlapping across jobs" case
CLAUDE.md's thermal rules explicitly call desired, not a violation. RAM
was never the binding constraint here — thermal/CPU contention is (see
worker count above).

QUEUED-but-not-running jobs cost ~0 RAM (a MomentDetectedPayload + a path
string sitting in an asyncio.Queue), so max_size isn't a RAM control
either — it's how large a backlog of pending transcriptions is tolerable
before back-pressuring whoever calls submit_moment(). max_size=5 is set
to the same ceiling tests/test_stream_sentinel_scale.py already uses
(MAX_CANDIDATES=5) — that's this codebase's own existing estimate of how
many MOMENT_DETECTED candidates one VOD's detection pass realistically
produces. A normal single-VOD burst shouldn't block its producer; a
backlog larger than that is a real signal something's wrong (multiple
VODs queuing up faster than ~185s/job can drain), and back-pressure is
the correct response, not a bigger buffer.

What "back-pressure" actually means once the queue is full: submit_moment()
BLOCKS the caller (await, never returns) until a slot frees — it does NOT
raise and does NOT silently drop the moment. This is inherited, unmodified
core.job_queue.JobQueue behavior (asyncio.Queue.put() semantics), and it's
the correct one here — confirmed empirically, not just assumed, by
tests/test_clip_whisper_bridge.py::test_submit_moment_blocks_not_drops_or_raises_when_queue_full,
which fills the queue to max_size, proves a 6th-style submission stays
genuinely pending, then proves it completes (not lost) once a slot frees.
A detected moment is never silently dropped this way — worst case it
waits, which at this queue's sizing could mean multiple minutes under
sustained overload (see worker-count reasoning above: ~185s/job, one at a
time). submit_moment() logs a WARNING immediately before a call that will
block, specifically so that wait is visible in logs rather than an
unexplained pause — see submit_moment()'s own docstring. Callers that
cannot tolerate blocking for minutes (e.g. code awaited synchronously by
something else with no timeout of its own) need to know this before
calling submit_moment() directly; there is no separate non-blocking
"reject immediately" path today.

PROVISIONAL: whether this ends up staying its own permanent queue, or
gets absorbed into Cutter's future queue (CLAUDE.md's hardware table
already times Whisper as part of one cohesive Whisper->Ollama->FFmpeg
per-clip pipeline, not a pillar-owned stage) is an open question — see
the "JobQueue ownership" section in CLAUDE.md, marked provisional there
too. Not resolved here; this module works standalone either way.

Why `source_path` is a parameter, not read off the event
------------------------------------------------------------
MOMENT_DETECTED's contract in CLAUDE.md is
`{vod_id, window_start, window_end, composite_score, chat_score,
audio_score, top_spam_tokens, detection_method}` — deliberately no
source_path. That field only exists on STREAM_VOD_INGESTED. Rather than
overload MOMENT_DETECTED with an unrelated field (CLAUDE.md: "never
overload an existing model with unrelated optional fields"), this module
takes source_path as an explicit argument. `orchestrator/clip_orchestrator.py`
is the vod_id -> source_path mapping this used to flag as missing — it
retains STREAM_VOD_INGESTED payloads in a bounded registry and resolves
source_path before calling submit_moment() here. This module still
doesn't subscribe to the EventBus itself, same reasoning as
clip_script_generator.py's — the orchestrator is what subscribes and
calls in, not this module reaching out on its own.

Seek-only extraction (window ± transcribe margin)
---------------------------------------------------
Only the flagged window PLUS a small fixed margin on each side —
[window_start - margin, window_end + margin), clamped at 0 — is ever read
from the source file: `-ss`/`-t` placed BEFORE `-i` for fast input-side
seeking, `-vn` to skip video decode entirely (CLAUDE.md Pillar 1/3: never
read the full VOD up front; a ±6s margin on a seek-only extraction is
seconds of extra audio, not a full-VOD read). Output is mono 16kHz WAV,
what faster-whisper wants natively. See _DEFAULT_TRANSCRIBE_MARGIN_SEC
for why the margin exists (auto-repair pad-to-minimum needs real adjacent
segments to extend into) and the full derivation of its size.

CPU-only, always
-----------------
device="cpu", compute_type="int8" are the hardcoded defaults, matching
CLAUDE.md's verified real config for this hardware (`gfx1103` has no
ROCm path; Whisper transcription is CPU-only by design, unaffected by
Ollama's GPU offload). Override params exist for testing, not for
reaching for CUDA.

Absolute timestamps
---------------------
faster-whisper transcribes the *extracted* clip, so its segment
timestamps start at 0 — code shifts every segment by +window_start before
building TranscriptSegment, so downstream (ClipScriptGenerator's repair
logic, and eventually Cutter) always sees real VOD-relative timestamps,
never clip-relative ones (CLAUDE.md Pillar 2: "Code maps segment_id -> real
Whisper timestamp").
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from brain.brain import Brain
from brain.clip_script_generator import (
    ClipScriptGenerationError,
    TranscriptSegment,
    WordTiming,
)
from brain.ollama_client import OllamaError
from core.event_bus import MomentDetectedPayload
from core.job_queue import JobQueue
from core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_TARGET_DURATION_RANGE: tuple[float, float] = (12.0, 75.0)

# CLAUDE.md-verified real config for this hardware — CPU-only, always.
_DEFAULT_MODEL_SIZE   = "large-v3"
_DEFAULT_DEVICE       = "cpu"
_DEFAULT_COMPUTE_TYPE = "int8"

# See "Queue sizing" in the module docstring for the reasoning behind both.
_DEFAULT_QUEUE_MAX_SIZE = 5

# Transcription margin: extract/transcribe [window_start - margin,
# window_end + margin), not just the detected window. Exists for exactly
# one reason — ClipScriptGenerator's auto-repair pad-to-minimum rule can
# only extend into segments that were actually transcribed, and a real
# full-pipeline run with a short detected window proved a
# bare window can be structurally un-paddable below target_duration_range's
# 12s minimum. The 6.0s value is derived, not arbitrary:
#
#   * Arithmetic floor: the detector's smallest emitted window is
#     pad_before(3) + one 1s bucket + pad_after(5) = 9s; the shortfall to
#     the 12s minimum is 3s -> 1.5s/side IF speech were continuous and
#     aligned to the extraction edges.
#   * Real speech isn't continuous: conversational gaps can clip or cause
#     VAD to drop a sentence at an extraction edge. A wider margin helps
#     preserve complete adjacent segments.
#   * Cost: measured warm large-v3/int8 throughput on this hardware is
#     ~0.72s CPU per audio-second (20s window -> 14.3s). ±6s adds 12s of
#     audio ≈ +9s CPU per clip — ~4-5% of the measured ~187-243s per-clip
#     pipeline. Cheap insurance against an unfixably short clip.
#   * Not larger: the margin serves ONLY the pad rule. 9 + 2*6 = 21s
#     reachable speech ≥ the 12s minimum with ~2x headroom; a bigger
#     margin just inflates the LLM's candidate list and lets selection
#     drift from the detected moment. If target_duration_range's minimum
#     ever grows past ~21s, re-derive this — don't scale it blindly.
#
# Selection and auto-repair still only choose from real transcribed
# segments — the margin gives them real transcript to extend into, it
# never invents content.
_DEFAULT_TRANSCRIBE_MARGIN_SEC = 6.0

Extractor   = Callable[[str, float, float, str], None]
Transcriber = Callable[[str], list[TranscriptSegment]]


class ClipWhisperBridgeError(RuntimeError):
    """Raised when audio extraction or transcription fails for a MOMENT_DETECTED window."""


# ── Default extractor: seek-only ffmpeg audio pull ───────────────────────────────


def _default_extract_window_audio(
    source_path:  str,
    window_start: float,
    window_end:   float,
    output_path:  str,
    ffmpeg_bin:   str = "ffmpeg",
) -> None:
    """
    Seek-extracts [window_start, window_end) from source_path's audio track
    to a mono 16kHz WAV at output_path. Never reads the rest of the file —
    `-ss`/`-t` before `-i` is fast input-side seeking, `-vn` skips video
    decode entirely.
    """
    duration = window_end - window_start
    cmd = [
        ffmpeg_bin, "-y",
        "-ss", f"{window_start}",
        "-t", f"{duration}",
        "-i", source_path,
        "-vn",                 # no video — audio only
        "-ac", "1",            # mono
        "-ar", "16000",        # 16kHz — what faster-whisper wants natively
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ClipWhisperBridgeError(
            f"ffmpeg not found on PATH (tried '{ffmpeg_bin}'). "
            "Install ffmpeg to extract clip audio."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ClipWhisperBridgeError(
            f"ffmpeg failed extracting [{window_start:.1f}s, {window_end:.1f}s) "
            f"from '{source_path}': {exc.stderr[:300]}"
        ) from exc


# ── Default transcriber: faster-whisper, CPU-only ─────────────────────────────────


class _WhisperTranscriber:
    """
    Thin wrapper around faster_whisper.WhisperModel. Lazily imports
    faster_whisper and lazily loads the model on first use (loading is the
    expensive part — seconds — so one instance keeps it warm across calls,
    same spirit as Ollama's keep_alive, applied to Whisper instead). Safe
    to keep warm across jobs here specifically because this queue's single
    worker guarantees only one transcription ever runs at a time — see
    "Queue sizing" in the module docstring.
    """

    def __init__(
        self,
        model_size:    str = _DEFAULT_MODEL_SIZE,
        device:        str = _DEFAULT_DEVICE,
        compute_type:  str = _DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self._model_size   = model_size
        self._device       = device
        self._compute_type = compute_type
        self._model = None  # lazy — never loaded unless actually called

    def __call__(self, audio_path: str) -> list[TranscriptSegment]:
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import — heavy dependency
            log.info(
                f"[ClipWhisperBridge] Loading Whisper model={self._model_size} "
                f"device={self._device} compute_type={self._compute_type}..."
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

        # word_timestamps=True: per-word start/end alongside each segment —
        # what Cutter's word-level "pop" caption timing runs on (CLAUDE.md
        # Pillar 3: "word-level karaoke timing from Whisper"). Real measured
        # cost on this hardware is small relative to the transcription
        # itself — see CLAUDE.md's hardware table note — not free, but
        # nowhere near a second transcription pass.
        raw_segments, _info = self._model.transcribe(audio_path, word_timestamps=True)

        segments: list[TranscriptSegment] = []
        for i, seg in enumerate(raw_segments):
            words = tuple(
                WordTiming(word=w.word.strip(), start=w.start, end=w.end)
                for w in (seg.words or ())
            )
            segments.append(
                TranscriptSegment(id=i, start=seg.start, end=seg.end,
                                  text=seg.text.strip(), words=words)
            )
        return segments


# ── Bridge ─────────────────────────────────────────────────────────────────────────


class ClipWhisperBridge:
    """
    Owns a bounded JobQueue for extraction+transcription (see module
    docstring for sizing). submit_moment() enqueues a job whose body
    extracts and transcribes the flagged window, then hands the resulting
    segments to brain.submit_clip_script_job() — Brain's own queue takes
    it from there.

    extractor/transcriber are injectable so tests never need a real ffmpeg
    binary or a loaded Whisper model — see tests/test_clip_whisper_bridge.py.
    `brain` is a required constructor argument, not defaulted to a fresh
    `Brain()` — a silently-constructed default would own its own queue
    that nobody starts, and would make it too easy to end up with two
    disconnected Brain instances by accident. Tests inject a real (but
    fully-faked-internals) Brain so this never needs a live Ollama call.
    """

    def __init__(
        self,
        brain:          Brain,
        extractor:      Extractor   | None = None,
        transcriber:    Transcriber | None = None,
        model_size:     str = _DEFAULT_MODEL_SIZE,
        device:         str = _DEFAULT_DEVICE,
        compute_type:   str = _DEFAULT_COMPUTE_TYPE,
        queue_max_size: int = _DEFAULT_QUEUE_MAX_SIZE,
        transcribe_margin_sec: float = _DEFAULT_TRANSCRIBE_MARGIN_SEC,
    ) -> None:
        self._brain          = brain
        self._extractor      = extractor or _default_extract_window_audio
        self._transcriber    = transcriber or _WhisperTranscriber(model_size, device, compute_type)
        self._queue_max_size = queue_max_size
        self._transcribe_margin_sec = transcribe_margin_sec
        self._queue          = JobQueue(max_size=queue_max_size)
        self._running        = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start this bridge's own extraction/transcription queue worker."""
        if self._running:
            log.warning("[ClipWhisperBridge] Already running — ignoring duplicate start().")
            return
        await self._queue.start()
        self._running = True
        log.info("[ClipWhisperBridge] Online — transcription queue started.")

    async def stop(self) -> None:
        """
        Stop this bridge's queue worker. Does NOT wait for queued/running
        jobs to finish — call `await bridge.drain()` first if that matters
        (see `drain()`).
        """
        if not self._running:
            return
        await self._queue.stop()
        self._running = False
        log.info("[ClipWhisperBridge] Stopped.")

    async def drain(self) -> None:
        """
        Block until every submit_moment() call made so far has finished
        extraction+transcription AND handed off to
        brain.submit_clip_script_job() (which only enqueues onto Brain's
        queue — it doesn't wait for Brain's own job to finish). Callers
        that need the WHOLE pipeline drained (through CLIP_SCRIPT_READY and
        CLIP_RENDER_READY) must also drain Brain and the renderer, in that
        order, AFTER this — see apex.py's --ingest-vod path.
        """
        await self._queue.join()

    # ── Public API ────────────────────────────────────────────────────────────

    async def submit_moment(
        self,
        moment:                  MomentDetectedPayload,
        source_path:             str | Path,
        target_duration_range:   tuple[float, float] = _DEFAULT_TARGET_DURATION_RANGE,
    ) -> None:
        """
        Enqueue extraction+transcription for one MOMENT_DETECTED window.
        Returns once the job is queued, not once it completes — the
        eventual hand-off to Brain (and CLIP_SCRIPT_READY beyond that)
        happens later, from inside this queue's single worker.

        WHEN THE QUEUE IS FULL (queue depth == max_size, i.e. one job
        already running plus max_size queued): this call BLOCKS — it
        awaits until the running job finishes and frees a slot. It does
        NOT raise, and it does NOT silently drop the moment (inherited,
        unmodified core.job_queue.JobQueue / asyncio.Queue.put()
        semantics — confirmed empirically, not just assumed, by
        tests/test_clip_whisper_bridge.py::test_submit_moment_blocks_not_drops_or_raises_when_queue_full).
        A WARNING is logged immediately before a call that will block, so
        the wait is visible in logs rather than an unexplained pause.
        Given this queue's sizing (one worker, ~185s/job worst case — see
        module docstring), a call made while already full can block for
        multiple minutes under sustained overload. Don't call this from
        somewhere that can't tolerate that.

        Raises
        ------
        ValueError  if the window itself is empty/inverted — checked here,
        synchronously, before the job ever occupies a queue slot; there's
        nothing a queued retry could do about a nonsensical window.
        """
        duration = moment.window_end - moment.window_start
        if duration <= 0:
            raise ValueError(
                f"MOMENT_DETECTED window is empty/inverted: "
                f"[{moment.window_start}, {moment.window_end}]"
            )

        queue_depth = self._queue.qsize()
        log.info(
            f"[ClipWhisperBridge] Queuing extraction+transcription — "
            f"vod_id={moment.vod_id} window=[{moment.window_start:.1f}s, "
            f"{moment.window_end:.1f}s) from {source_path} (queue depth={queue_depth})"
        )
        if queue_depth >= self._queue_max_size:
            log.warning(
                f"[ClipWhisperBridge] Queue at capacity ({queue_depth}/{self._queue_max_size}) — "
                f"submit_moment() for vod_id={moment.vod_id} will BLOCK until a running job "
                f"frees a slot (up to ~185s worst case). The moment is NOT lost, just waiting."
            )
        await self._queue.enqueue(
            self._transcribe_and_submit, moment, source_path, target_duration_range
        )

    # ── Job worker ────────────────────────────────────────────────────────────

    async def _transcribe_and_submit(
        self,
        moment:                  MomentDetectedPayload,
        source_path:             str | Path,
        target_duration_range:   tuple[float, float],
    ) -> None:
        """
        Runs inside this bridge's own JobQueue single worker — never
        Brain's. Extracts + transcribes (the heavy CPU work), then runs
        selection on Brain's queue via brain.select_clip_script() and
        inspects the result for a TRANSCRIPT-BOUNDARY TOUCH before letting
        it publish:

        Boundary widen-retry (found by watching real output, 2026-07-31)
        ------------------------------------------------------------------
        A selection whose end_segment_id equals the LAST transcribed
        segment id usually means the moment's real conclusion lay beyond
        the transcribe margin — the LLM ran out of transcript, it didn't
        choose to stop there. In that case: re-extract/re-transcribe with
        the END margin doubled (start side untouched — the boundary was
        touched at the end), retry selection ONCE against the wider
        transcript, then publish whatever that yields — a second boundary
        touch is accepted with a warning (the content genuinely continues,
        or the VOD ended). Capped at one retry to bound re-transcription
        cost (~0.72s CPU per audio-second, measured). The re-transcription
        runs HERE, inside this bridge's heavy-CPU worker — never inside
        Brain's light LLM queue; only the short Ollama-shaped selection
        runs on Brain's queue, via the result-returning
        select_clip_script(publish=False) + publish_clip_script() pair.

        Error isolation mirrors Brain's _generate_clip_and_emit: known
        failure modes are caught and logged; JobQueue's catch-all is the
        backstop — one bad job must never take down the worker.
        """
        try:
            segments = await self._transcribe(moment, source_path)
        except ClipWhisperBridgeError as exc:
            log.error(f"[ClipWhisperBridge] vod_id={moment.vod_id}: {exc}")
            return

        try:
            payload = await self._brain.select_clip_script(
                moment, segments, target_duration_range, publish=False
            )

            if payload.end_segment_id == max(s.id for s in segments):
                log.warning(
                    f"[ClipWhisperBridge] vod_id={moment.vod_id}: selection ends at the "
                    f"last transcribed segment (id {payload.end_segment_id}) — selection "
                    f"touches transcript boundary, content may be truncated. Widening end "
                    f"margin to {2 * self._transcribe_margin_sec:.1f}s and retrying once."
                )
                try:
                    segments = await self._transcribe(
                        moment, source_path,
                        end_margin_sec=2 * self._transcribe_margin_sec,
                    )
                    payload = await self._brain.select_clip_script(
                        moment, segments, target_duration_range, publish=False
                    )
                    if payload.end_segment_id == max(s.id for s in segments):
                        log.warning(
                            f"[ClipWhisperBridge] vod_id={moment.vod_id}: selection still "
                            f"touches the transcript boundary after widening — accepting "
                            f"(content genuinely continues past the margin, or the VOD ended)."
                        )
                except ClipWhisperBridgeError as exc:
                    log.error(
                        f"[ClipWhisperBridge] vod_id={moment.vod_id}: widened "
                        f"re-transcription failed ({exc}) — publishing the original "
                        f"boundary-touching selection rather than losing the clip."
                    )

            await self._brain.publish_clip_script(payload)

        except (ClipScriptGenerationError, OllamaError) as exc:
            log.error(
                f"[ClipWhisperBridge] vod_id={moment.vod_id}: clip selection failed "
                f"({exc}) — no CLIP_SCRIPT_READY published."
            )

    async def _transcribe(
        self,
        moment:          MomentDetectedPayload,
        source_path:     str | Path,
        end_margin_sec:  float | None = None,
    ) -> list[TranscriptSegment]:
        """
        Extraction + transcription only — no Ollama call, no Brain
        involvement. Returns VOD-absolute-timestamped TranscriptSegments.
        `end_margin_sec` overrides the END-side margin only (the boundary
        widen-retry doubles it); None means the configured default.
        """
        # Widen extraction by the transcribe margin (see
        # _DEFAULT_TRANSCRIBE_MARGIN_SEC for the derivation) — clamped at
        # the VOD's start; overshooting the VOD's end is harmless (ffmpeg
        # just returns less audio).
        end_margin = self._transcribe_margin_sec if end_margin_sec is None else end_margin_sec
        extract_start = max(0.0, moment.window_start - self._transcribe_margin_sec)
        extract_end   = moment.window_end + end_margin

        log.info(
            f"[ClipWhisperBridge] vod_id={moment.vod_id}: extracting "
            f"[{extract_start:.1f}s, {extract_end:.1f}s) "
            f"(window [{moment.window_start:.1f}s, {moment.window_end:.1f}s) "
            f"-{self._transcribe_margin_sec:.1f}s/+{end_margin:.1f}s margin) from {source_path}"
        )

        fd, tmp_audio_path = tempfile.mkstemp(suffix=".wav", prefix="apex_clip_")
        os.close(fd)
        try:
            try:
                await asyncio.to_thread(
                    self._extractor,
                    str(source_path), extract_start, extract_end, tmp_audio_path,
                )
            except ClipWhisperBridgeError:
                raise
            except Exception as exc:
                raise ClipWhisperBridgeError(
                    f"audio extraction failed for vod_id={moment.vod_id}: {exc}"
                ) from exc

            try:
                raw_segments = await asyncio.to_thread(self._transcriber, tmp_audio_path)
            except Exception as exc:
                raise ClipWhisperBridgeError(
                    f"Whisper transcription failed for vod_id={moment.vod_id}: {exc}"
                ) from exc

            if not raw_segments:
                raise ClipWhisperBridgeError(
                    f"Whisper produced zero segments for vod_id={moment.vod_id} "
                    f"window [{moment.window_start:.1f}s, {moment.window_end:.1f}s) — "
                    "no speech detected, nothing to select from"
                )

            # Shift clip-relative Whisper timestamps to VOD-absolute — the
            # segment AND every word inside it; a words list left
            # clip-relative would burn captions at the wrong offset later.
            # The shift is the (clamped) EXTRACTION start, not window_start:
            # with the margin, the extracted audio's t=0 sits margin seconds
            # before the detected window.
            segments = [
                TranscriptSegment(
                    id=s.id,
                    start=s.start + extract_start,
                    end=s.end + extract_start,
                    text=s.text,
                    words=tuple(
                        WordTiming(
                            word=w.word,
                            start=w.start + extract_start,
                            end=w.end + extract_start,
                        )
                        for w in s.words
                    ),
                )
                for s in raw_segments
            ]

            log.info(
                f"[ClipWhisperBridge] vod_id={moment.vod_id}: "
                f"{len(segments)} transcript segment(s) recovered"
            )
            return segments

        finally:
            try:
                os.remove(tmp_audio_path)
            except OSError:
                pass
