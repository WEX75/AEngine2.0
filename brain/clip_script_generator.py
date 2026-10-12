"""
brain/clip_script_generator.py — The Brain: Clip Segment Selector
Apex Engine 2.0  |  Pillar 2 (stream-clipping mode)

Consumes a MOMENT_DETECTED window plus the corresponding Whisper indexed-
segment array for that window, prompts the local LLM to *select* a
start_segment_id/end_segment_id range (never raw timestamps — CLAUDE.md
Pillar 2: local 8B-class models are unreliable at numeric precision), and
publishes CLIP_SCRIPT_READY.

Pipeline
--------
    MOMENT_DETECTED + [TranscriptSegment] --> render prompt --> OllamaClient
        --> parse JSON --> validate against ClipScript (Pydantic)
        --> auto-repair (clamp/pad/trim) out-of-range selections
        --> build CLIP_SCRIPT_READY payload --> publish

Two distinct failure modes, two distinct responses
----------------------------------------------------
* Malformed JSON, or a segment_id that doesn't even type-check (Pydantic
  ValidationError) — the model's output is structurally broken, not just
  semantically off. Re-prompt ONCE with the error appended to the prompt
  (reflection), then fail the job cleanly if the retry also fails. A
  single bad LLM response must never crash the pipeline.
* A well-typed segment_id that is simply out of range (not in the
  transcript's id set), or end_segment_id < start_segment_id, or a window
  whose duration falls outside target_duration_range — this is NOT a
  malformed-output problem, it's a bounds problem, and re-prompting for it
  wastes an LLM call. Auto-repair (clamp / swap / pad / trim) instead,
  the same philosophy as ScriptGenerator's body pad/trim in
  brain/script_generator.py.

Why this module doesn't subscribe to the EventBus itself
------------------------------------------------------------
MOMENT_DETECTED alone doesn't carry the Whisper segment array for its
window, so generate() takes segments as a direct parameter rather than
subscribing itself. That's still correct now that
orchestrator/clip_orchestrator.py exists: the orchestrator drives
clip_whisper_bridge.submit_moment() (which produces segments and hands
off to Brain), not this module directly — ClipScriptGenerator is reached
only via Brain's queue (brain.brain.Brain._generate_clip_and_emit), never
subscribes to the EventBus on its own, and shouldn't start now either.

Clip length
-----------
target_duration_range is a parameter, never a hardcoded constant (CLAUDE.md
"Clip length"): the LLM is told the range and picks a natural boundary
inside it; auto-repair only pads/trims when the LLM's pick falls outside it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jinja2 import Environment, FileSystemLoader, TemplateNotFound  # type: ignore[import]
from pydantic import BaseModel, Field, ValidationError

from brain.ollama_client import OllamaClient, OllamaError
from core.event_bus import (
    CLIP_SCRIPT_READY,
    ClipScriptReadyPayload,
    EventBus,
    MomentDetectedPayload,
    get_bus,
)
from core.logger import get_logger

log = get_logger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _PROJECT_ROOT / "config" / "prompts"
_TEMPLATE_FILE = "clip_selection.j2"

_DEFAULT_TARGET_DURATION_RANGE: tuple[float, float] = (12.0, 75.0)  # CLAUDE.md "Clip length" example, a default not a rule
_MAX_ATTEMPTS = 2   # one original call + one reflection retry


class ClipScriptGenerationError(RuntimeError):
    """Raised when a clip script cannot be generated or validated, even after retrying once."""


# ── Data contracts ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WordTiming:
    """
    One word's real Whisper timestamp (word_timestamps=True): the word text
    plus its [start, end) span. Same timestamp frame as the owning
    TranscriptSegment (clip-relative out of Whisper, shifted to
    VOD-absolute by clip_whisper_bridge alongside the segment itself).
    Carries what Cutter's word-level "pop" caption timing needs — the LLM
    never sees these; selection stays segment-id-based per CLAUDE.md
    Pillar 2.
    """
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptSegment:
    """
    One indexed Whisper transcript segment: {id, start, end, text} per
    CLAUDE.md Pillar 2. The LLM selects id ranges from a list of these;
    code (never the LLM) maps ids back to the real start/end timestamps.

    `words` (default empty — every pre-word-timestamps construction site
    stays valid) carries the per-word timings when the transcriber ran
    with word_timestamps=True; Cutter uses them for pop-style caption
    beats and falls back to segment-level blocks when absent.
    """
    id: int
    start: float
    end: float
    text: str
    words: tuple[WordTiming, ...] = ()


class ClipScript(BaseModel):
    """
    Pydantic model validating the LLM's raw JSON response. Deliberately
    narrow — a *selection*, not a generation: two segment ids, a confidence
    score, and a one-line reason. No timestamps, no clip content.
    """
    start_segment_id: int
    end_segment_id: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    hook_reason: str = Field(min_length=1)


def caption_segments_from_transcript(segments: Sequence[TranscriptSegment]) -> list[dict[str, Any]]:
    """
    Converts TranscriptSegments into the plain-dict shape
    CLIP_SCRIPT_READY.caption_segments carries — the exact contract
    ClipRenderer's `_build_ass_file` consumes. Shared by
    ClipScriptGenerator.generate() (filters to the LLM's selected id range
    first) and apex.py's --render-manual-window path (uses every segment
    in the manually-specified window — there's no LLM selection to filter
    by). One real implementation of this shape, not two copies.
    """
    return [
        {
            "id": s.id, "start": s.start, "end": s.end, "text": s.text,
            # Always present, explicitly empty when the transcriber ran
            # without word timestamps — one stable contract shape, not
            # an optional key that sometimes vanishes.
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in s.words
            ],
        }
        for s in segments
    ]


# ── Generator ──────────────────────────────────────────────────────────────────


class ClipScriptGenerator:
    """
    Converts a MOMENT_DETECTED window + its Whisper segments into a
    CLIP_SCRIPT_READY event by:
      1. Rendering the Jinja2 selection prompt
      2. Calling OllamaClient.generate() (format:json)
      3. Parsing + Pydantic-validating the JSON response
         (malformed JSON / bad types -> one reflection retry)
      4. Auto-repairing out-of-range ids / durations (clamp/swap/pad/trim)
      5. Publishing CLIP_SCRIPT_READY

    One instance is safe to reuse across moments (no mutable state after
    __init__ — same pattern as ScriptGenerator).
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        bus:    EventBus     | None = None,
    ) -> None:
        self._client = client or OllamaClient()
        self._bus    = bus or get_bus()
        self._env    = self._build_jinja_env()

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        moment:                 MomentDetectedPayload,
        segments:               Sequence[TranscriptSegment],
        target_duration_range:  tuple[float, float] = _DEFAULT_TARGET_DURATION_RANGE,
        publish:                bool = True,
    ) -> ClipScriptReadyPayload:
        """
        Full selection pipeline for one MOMENT_DETECTED window.

        Parameters
        ----------
        moment   : MomentDetectedPayload from Sentinel (Pillar 1)
        segments : Whisper indexed segments covering (at least) the
                   moment's window — the LLM selects start/end ids from
                   this list's own `id` values.
        target_duration_range : (min_sec, max_sec) job parameter — never
                   hardcoded (CLAUDE.md "Clip length"). Auto-repair pads or
                   trims the LLM's selection to fit this window only when
                   it falls outside it.
        publish  : True (default) publishes CLIP_SCRIPT_READY as before.
                   False returns the payload WITHOUT publishing — used by
                   clip_whisper_bridge's transcript-boundary widen-retry,
                   which must inspect a selection before deciding whether
                   it's safe to let downstream render it.

        Returns
        -------
        ClipScriptReadyPayload — already published to the bus when
        `publish` is True; unpublished otherwise.

        Raises
        ------
        ClipScriptGenerationError  if Ollama fails, or the response is
        still structurally invalid after one reflection retry.
        ValueError  if `segments` is empty (nothing to select from).
        """
        if not segments:
            raise ValueError("ClipScriptGenerator.generate() requires a non-empty segments array")

        ordered = sorted(segments, key=lambda s: s.id)
        segments_by_id = {s.id: s for s in ordered}
        min_id, max_id = ordered[0].id, ordered[-1].id

        log.info(
            f"[ClipScriptGenerator] Selecting clip for vod_id={moment.vod_id} "
            f"window=[{moment.window_start:.1f}s, {moment.window_end:.1f}s] "
            f"({len(ordered)} candidate segments, ids {min_id}-{max_id})"
        )

        clip, attempts = await self._select_with_retry(moment, ordered, target_duration_range)

        start_id, end_id, repair_notes = self._repair_selection(
            clip, segments_by_id, min_id, max_id, target_duration_range
        )
        if repair_notes:
            log.warning(
                f"[ClipScriptGenerator] Auto-repaired selection for "
                f"vod_id={moment.vod_id}: {'; '.join(repair_notes)}"
            )

        caption_segments = caption_segments_from_transcript(
            [s for s in ordered if start_id <= s.id <= end_id]
        )

        payload = ClipScriptReadyPayload(
            vod_id           = moment.vod_id,
            start_segment_id = start_id,
            end_segment_id   = end_id,
            caption_segments = caption_segments,
            hook_reason      = clip.hook_reason,
        )

        log.info(
            f"[ClipScriptGenerator] >> "
            f"{'Emitting CLIP_SCRIPT_READY' if publish else 'Selection complete (publish deferred to caller)'} | "
            f"vod_id={moment.vod_id} | segments [{start_id}, {end_id}] | "
            f"confidence={clip.confidence_score:.2f} | attempts={attempts} | "
            f"reason='{clip.hook_reason[:60]}'"
        )
        if publish:
            await self._bus.publish(CLIP_SCRIPT_READY, payload)
        return payload

    # ── Private: prompt + retry loop ─────────────────────────────────────────────

    async def _select_with_retry(
        self,
        moment:                 MomentDetectedPayload,
        segments:               Sequence[TranscriptSegment],
        target_duration_range:  tuple[float, float],
    ) -> tuple[ClipScript, int]:
        """
        Calls Ollama, parses + Pydantic-validates the response. On a
        structural failure (bad JSON, or a field that doesn't even
        type-check — e.g. a non-numeric segment_id, or a missing required
        key) re-prompts ONCE with the error appended, then raises if the
        retry is also structurally broken. Returns (validated ClipScript,
        attempt_count).
        """
        retry_error: str | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            prompt = self._render_prompt(moment, segments, target_duration_range, retry_error)

            try:
                result = await self._client.generate(prompt=prompt, expect_json=True)
            except OllamaError as exc:
                raise ClipScriptGenerationError(
                    f"Ollama call failed for vod_id={moment.vod_id}: {exc}"
                ) from exc

            raw_text = result["text"]
            log.debug(
                f"[ClipScriptGenerator] Attempt {attempt} raw Ollama output "
                f"({len(raw_text)} chars): {raw_text[:200]}"
            )

            try:
                parsed = self._parse_json(raw_text)
                clip = ClipScript(**parsed)
                return clip, attempt

            except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as exc:
                retry_error = self._describe_error(exc, raw_text)
                log.warning(
                    f"[ClipScriptGenerator] Attempt {attempt} produced invalid output for "
                    f"vod_id={moment.vod_id}: {retry_error}"
                )
                if attempt >= _MAX_ATTEMPTS:
                    raise ClipScriptGenerationError(
                        f"LLM output still invalid after {_MAX_ATTEMPTS} attempt(s) "
                        f"for vod_id={moment.vod_id}: {retry_error}"
                    ) from exc
                # else: loop again — retry_error is now set, so the next
                # rendered prompt includes it (reflection).

        # Unreachable: the loop above always returns or raises.
        raise ClipScriptGenerationError("unreachable: retry loop exited without a result")

    @staticmethod
    def _describe_error(exc: Exception, raw_text: str) -> str:
        if isinstance(exc, ValidationError):
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return f"schema validation failed ({problems}). Raw output: {raw_text[:200]}"
        return f"{type(exc).__name__}: {exc}. Raw output: {raw_text[:200]}"

    # ── Private: JSON parsing (same fence-stripping strategy as ScriptGenerator) ──

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        brace_idx = text.find("{")
        if brace_idx > 0:
            text = text[brace_idx:]
        last_brace = text.rfind("}")
        if last_brace != -1 and last_brace < len(text) - 1:
            text = text[: last_brace + 1]

        parsed = json.loads(text)  # raises json.JSONDecodeError on malformed JSON
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM returned JSON but it is not an object (got {type(parsed).__name__})")
        return parsed

    # ── Private: auto-repair (pad/trim/clamp) ─────────────────────────────────────

    @staticmethod
    def _repair_selection(
        clip:                    ClipScript,
        segments_by_id:          dict[int, TranscriptSegment],
        min_id:                  int,
        max_id:                  int,
        target_duration_range:   tuple[float, float],
    ) -> tuple[int, int, list[str]]:
        """
        Corrects a well-typed but out-of-bounds selection. Never re-prompts
        the LLM — this is pure code, matching CLAUDE.md's "code maps
        segment_id -> real Whisper timestamp" design.

        Repair rules
        ------------
        * ids outside [min_id, max_id]        -> clamp to the nearest valid id
        * end_segment_id < start_segment_id   -> swap
        * duration < target_duration_range[0] -> extend outward (end first,
          then start) until in range or the transcript is exhausted
        * duration > target_duration_range[1] -> trim end backward toward
          start until in range (never crosses start)
        """
        notes: list[str] = []

        start_id = clip.start_segment_id
        end_id   = clip.end_segment_id

        clamped_start = min(max(start_id, min_id), max_id)
        clamped_end   = min(max(end_id, min_id), max_id)
        if (clamped_start, clamped_end) != (start_id, end_id):
            notes.append(
                f"clamped out-of-range ids ({start_id}, {end_id}) into "
                f"valid range [{min_id}, {max_id}] -> ({clamped_start}, {clamped_end})"
            )
        start_id, end_id = clamped_start, clamped_end

        if end_id < start_id:
            notes.append(f"end_segment_id {end_id} < start_segment_id {start_id} — swapped")
            start_id, end_id = end_id, start_id

        min_dur, max_dur = target_duration_range

        def duration() -> float:
            return segments_by_id[end_id].end - segments_by_id[start_id].start

        dur = duration()
        if dur < min_dur:
            while dur < min_dur and (end_id < max_id or start_id > min_id):
                if end_id < max_id:
                    end_id += 1
                elif start_id > min_id:
                    start_id -= 1
                dur = duration()
            notes.append(f"padded short window to {dur:.1f}s (target min {min_dur:.1f}s)")

        elif dur > max_dur:
            while dur > max_dur and end_id > start_id:
                end_id -= 1
                dur = duration()
            notes.append(f"trimmed overlong window to {dur:.1f}s (target max {max_dur:.1f}s)")

        return start_id, end_id, notes

    # ── Private: prompt rendering ─────────────────────────────────────────────────

    def _render_prompt(
        self,
        moment:                 MomentDetectedPayload,
        segments:               Sequence[TranscriptSegment],
        target_duration_range:  tuple[float, float],
        retry_error:            str | None,
    ) -> str:
        try:
            template = self._env.get_template(_TEMPLATE_FILE)
        except TemplateNotFound:
            raise ClipScriptGenerationError(
                f"Prompt template not found: {_TEMPLATES_DIR / _TEMPLATE_FILE}\n"
                "Make sure config/prompts/clip_selection.j2 exists."
            )

        return template.render(
            vod_id            = moment.vod_id,
            window_start      = moment.window_start,
            window_end        = moment.window_end,
            composite_score   = moment.composite_score,
            detection_method  = moment.detection_method,
            top_spam_tokens   = moment.top_spam_tokens,
            target_min        = target_duration_range[0],
            target_max        = target_duration_range[1],
            segments          = segments,
            retry_error       = retry_error,
        )

    # ── Jinja2 env ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_jinja_env() -> Environment:
        if not _TEMPLATES_DIR.exists():
            raise ClipScriptGenerationError(
                f"Prompts directory missing: {_TEMPLATES_DIR}. "
                "Expected: config/prompts/clip_selection.j2"
            )
        return Environment(
            loader                 = FileSystemLoader(str(_TEMPLATES_DIR)),
            trim_blocks            = True,
            lstrip_blocks          = True,
            keep_trailing_newline  = False,
        )
