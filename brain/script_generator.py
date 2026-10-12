"""
brain/script_generator.py  —  The Brain: Script Generator
Apex Engine 2.0  |  Pillar 2

Ingests a TrendFoundPayload, renders the Jinja2 prompt template,
calls OllamaClient, validates the structured JSON response,
and returns a ViralScript dataclass.

Validation strategy
-------------------
llama3:8b with format:"json" produces valid JSON reliably, but may
occasionally produce the wrong keys, wrong body length, or empty strings.
The validator applies corrections before raising so partial outputs are
recovered where possible rather than thrown away.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound  # type: ignore[import]

from brain.models import ViralScript
from brain.ollama_client import OllamaClient, OllamaError
from core.config_loader import get_config
from core.event_bus import TrendFoundPayload
from core.logger import get_logger

log = get_logger(__name__)

# Template location (relative to project root)
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR  = _PROJECT_ROOT / "config" / "prompts"
_TEMPLATE_FILE  = "viral_script.j2"

# Validation limits (must stay in sync with template rules)
_MAX_HOOK_WORDS  = 15
_MAX_BODY_WORDS  = 20
_MAX_CTA_WORDS   = 12
_MIN_BODY_ITEMS  = 3    # Absolute floor if model returns fewer than requested


class ScriptGenerationError(RuntimeError):
    """Raised when a script cannot be generated or validated."""


class ScriptGenerator:
    """
    Converts a TrendFoundPayload into a ViralScript by:
      1. Rendering the Jinja2 prompt template
      2. Calling OllamaClient.generate()
      3. Parsing + validating the JSON response
      4. Assembling and returning a ViralScript

    One instance per Brain — thread-safe (no mutable state after __init__).
    """

    def __init__(self, client: OllamaClient | None = None) -> None:
        self._client  = client or OllamaClient()
        self._env     = self._build_jinja_env()
        cfg           = get_config()
        self._max_tok = cfg.get("brain", {}).get("max_tokens", 500)

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(self, payload: TrendFoundPayload) -> ViralScript:
        """
        Full generation pipeline for one trend.

        Parameters
        ----------
        payload : TrendFoundPayload from The Sentinel

        Returns
        -------
        ViralScript — validated, ready for TTS

        Raises
        ------
        ScriptGenerationError  if Ollama fails or output cannot be recovered.
        """
        log.info(
            f"[ScriptGenerator] Generating script for trend_id={payload.trend_id} "
            f"| topic='{payload.topic[:50]}'"
        )

        # 1. Render prompt
        prompt = self._render_prompt(payload)
        log.debug(f"[ScriptGenerator] Prompt length: {len(prompt)} chars")

        # 2. Call Ollama
        try:
            result = await self._client.generate(
                prompt      = prompt,
                max_tokens  = self._max_tok,
                expect_json = True,
            )
        except OllamaError as exc:
            raise ScriptGenerationError(
                f"Ollama call failed for trend '{payload.topic}': {exc}"
            ) from exc

        raw_text = result["text"]
        log.debug(f"[ScriptGenerator] Raw Ollama output ({len(raw_text)} chars): {raw_text[:200]}")

        # 3. Parse JSON
        parsed = self._parse_json(raw_text, payload.topic)

        # 4. Validate and repair
        validated = self._validate(parsed, payload.hook_count, payload.topic)

        # 5. Assemble ViralScript
        script = self._assemble(payload, validated, raw_text)

        log.info(
            f"[ScriptGenerator] Script ready — {script.summary()} "
            f"(~{script.word_count()} words, "
            f"~{script.estimated_duration_seconds():.0f}s TTS)"
        )
        return script

    # ── Private: prompt rendering ─────────────────────────────────────────────

    def _render_prompt(self, payload: TrendFoundPayload) -> str:
        """Render the Jinja2 template with payload data."""
        try:
            template = self._env.get_template(_TEMPLATE_FILE)
        except TemplateNotFound:
            raise ScriptGenerationError(
                f"Prompt template not found: {_TEMPLATES_DIR / _TEMPLATE_FILE}\n"
                "Make sure config/prompts/viral_script.j2 exists."
            )

        # Clamp hook_count to sane range for the LLM
        hook_count = max(3, min(payload.hook_count, 7))

        return template.render(
            topic      = payload.topic,
            hook_type  = payload.hook_type or "generic",
            hook_count = hook_count,
            keywords   = payload.keywords[:6],  # keep prompt tight
            source_url = payload.source_url,
        )

    # ── Private: JSON parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str, topic: str) -> dict[str, Any]:
        """
        Attempts to parse the raw LLM output as JSON.

        llama3:8b with format:json is usually clean, but occasionally wraps
        the object in markdown fences (```json ... ```) or adds a leading
        sentence. We strip common wrapping patterns before parsing.
        """
        text = raw.strip()

        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
        text = text.strip()

        # If there's prose before the first '{', discard it
        brace_idx = text.find("{")
        if brace_idx > 0:
            log.debug(
                f"[ScriptGenerator] Stripped {brace_idx} leading chars before JSON object."
            )
            text = text[brace_idx:]

        # Trim to last '}'
        last_brace = text.rfind("}")
        if last_brace != -1 and last_brace < len(text) - 1:
            text = text[: last_brace + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScriptGenerationError(
                f"Could not parse LLM output as JSON for topic '{topic}'. "
                f"Error: {exc}. Raw output: {raw[:300]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ScriptGenerationError(
                f"LLM returned JSON but it is not an object. Got: {type(parsed).__name__}. "
                f"Raw: {raw[:200]}"
            )
        return parsed

    # ── Private: validation + repair ─────────────────────────────────────────

    @staticmethod
    def _validate(
        data:       dict[str, Any],
        hook_count: int,
        topic:      str,
    ) -> dict[str, Any]:
        """
        Validates and auto-repairs the parsed JSON.

        Repair rules
        ------------
        * Missing "hook"  → generate a generic hook from the topic
        * Missing "cta"   → use a default CTA
        * Missing "body"  → raise (unrecoverable)
        * body too short  → pad with generic points
        * body too long   → trim to hook_count
        * body items that are lists/dicts → convert to strings
        * Truncated words → trim to word limits (no hard error, just a log)
        """
        # ── hook ──────────────────────────────────────────────────────────────
        hook = str(data.get("hook", "")).strip()
        if not hook:
            log.warning("[ScriptGenerator] Missing 'hook' — using fallback.")
            hook = f"What they never told you about {topic}."
        else:
            words = hook.split()
            if len(words) > _MAX_HOOK_WORDS + 5:  # 5-word grace period
                hook = " ".join(words[: _MAX_HOOK_WORDS]) + "."
                log.debug(f"[ScriptGenerator] Hook trimmed to {_MAX_HOOK_WORDS} words.")

        # ── body ──────────────────────────────────────────────────────────────
        raw_body = data.get("body")
        if not raw_body or not isinstance(raw_body, list):
            raise ScriptGenerationError(
                f"LLM did not return a 'body' list for topic '{topic}'. "
                f"Keys returned: {list(data.keys())}"
            )

        # Flatten: if items are dicts/lists, convert to string
        body: list[str] = []
        for item in raw_body:
            if isinstance(item, str):
                body.append(item.strip())
            elif isinstance(item, dict):
                # Some models return {"point": "..."} or {"text": "..."}
                text = (
                    item.get("point") or item.get("text") or
                    item.get("fact")  or item.get("content") or
                    " ".join(str(v) for v in item.values())
                )
                body.append(str(text).strip())
            else:
                body.append(str(item).strip())

        body = [b for b in body if b]   # drop empties

        # Trim overlong items
        trimmed_body: list[str] = []
        for item in body:
            words = item.split()
            if len(words) > _MAX_BODY_WORDS + 5:
                item = " ".join(words[: _MAX_BODY_WORDS]) + "."
            trimmed_body.append(item)

        # Enforce length
        target = max(_MIN_BODY_ITEMS, hook_count)
        if len(trimmed_body) < target:
            shortfall = target - len(trimmed_body)
            log.warning(
                f"[ScriptGenerator] Body has {len(trimmed_body)} items, "
                f"expected {target}. Padding with {shortfall} generic point(s)."
            )
            for i in range(shortfall):
                trimmed_body.append(f"This is one of the most underreported facts about {topic}.")
        elif len(trimmed_body) > target:
            log.debug(
                f"[ScriptGenerator] Body trimmed from {len(trimmed_body)} → {target} items."
            )
            trimmed_body = trimmed_body[:target]

        # ── cta ───────────────────────────────────────────────────────────────
        cta = str(data.get("cta", "")).strip()
        if not cta:
            log.warning("[ScriptGenerator] Missing 'cta' — using fallback.")
            cta = "Follow before they remove this video."
        else:
            words = cta.split()
            if len(words) > _MAX_CTA_WORDS + 3:
                cta = " ".join(words[: _MAX_CTA_WORDS]) + "."

        return {"hook": hook, "body": trimmed_body, "cta": cta}

    # ── Private: assembly ─────────────────────────────────────────────────────

    @staticmethod
    def _assemble(
        payload:   TrendFoundPayload,
        validated: dict[str, Any],
        raw_json:  str,
    ) -> ViralScript:
        """Builds the final ViralScript from validated parts."""
        hook = validated["hook"]
        body = validated["body"]
        cta  = validated["cta"]

        # Full script: hook → body points → CTA
        # Punctuation normalisation: ensure each sentence ends with punctuation
        def _ensure_punct(s: str) -> str:
            s = s.strip()
            return s if s and s[-1] in ".!?," else s + "."

        full_parts = [_ensure_punct(hook)]
        full_parts += [_ensure_punct(b) for b in body]
        full_parts.append(_ensure_punct(cta))
        full_script = " ".join(full_parts)

        return ViralScript(
            trend_id     = payload.trend_id,
            topic        = payload.topic,
            hook_type    = payload.hook_type,
            hook_count   = payload.hook_count,
            hook         = hook,
            body         = body,
            cta          = cta,
            full_script  = full_script,
            raw_llm_json = raw_json,
        )

    # ── Jinja2 env ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_jinja_env() -> Environment:
        if not _TEMPLATES_DIR.exists():
            raise ScriptGenerationError(
                f"Prompts directory missing: {_TEMPLATES_DIR}. "
                "Expected: config/prompts/viral_script.j2"
            )
        return Environment(
            loader         = FileSystemLoader(str(_TEMPLATES_DIR)),
            trim_blocks    = True,
            lstrip_blocks  = True,
            keep_trailing_newline = False,
        )
