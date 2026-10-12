"""
brain/ollama_client.py  —  The Brain: Async Ollama Client
Apex Engine 2.0  |  Pillar 2

Communicates with a local Ollama instance via its REST API using aiohttp.
Designed for AMD Ryzen hardware (CPU-only inference, 10–30s per prompt).

Key design decisions
--------------------
* Non-streaming (`"stream": false`) — waits for the complete JSON response
  in a single HTTP round-trip. Simpler than streaming for structured output.
* `format: "json"` — instructs Ollama to force the model's output into valid
  JSON, preventing markdown fences and prose contaminating the response.
* All async so the EventBus event loop stays completely free during inference.
* Configurable timeout with a hard outer cap so a stalled model never blocks
  the system indefinitely.
* Connection-refused errors produce a human-readable message pointing the
  user to `ollama serve`.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from core.config_loader import get_config
from core.logger import get_logger

log = get_logger(__name__)

# Hard cap: if Ollama hasn't responded in this long, give up.
# Set generously for slow CPU inference; configurable via settings.yaml.
_ABSOLUTE_TIMEOUT_S: float = 300.0  # 5 minutes — should never be hit in practice


class OllamaError(RuntimeError):
    """Raised when Ollama returns an error or times out."""


class OllamaClient:
    """
    Async HTTP client for the Ollama REST API.

    Usage (from an async context)
    -----------------------------
        client = OllamaClient()
        result = await client.generate(prompt="...", model="llama3:8b")
        text = result["text"]   # The model's raw text output
    """

    def __init__(self) -> None:
        cfg = get_config()
        brain_cfg = cfg.get("brain", {})

        self._host:    str   = brain_cfg.get("ollama_host", "http://localhost:11434")
        self._model:   str   = brain_cfg.get("model",       "llama3:8b")
        self._temp:    float = brain_cfg.get("temperature",  0.85)
        self._max_tok: int   = brain_cfg.get("max_tokens",   500)
        self._timeout: float = min(
            brain_cfg.get("timeout_seconds", 120),
            _ABSOLUTE_TIMEOUT_S,
        )

        self._endpoint = f"{self._host.rstrip('/')}/api/generate"

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt:      str,
        model:       str  | None = None,
        temperature: float| None = None,
        max_tokens:  int  | None = None,
        expect_json: bool        = True,
        num_gpu:     int  | None = None,
    ) -> dict[str, Any]:
        """
        Send a generation request to Ollama and return a parsed result dict.

        Parameters
        ----------
        prompt       : The full prompt string.
        model        : Override the configured model for this call.
        temperature  : Override the configured temperature.
        max_tokens   : Override the configured max_tokens.
        expect_json  : If True, adds `"format": "json"` to the payload so
                       Ollama forces the model to output valid JSON.
        num_gpu      : Override how many layers Ollama offloads to GPU for
                       this call (0 forces CPU-only). Left unset by default
                       so the server's own GPU configuration decides.

        Returns
        -------
        dict with keys:
            "text"   — raw text from the model (JSON string if expect_json)
            "model"  — model name used
            "done"   — True if generation finished normally
            "tokens" — approximate token count (if reported by Ollama)

        Raises
        ------
        OllamaError  on connection failure, timeout, or HTTP error.
        """
        payload = self._build_payload(prompt, model, temperature, max_tokens, expect_json, num_gpu)

        log.info(
            f"[OllamaClient] Sending request → {self._endpoint} "
            f"(model={payload['model']}, timeout={self._timeout:.0f}s)"
        )

        connector = aiohttp.TCPConnector(limit=1)  # one concurrent call — respects AMD RAM
        timeout   = aiohttp.ClientTimeout(total=self._timeout)

        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            ) as session:
                async with session.post(
                    self._endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    return await self._handle_response(resp)

        except aiohttp.ClientConnectorError as exc:
            raise OllamaError(
                f"Cannot connect to Ollama at '{self._host}'. "
                "Make sure Ollama is running: open a terminal and type 'ollama serve'. "
                f"(detail: {exc})"
            ) from exc

        except asyncio.TimeoutError:
            raise OllamaError(
                f"Ollama did not respond within {self._timeout:.0f}s. "
                "The model may be too large for your hardware. "
                "Try: ollama pull llama3:8b  to ensure the model is cached locally."
            )

        except aiohttp.ClientError as exc:
            raise OllamaError(f"HTTP client error communicating with Ollama: {exc}") from exc

    async def health_check(self) -> bool:
        """
        Returns True if Ollama is reachable, False otherwise.
        Does NOT raise — safe to call in startup checks.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self._host.rstrip('/')}/api/tags") as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        prompt:      str,
        model:       str   | None,
        temperature: float | None,
        max_tokens:  int   | None,
        expect_json: bool,
        num_gpu:     int   | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":  model       or self._model,
            "prompt": prompt,
            "stream": False,      # collect the full response in one shot
            "options": {
                "temperature": temperature or self._temp,
                "num_predict": max_tokens  or self._max_tok,
                "stop": [],       # let the model decide when to stop
            },
        }
        if expect_json:
            payload["format"] = "json"
        if num_gpu is not None:
            payload["options"]["num_gpu"] = num_gpu
        return payload

    @staticmethod
    async def _handle_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
        """Parse and validate the Ollama HTTP response."""
        if resp.status != 200:
            body = await resp.text()
            raise OllamaError(
                f"Ollama returned HTTP {resp.status}. Body: {body[:300]}"
            )

        try:
            data: dict = await resp.json(content_type=None)
        except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
            raw = await resp.text()
            raise OllamaError(
                f"Ollama response was not valid JSON: {exc}. Raw: {raw[:200]}"
            ) from exc

        if "error" in data:
            raise OllamaError(f"Ollama error: {data['error']}")

        text = data.get("response", "")
        if not text:
            raise OllamaError("Ollama returned an empty response field.")

        log.info(
            f"[OllamaClient] Response received — "
            f"{len(text)} chars, done={data.get('done', '?')}"
        )

        return {
            "text":   text,
            "model":  data.get("model", ""),
            "done":   data.get("done", False),
            "tokens": data.get("eval_count", 0),
        }
