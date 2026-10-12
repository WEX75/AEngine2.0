"""
brain/models.py  —  The Brain: Data Contracts
Apex Engine 2.0  |  Pillar 2

ViralScript is the intermediate output of ScriptGenerator.
It is then converted to ScriptReadyPayload before being published
to the EventBus so downstream modules (The Factory) can consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ViralScript:
    """
    A fully-structured viral video script produced by ScriptGenerator.

    Fields
    ------
    hook        First 3-second opener — aggressive, curiosity-driving.
    body        List of high-retention body points (length = hook_count).
    cta         Closing call-to-action or cliffhanger sentence.
    full_script Concatenated hook + body + cta, ready for TTS input.
    """
    trend_id:       str
    topic:          str
    hook_type:      str
    hook_count:     int

    hook:           str
    body:           list[str]
    cta:            str
    full_script:    str

    raw_llm_json:   str      = ""   # Raw Ollama response (kept for debugging)
    generated_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Convenience ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line human-readable summary for logging."""
        return (
            f"[{self.hook_type}] '{self.hook[:50]}' "
            f"+ {len(self.body)} body points + CTA"
        )

    def word_count(self) -> int:
        return len(self.full_script.split())

    def estimated_duration_seconds(self, wpm: int = 155) -> float:
        """Approximate TTS read time at typical voiceover pace."""
        return (self.word_count() / wpm) * 60
