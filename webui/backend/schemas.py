"""
webui/backend/schemas.py

Pydantic request/response models — the API's wire contract. Kept separate
from job_manager.py's Job/JobEvent dataclasses, which carry server-only
state (asyncio.Queue subscriber lists) that must never be serialized.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestJobRequest(BaseModel):
    """Mirrors --ingest-vod's parameters (apex.py's run_ingest_vod)."""

    video_path: str = Field(..., description="Local filesystem path to the source VOD.")
    streamer_id: str
    chat_log_path: "str | None" = None
    subtitle_track_path: "str | None" = None
    vod_id: "str | None" = None
    drain_timeout: "float | None" = None


class ManualWindowJobRequest(BaseModel):
    """Mirrors --render-manual-window's parameters (apex.py's run_render_manual_window)."""

    video_path: str
    streamer_id: str
    start: float
    end: float
    vod_id: "str | None" = None
    drain_timeout: "float | None" = None


class JobCreateResponse(BaseModel):
    job_id: str


class JobEventResponse(BaseModel):
    type: str
    stage: str
    message: str
    data: "dict[str, Any] | None" = None
    ts: str


class JobStatusResponse(BaseModel):
    job_id: str
    kind: Literal["ingest", "manual_window"]
    params: "dict[str, Any]"
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    output_paths: "list[str]"
    error: "str | None" = None
    events: "list[JobEventResponse]" = []


class UploadResponse(BaseModel):
    path: str
