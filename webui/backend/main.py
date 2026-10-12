"""
webui/backend/main.py

Local-only FastAPI control panel for apex.py's --ingest-vod /
--render-manual-window one-shot CLI entry points. Async-native throughout
(no sync/async impedance mismatch with the rest of the asyncio pillar
architecture) — routes return a job_id immediately, JobManager runs the
real pipeline call as a background asyncio task, and live progress is
pushed over a WebSocket. See job_manager.py / job_runner.py for the
execution model.

Placement — why webui/, not tools/ or a pillar directory
-----------------------------------------------------------
This isn't a pipeline pillar (Sentinel/Brain/Cutter own detection/
selection/render; this owns none of that — it only calls the existing
one-shot entry points) and it isn't a one-off manual script like
tools/manual_smoke_test_e2e.py either — it's a standing app with its own
backend + frontend subtrees. A new top-level `webui/` directory keeps
those together without overloading an existing pillar's namespace or
`tools/`'s "single manual script" convention, and reads naturally
alongside the `campaigns/` directory a later task is expected to add.

Run (from the project root, with the project's .venv active):
    uvicorn webui.backend.main:app --reload --host 127.0.0.1 --port 8000

Local-only: CORS and WebSocket connections are scoped to the Vite dev
server origins below, and the run command above binds to 127.0.0.1, not
0.0.0.0 — never expose this beyond localhost as-is, it can render/
download arbitrary local files. Every state-changing route also requires
a per-process shared-secret header (see auth.py's module docstring for
why CORS alone doesn't cover this) — the token is printed to the console
and written to .local_token next to this file on startup.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from webui.backend.auth import TOKEN_HEADER_NAME, generate_token, make_token_dependency
from webui.backend.job_manager import Job, JobManager
from webui.backend.job_runner import run_ingest_job, run_manual_window_job
from webui.backend.schemas import (
    IngestJobRequest,
    JobCreateResponse,
    JobStatusResponse,
    ManualWindowJobRequest,
    UploadResponse,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Secondary upload path lands next to the project's existing input/ media
# tree (see input/vods/ already in use), not a bespoke location.
_UPLOAD_DIR = _PROJECT_ROOT / "input" / "uploads"

ALLOWED_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")


def create_app(job_manager: "JobManager | None" = None) -> FastAPI:
    """
    Factory, not a bare module-level app, so tests can pass a JobManager
    wired to fake runners (no real ffmpeg/Whisper/Ollama — same offline
    testing discipline CLAUDE.md requires of every pillar) instead of the
    real one this module builds at import time below. Each call also gets
    its own fresh auth token — never shared across app instances.
    """
    jm = job_manager or JobManager(run_ingest_job, run_manual_window_job)
    token = generate_token()
    require_token = make_token_dependency(token)

    app = FastAPI(title="Apex Engine 2.0 — Control Panel")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.job_manager = jm
    app.state.local_token = token

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/auth/token")
    async def auth_token() -> dict:
        """
        Bootstrap endpoint, deliberately unauthenticated (there's no token
        to present until this returns one). Safe: CORS still stops a
        disallowed origin's JS from reading this response — see auth.py.
        """
        return {"token": token}

    @app.post("/jobs/ingest", response_model=JobCreateResponse, status_code=202, dependencies=[Depends(require_token)])
    async def create_ingest_job(req: IngestJobRequest) -> JobCreateResponse:
        job = jm.create_job("ingest", req.model_dump())
        return JobCreateResponse(job_id=job.job_id)

    @app.post(
        "/jobs/manual-window",
        response_model=JobCreateResponse,
        status_code=202,
        dependencies=[Depends(require_token)],
    )
    async def create_manual_window_job(req: ManualWindowJobRequest) -> JobCreateResponse:
        job = jm.create_job("manual_window", req.model_dump())
        return JobCreateResponse(job_id=job.job_id)

    @app.post("/uploads", response_model=UploadResponse, dependencies=[Depends(require_token)])
    async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
        """
        Secondary source-selection path, for smaller clips only (see
        main.py's module docstring / the control-panel spec) — the
        primary path is a local file path, which never needs an upload at
        all. Streams to disk in chunks rather than reading the whole
        upload into memory.
        """
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload").name  # strip any path components from the client-supplied name
        dest = _UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        await file.close()
        return UploadResponse(path=str(dest.resolve()))

    @app.get("/jobs", response_model=list[JobStatusResponse])
    async def list_jobs() -> "list[JobStatusResponse]":
        return [JobStatusResponse(**j.to_dict(include_events=False)) for j in jm.list_jobs()]

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        job = jm.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return JobStatusResponse(**job.to_dict(include_events=True))

    @app.get("/jobs/{job_id}/download")
    async def download_job_output(job_id: str, index: int = 0) -> FileResponse:
        job = jm.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        if job.status != "succeeded":
            raise HTTPException(status_code=409, detail=f"job status is {job.status!r}, not 'succeeded'")
        if not job.output_paths:
            raise HTTPException(status_code=404, detail="job succeeded but produced no output files")
        if index < 0 or index >= len(job.output_paths):
            raise HTTPException(
                status_code=404, detail=f"no output at index={index} (job has {len(job.output_paths)})"
            )
        path = Path(job.output_paths[index])
        if not path.is_file():
            raise HTTPException(status_code=410, detail=f"output file no longer exists on disk: {path}")
        return FileResponse(path, filename=path.name)

    @app.websocket("/jobs/{job_id}/status")
    async def job_status_ws(websocket: WebSocket, job_id: str) -> None:
        if websocket.headers.get("origin") not in ALLOWED_ORIGINS:
            await websocket.close(code=4403, reason="origin not allowed")
            return

        job = jm.get(job_id)
        if job is None:
            await websocket.close(code=4404, reason=f"no such job: {job_id}")
            return

        await websocket.accept()
        queue = jm.subscribe(job_id)
        try:
            while True:
                event = await queue.get()
                if event is None:  # terminal sentinel — job reached succeeded/failed
                    break
                await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            jm.unsubscribe(job_id, queue)

    return app


app = create_app()

# Real-deployment-only side effect (deliberately outside create_app() so
# every test app doesn't also spam this file/console — see create_app()'s
# docstring: each call gets its own token, tests read app.state.local_token
# or GET /auth/token directly instead).
_TOKEN_FILE = Path(__file__).resolve().parent / ".local_token"
_TOKEN_FILE.write_text(app.state.local_token, encoding="utf-8")
_TOKEN_FILE.chmod(0o600)
print(
    f"\n[Apex Control Panel] Local auth token (also written to {_TOKEN_FILE}):\n"
    f"  {app.state.local_token}\n"
    f"  Required as the '{TOKEN_HEADER_NAME}' header on POST /jobs/*, POST /uploads.\n"
)
