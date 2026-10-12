"""
tests/test_webui_api.py — webui/backend/main.py (HTTP + WebSocket layer)
and webui/backend/job_runner.py's path-validation reuse.

Fully offline. Two kinds of tests here:

1. Route-level tests build the FastAPI app via create_app(job_manager=...)
   with a JobManager wired to FAKE runners (never apex.run_ingest_vod()/
   run_render_manual_window()) — same "never depend on a live [...] call"
   rule CLAUDE.md applies everywhere else, applied to HTTP/WS routing.
2. job_runner reuse tests call run_ingest_job()/run_manual_window_job()
   directly (bypassing the API entirely) against a nonexistent path, and
   assert the raised error is exactly apex._resolve_ingest_paths()'s /
   apex._resolve_manual_video_path()'s own FileNotFoundError text — proof
   this module reuses those functions rather than re-implementing path
   existence checks.

No pytest-asyncio in this project — async bodies use asyncio.run(run()),
matching every other test file's convention.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

import apex
from webui.backend.job_manager import Job, JobManager
from webui.backend.job_runner import run_ingest_job, run_manual_window_job
from webui.backend.main import create_app

# ── Fakes ──────────────────────────────────────────────────────────────────────


def _make_runner(*, outputs=None, fail=False):
    async def _runner(job, emit) -> "list[str]":
        await emit("log", "log", "fake stage narration")
        if fail:
            raise RuntimeError("boom")
        return outputs if outputs is not None else ["/tmp/fake_output.mp4"]

    return _runner


def _client(*, ingest_outputs=None, ingest_fail=False) -> TestClient:
    """
    A client pre-armed with the app's own X-Apex-Local-Token — auth
    itself is tests/test_webui_auth.py's job; every test in *this* file is
    exercising something else and shouldn't have to fight the 401s that
    are the whole point of that other file.
    """
    jm = JobManager(_make_runner(outputs=ingest_outputs, fail=ingest_fail), _make_runner())
    app = create_app(job_manager=jm)
    client = TestClient(app)
    client.headers["X-Apex-Local-Token"] = app.state.local_token
    return client


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    waited = 0.0
    while not predicate():
        if waited >= timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.01)
        waited += 0.01


# ── Health ────────────────────────────────────────────────────────────────────


def test_health() -> None:
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Job creation ─────────────────────────────────────────────────────────────


def test_create_ingest_job_returns_202_and_job_id() -> None:
    client = _client()
    resp = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"})
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body and body["job_id"]


def test_create_manual_window_job_returns_202_and_job_id() -> None:
    client = _client()
    resp = client.post(
        "/jobs/manual-window",
        json={"video_path": "irrelevant.mp4", "streamer_id": "s1", "start": 1.0, "end": 20.0},
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"]


def test_ingest_job_request_requires_streamer_id() -> None:
    client = _client()
    resp = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4"})
    assert resp.status_code == 422


# ── Job status / list / not-found ────────────────────────────────────────────


def test_get_job_reflects_eventual_success() -> None:
    client = _client(ingest_outputs=["/tmp/clip.mp4"])
    job_id = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"}).json()["job_id"]

    async def run() -> None:
        jm: JobManager = client.app.state.job_manager
        await _wait_until(lambda: jm.get(job_id).status in ("succeeded", "failed"))

    asyncio.run(run())

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["output_paths"] == ["/tmp/clip.mp4"]
    assert any(e["stage"] == "log" for e in body["events"])


def test_get_unknown_job_is_404() -> None:
    client = _client()
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_list_jobs_includes_created_job_without_events() -> None:
    client = _client()
    job_id = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"}).json()["job_id"]

    resp = client.get("/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert any(j["job_id"] == job_id for j in jobs)
    assert all(j["events"] == [] for j in jobs)  # list view is deliberately light


# ── Download ─────────────────────────────────────────────────────────────────


def test_download_before_job_succeeds_is_409() -> None:
    # A runner that never completes within the test — status stays "running".
    async def _hang_runner(job, emit):
        await asyncio.Event().wait()

    jm = JobManager(_hang_runner, _hang_runner)
    app = create_app(job_manager=jm)
    client = TestClient(app)
    client.headers["X-Apex-Local-Token"] = app.state.local_token
    job_id = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"}).json()["job_id"]

    resp = client.get(f"/jobs/{job_id}/download")
    assert resp.status_code == 409


def test_download_unknown_job_is_404() -> None:
    client = _client()
    resp = client.get("/jobs/does-not-exist/download")
    assert resp.status_code == 404


def test_download_serves_real_output_file(tmp_path) -> None:
    output_file = tmp_path / "clip.mp4"
    output_file.write_bytes(b"fake mp4 bytes")

    client = _client(ingest_outputs=[str(output_file)])
    job_id = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"}).json()["job_id"]

    async def run() -> None:
        jm: JobManager = client.app.state.job_manager
        await _wait_until(lambda: jm.get(job_id).status == "succeeded")

    asyncio.run(run())

    resp = client.get(f"/jobs/{job_id}/download")
    assert resp.status_code == 200
    assert resp.content == b"fake mp4 bytes"


# ── WebSocket ────────────────────────────────────────────────────────────────


def test_websocket_streams_events_and_closes_after_terminal_status() -> None:
    client = _client(ingest_outputs=["/tmp/clip.mp4"])
    job_id = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"}).json()["job_id"]

    with client.websocket_connect(
        f"/jobs/{job_id}/status",
        headers={"origin": "http://127.0.0.1:5173"},
    ) as ws:
        stages = []
        while True:
            msg = ws.receive_json()
            stages.append(msg["stage"])
            if msg["type"] == "status" and msg["stage"] in ("succeeded", "failed"):
                break

    assert stages[0] == "queued"
    assert stages[-1] == "succeeded"
    assert "log" in stages


def test_websocket_on_unknown_job_closes_immediately() -> None:
    client = _client()
    with pytest.raises(Exception):
        with client.websocket_connect(
            "/jobs/does-not-exist/status",
            headers={"origin": "http://127.0.0.1:5173"},
        ) as ws:
            ws.receive_json()


def test_websocket_rejects_untrusted_origin() -> None:
    client = _client()
    with pytest.raises(Exception):
        with client.websocket_connect(
            "/jobs/does-not-exist/status",
            headers={"origin": "http://evil.example"},
        ):
            pass


# ── Upload (secondary path) ──────────────────────────────────────────────────


def test_upload_saves_file_and_returns_a_real_path() -> None:
    client = _client()
    resp = client.post("/uploads", files={"file": ("clip.mp4", b"small clip bytes", "video/mp4")})
    assert resp.status_code == 200
    saved_path = Path(resp.json()["path"])
    try:
        assert saved_path.is_file()
        assert saved_path.read_bytes() == b"small clip bytes"
        assert saved_path.name.endswith("clip.mp4")
    finally:
        saved_path.unlink(missing_ok=True)


# ── job_runner.py reuses apex.py's own path validation (not duplicated) ──────


def test_run_ingest_job_reuses_resolve_ingest_paths_error_text() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.mp4")
            job = Job(job_id="t1", kind="ingest", params={"video_path": missing, "streamer_id": "s1"})

            async def emit(*args, **kwargs) -> None:
                pass

            with pytest.raises(ValueError) as excinfo:
                await run_ingest_job(job, emit)

            # Must be apex._resolve_ingest_paths()'s own message, not a re-implementation.
            try:
                apex._resolve_ingest_paths(missing, None)
            except FileNotFoundError as expected:
                assert str(expected) == str(excinfo.value)

    asyncio.run(run())


def test_run_manual_window_job_reuses_resolve_manual_video_path_error_text() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "does_not_exist.mp4")
            job = Job(
                job_id="t2", kind="manual_window",
                params={"video_path": missing, "streamer_id": "s1", "start": 1.0, "end": 20.0},
            )

            async def emit(*args, **kwargs) -> None:
                pass

            with pytest.raises(ValueError) as excinfo:
                await run_manual_window_job(job, emit)

            try:
                apex._resolve_manual_video_path(missing)
            except FileNotFoundError as expected:
                assert str(expected) == str(excinfo.value)

    asyncio.run(run())
