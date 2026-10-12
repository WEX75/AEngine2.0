"""
tests/test_webui_auth.py — webui/backend/auth.py + main.py's wiring of it.

Covers the gap flagged before this went anywhere near a commit: CORS's
allow_origins only blocks a browser from *reading* a cross-origin
response, not from *sending* a simple cross-origin request — so an
unauthenticated POST /jobs/ingest could be triggered blind by a page open
in some other tab. Fixed with a per-process shared-secret header
(X-Apex-Local-Token) on every state-changing route, backed by a CORS
policy restricted to exactly one origin.

Fully offline — a fake JobManager (never apex.run_ingest_vod()/
run_render_manual_window()), same discipline as test_webui_api.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from webui.backend.job_manager import JobManager
from webui.backend.main import ALLOWED_ORIGINS, create_app


async def _noop_runner(job, emit) -> "list[str]":
    return ["/tmp/fake_output.mp4"]


def _client() -> TestClient:
    jm = JobManager(_noop_runner, _noop_runner)
    app = create_app(job_manager=jm)
    return TestClient(app)


def _token(client: TestClient) -> str:
    resp = client.get("/auth/token")
    assert resp.status_code == 200
    return resp.json()["token"]


# ── Token endpoint ────────────────────────────────────────────────────────────


def test_auth_token_endpoint_returns_a_nonempty_token() -> None:
    client = _client()
    resp = client.get("/auth/token")
    assert resp.status_code == 200
    token = resp.json()["token"]
    assert isinstance(token, str) and len(token) > 20


def test_each_app_instance_gets_its_own_token() -> None:
    token_a = _token(_client())
    token_b = _token(_client())
    assert token_a != token_b


# ── State-changing routes require the token ─────────────────────────────────


def test_create_ingest_job_without_token_is_401() -> None:
    client = _client()
    resp = client.post("/jobs/ingest", json={"video_path": "irrelevant.mp4", "streamer_id": "s1"})
    assert resp.status_code == 401


def test_create_ingest_job_with_wrong_token_is_401() -> None:
    client = _client()
    resp = client.post(
        "/jobs/ingest",
        json={"video_path": "irrelevant.mp4", "streamer_id": "s1"},
        headers={"X-Apex-Local-Token": "not-the-real-token"},
    )
    assert resp.status_code == 401


def test_create_ingest_job_with_correct_token_succeeds() -> None:
    client = _client()
    token = _token(client)
    resp = client.post(
        "/jobs/ingest",
        json={"video_path": "irrelevant.mp4", "streamer_id": "s1"},
        headers={"X-Apex-Local-Token": token},
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"]


def test_create_manual_window_job_without_token_is_401() -> None:
    client = _client()
    resp = client.post(
        "/jobs/manual-window",
        json={"video_path": "irrelevant.mp4", "streamer_id": "s1", "start": 1.0, "end": 20.0},
    )
    assert resp.status_code == 401


def test_create_manual_window_job_with_correct_token_succeeds() -> None:
    client = _client()
    token = _token(client)
    resp = client.post(
        "/jobs/manual-window",
        json={"video_path": "irrelevant.mp4", "streamer_id": "s1", "start": 1.0, "end": 20.0},
        headers={"X-Apex-Local-Token": token},
    )
    assert resp.status_code == 202


def test_upload_without_token_is_401() -> None:
    client = _client()
    resp = client.post("/uploads", files={"file": ("clip.mp4", b"bytes", "video/mp4")})
    assert resp.status_code == 401


def test_upload_with_correct_token_succeeds() -> None:
    client = _client()
    token = _token(client)
    resp = client.post(
        "/uploads",
        files={"file": ("clip.mp4", b"bytes", "video/mp4")},
        headers={"X-Apex-Local-Token": token},
    )
    assert resp.status_code == 200
    Path(resp.json()["path"]).unlink(missing_ok=True)


# ── Read-only routes stay open (no token needed) ─────────────────────────────


def test_health_does_not_require_token() -> None:
    client = _client()
    assert client.get("/health").status_code == 200


def test_list_and_get_job_do_not_require_token() -> None:
    client = _client()
    token = _token(client)
    job_id = client.post(
        "/jobs/ingest",
        json={"video_path": "irrelevant.mp4", "streamer_id": "s1"},
        headers={"X-Apex-Local-Token": token},
    ).json()["job_id"]

    assert client.get("/jobs").status_code == 200
    assert client.get(f"/jobs/{job_id}").status_code == 200


# ── CORS is locked to exactly one origin ─────────────────────────────────────


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_cors_preflight_allows_the_configured_frontend_origins(origin: str) -> None:
    client = _client()
    resp = client.options(
        "/jobs/ingest",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-apex-local-token,content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == origin


def test_cors_preflight_rejects_an_unrelated_origin() -> None:
    client = _client()
    resp = client.options(
        "/jobs/ingest",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-apex-local-token,content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example"
