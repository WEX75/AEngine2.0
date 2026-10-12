"""
webui/backend/auth.py

Closes a real gap in a "local-only, no auth" API: CORS's allow_origins
only stops a browser from *reading* a cross-origin response — it does
nothing to stop the browser from *sending* a "simple" cross-origin
request in the first place (plain GET/POST, no custom headers). A
malicious page open in some other tab could blind-POST straight to
POST /jobs/ingest or POST /uploads and trigger a real local render with
no way for the operator to see it coming; unconfigured/permissive CORS
does not protect against this, contrary to how it's often assumed to.

Fix: a random per-process token, required as a custom header
(X-Apex-Local-Token) on every state-changing route. Adding ANY custom
header turns a request into a "not simple" one per the CORS spec, which
forces the browser to run an OPTIONS preflight first — and our
CORSMiddleware only answers that preflight successfully for the one
allowed origin (see main.py's ALLOWED_ORIGINS), so a page loaded from
anywhere else never even gets to send the real POST. The token's actual
value is a second, independent layer on top of that (defense in depth —
protects against a future CORS misconfiguration, not just today's).

Scope, deliberately: only the state-changing routes (POST /jobs/*,
POST /uploads) require this. GET routes stay open — they're read-only,
and CORS already blocks a disallowed origin from reading their responses.
Not in scope here either: non-browser local processes calling the API
directly (curl, another local program) — CORS/Origin checks don't apply
to those at all, and defending against arbitrary co-resident processes on
the same machine is a different problem (they can already read this
user's files directly) that a shared HTTP secret doesn't solve.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

TOKEN_HEADER_NAME = "X-Apex-Local-Token"


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def make_token_dependency(expected_token: str):
    """Returns a FastAPI dependency that 401s unless X-Apex-Local-Token matches."""

    async def require_local_token(x_apex_local_token: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_apex_local_token, expected_token):
            raise HTTPException(status_code=401, detail=f"missing or invalid {TOKEN_HEADER_NAME} header")

    return require_local_token
