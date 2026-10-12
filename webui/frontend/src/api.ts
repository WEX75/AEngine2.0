// src/api.ts — thin client for webui/backend/main.py. No framework, just
// fetch + a raw WebSocket, matching the backend's own "no more than it
// needs" approach.

export const API_BASE = "http://127.0.0.1:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

export type JobKind = "ingest" | "manual_window";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface JobEvent {
  type: "status" | "log" | "milestone";
  stage: string;
  message: string;
  data?: Record<string, unknown> | null;
  ts: string;
}

export interface JobStatusResponse {
  job_id: string;
  kind: JobKind;
  params: Record<string, unknown>;
  status: JobStatus;
  created_at: string;
  output_paths: string[];
  error?: string | null;
  events: JobEvent[];
}

export interface IngestJobRequest {
  video_path: string;
  streamer_id: string;
  chat_log_path?: string | null;
  subtitle_track_path?: string | null;
  vod_id?: string | null;
  drain_timeout?: number | null;
}

export interface ManualWindowJobRequest {
  video_path: string;
  streamer_id: string;
  start: number;
  end: number;
  vod_id?: string | null;
  drain_timeout?: number | null;
}

async function asJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // body wasn't JSON — fall back to statusText
    }
    throw new Error(`${resp.status} ${detail}`);
  }
  return resp.json() as Promise<T>;
}

// The backend requires this on every state-changing route (POST /jobs/*,
// POST /uploads) — see webui/backend/auth.py's module docstring for why
// CORS's allow_origins alone doesn't stop a blind cross-origin POST.
// Fetched once from GET /auth/token (itself unauthenticated — there's no
// token to present until this returns one; CORS still keeps a disallowed
// origin from reading the response) and cached for the tab's lifetime.
const TOKEN_HEADER = "X-Apex-Local-Token";
let tokenPromise: Promise<string> | null = null;

function getToken(): Promise<string> {
  if (!tokenPromise) {
    tokenPromise = fetch(`${API_BASE}/auth/token`)
      .then((resp) => asJson<{ token: string }>(resp))
      .then((body) => body.token);
  }
  return tokenPromise;
}

async function authHeaders(extra?: Record<string, string>): Promise<Record<string, string>> {
  return { [TOKEN_HEADER]: await getToken(), ...extra };
}

export async function submitIngestJob(req: IngestJobRequest): Promise<{ job_id: string }> {
  const resp = await fetch(`${API_BASE}/jobs/ingest`, {
    method: "POST",
    headers: await authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(req),
  });
  return asJson(resp);
}

export async function submitManualWindowJob(req: ManualWindowJobRequest): Promise<{ job_id: string }> {
  const resp = await fetch(`${API_BASE}/jobs/manual-window`, {
    method: "POST",
    headers: await authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(req),
  });
  return asJson(resp);
}

export async function uploadFile(file: File): Promise<{ path: string }> {
  const form = new FormData();
  form.append("file", file);
  // No Content-Type header here on purpose — fetch sets the multipart
  // boundary itself; overriding it breaks the upload.
  const resp = await fetch(`${API_BASE}/uploads`, { method: "POST", headers: await authHeaders(), body: form });
  return asJson(resp);
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  const resp = await fetch(`${API_BASE}/jobs/${jobId}`);
  return asJson(resp);
}

export async function listJobs(): Promise<JobStatusResponse[]> {
  const resp = await fetch(`${API_BASE}/jobs`);
  return asJson(resp);
}

export function downloadUrl(jobId: string, index = 0): string {
  return `${API_BASE}/jobs/${jobId}/download?index=${index}`;
}

export function openJobSocket(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onClose: () => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/jobs/${jobId}/status`);
  ws.onmessage = (message) => onEvent(JSON.parse(message.data as string) as JobEvent);
  ws.onclose = () => onClose();
  ws.onerror = () => onClose();
  return ws;
}
