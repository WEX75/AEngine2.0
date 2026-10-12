import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  getJob,
  listJobs,
  openJobSocket,
  submitIngestJob,
  submitManualWindowJob,
  uploadFile,
  type JobEvent,
  type JobStatusResponse,
} from "./api";
import { SignalChain } from "./components/SignalChain";
import { JobHistory } from "./components/JobHistory";

type SourceMode = "path" | "upload";
type RunMode = "auto" | "manual";

function applyEvent(job: JobStatusResponse, event: JobEvent): JobStatusResponse {
  const events = [...job.events, event];
  let status = job.status;
  let output_paths = job.output_paths;
  if (event.type === "status") {
    if (event.stage === "queued" || event.stage === "running" || event.stage === "succeeded" || event.stage === "failed") {
      status = event.stage;
    }
    if (event.stage === "succeeded" && event.data && Array.isArray(event.data.output_paths)) {
      output_paths = event.data.output_paths as string[];
    }
  }
  return { ...job, events, status, output_paths };
}

function shortName(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export default function App() {
  // ── Backend liveness ────────────────────────────────────────────
  const [backendUp, setBackendUp] = useState(false);

  useEffect(() => {
    let alive = true;
    const check = () =>
      fetch(`${API_BASE}/health`)
        .then((r) => alive && setBackendUp(r.ok))
        .catch(() => alive && setBackendUp(false));
    check();
    const id = setInterval(check, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // ── Source ──────────────────────────────────────────────────────
  const [sourceMode, setSourceMode] = useState<SourceMode>("path");
  const [videoPath, setVideoPath] = useState("");
  const [uploadBusy, setUploadBusy] = useState(false);

  async function handleFileChosen(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploadBusy(true);
    setSubmitError(null);
    try {
      const { path } = await uploadFile(file);
      setVideoPath(path);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploadBusy(false);
    }
  }

  // ── Job configuration ───────────────────────────────────────────
  const [streamerId, setStreamerId] = useState("");
  const [mode, setMode] = useState<RunMode>("auto");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [vodId, setVodId] = useState("");
  const [chatLogPath, setChatLogPath] = useState("");
  const [subtitleTrackPath, setSubtitleTrackPath] = useState("");
  const [drainTimeout, setDrainTimeout] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // ── Jobs (session-only history + live state) ────────────────────
  const [jobsById, setJobsById] = useState<Record<string, JobStatusResponse>>({});
  const [jobOrder, setJobOrder] = useState<string[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const connected = useRef<Set<string>>(new Set());

  const jobs = useMemo(() => jobOrder.map((id) => jobsById[id]).filter(Boolean), [jobOrder, jobsById]);
  const selectedJob = selectedJobId ? jobsById[selectedJobId] ?? null : null;

  function connectJob(jobId: string) {
    if (connected.current.has(jobId)) return;
    connected.current.add(jobId);
    openJobSocket(
      jobId,
      (event) => {
        setJobsById((prev) => {
          const existing = prev[jobId];
          if (!existing) return prev;
          return { ...prev, [jobId]: applyEvent(existing, event) };
        });
      },
      () => {
        connected.current.delete(jobId);
        // Reconcile once the stream closes — picks up job.error, the one
        // field the live event stream doesn't carry structured data for.
        getJob(jobId)
          .then((full) => setJobsById((prev) => ({ ...prev, [jobId]: full })))
          .catch(() => {});
      },
    );
  }

  // Resume any jobs already known to the backend (same server process,
  // e.g. after a page reload) and reconnect to whichever is still live.
  useEffect(() => {
    listJobs()
      .then((list) => {
        const byId: Record<string, JobStatusResponse> = {};
        const order: string[] = [];
        for (const job of list) {
          byId[job.job_id] = job;
          order.push(job.job_id);
        }
        setJobsById(byId);
        setJobOrder(order);
        for (const job of list) {
          if (job.status === "queued" || job.status === "running") connectJob(job.job_id);
        }
        if (order.length > 0) handleSelect(order[order.length - 1]);
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSelect(jobId: string) {
    setSelectedJobId(jobId);
    // Always refetch the full record (never trust a possibly-stale
    // `jobsById` closure here — this runs synchronously right after
    // setJobsById() during mount hydration too). GET /jobs/{id} always
    // includes events, unlike the list endpoint (GET /jobs), which
    // deliberately omits them to stay light — so this is also what fills
    // in the signal chain / log feed for a job that was hydrated from the
    // list (on mount) or re-opened from the reel log after finishing.
    getJob(jobId)
      .then((full) => {
        setJobsById((prev) => ({ ...prev, [jobId]: full }));
        if (full.status === "queued" || full.status === "running") connectJob(jobId);
      })
      .catch(() => {});
  }

  async function handleRun() {
    setSubmitError(null);
    const path = videoPath.trim();
    const streamer = streamerId.trim();
    if (!path) {
      setSubmitError("Provide a source path, or upload a file, first.");
      return;
    }
    if (!streamer) {
      setSubmitError("Streamer ID is required.");
      return;
    }

    setSubmitting(true);
    try {
      const timeout = drainTimeout.trim() ? Number(drainTimeout) : undefined;
      let jobId: string;
      if (mode === "auto") {
        const resp = await submitIngestJob({
          video_path: path,
          streamer_id: streamer,
          chat_log_path: chatLogPath.trim() || undefined,
          subtitle_track_path: subtitleTrackPath.trim() || undefined,
          vod_id: vodId.trim() || undefined,
          drain_timeout: timeout,
        });
        jobId = resp.job_id;
      } else {
        const startNum = Number(start);
        const endNum = Number(end);
        if (!Number.isFinite(startNum) || !Number.isFinite(endNum)) {
          throw new Error("Start/end must be numbers, in seconds.");
        }
        const resp = await submitManualWindowJob({
          video_path: path,
          streamer_id: streamer,
          start: startNum,
          end: endNum,
          vod_id: vodId.trim() || undefined,
          drain_timeout: timeout,
        });
        jobId = resp.job_id;
      }

      const initial = await getJob(jobId);
      setJobsById((prev) => ({ ...prev, [jobId]: initial }));
      setJobOrder((prev) => [...prev, jobId]);
      setSelectedJobId(jobId);
      connectJob(jobId);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  const isBusy = submitting || (selectedJob !== null && jobs.some((j) => j.status === "running"));

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead__wordmark">
          <span className="masthead__tally" />
          Apex Engine <span>Control Panel</span>
        </div>
        <div className="masthead__status">
          <span className="masthead__status-dot" data-live={backendUp} />
          {backendUp ? "backend online" : "backend unreachable"}
        </div>
      </header>

      <div className="module">
        <div className="module__eyebrow">Source</div>
        <div className="source__tabs">
          <button className="source__tab" data-active={sourceMode === "path"} onClick={() => setSourceMode("path")}>
            Path
          </button>
          <button className="source__tab" data-active={sourceMode === "upload"} onClick={() => setSourceMode("upload")}>
            Upload
          </button>
        </div>

        {sourceMode === "path" ? (
          <div className="field">
            <label className="field__label">Video path</label>
            <input
              className="field__input"
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              placeholder="/path/to/video.mp4"
            />
            <div className="field__hint">
              A real local filesystem path — the primary way to select a source. Large VODs are never
              re-uploaded over HTTP.
            </div>
          </div>
        ) : (
          <div className="field">
            <label className="field__label">Upload a clip</label>
            <div className="source__upload">
              <label className="source__upload-btn">
                {uploadBusy ? "Uploading…" : "Choose file"}
                <input type="file" accept="video/*" hidden disabled={uploadBusy} onChange={handleFileChosen} />
              </label>
              {videoPath && <span className="source__upload-status">→ {shortName(videoPath)}</span>}
            </div>
            <div className="field__hint">Secondary path, for smaller clips only — large VODs should use Path instead.</div>
          </div>
        )}
      </div>

      <div className="module">
        <div className="module__eyebrow">Job</div>

        <div className="field">
          <label className="field__label">Streamer ID</label>
          <input
            className="field__input"
            value={streamerId}
            onChange={(e) => setStreamerId(e.target.value)}
            placeholder="streamer-1"
          />
        </div>

        <div className="mode">
          <button className="mode__btn" data-active={mode === "auto"} onClick={() => setMode("auto")}>
            <span className="mode__dot" /> Auto-detect
          </button>
          <button className="mode__btn" data-active={mode === "manual"} onClick={() => setMode("manual")}>
            <span className="mode__dot" /> Manual window
          </button>
        </div>

        {mode === "manual" && (
          <div className="field__row">
            <div className="field">
              <label className="field__label">Start (sec)</label>
              <input className="field__input" value={start} onChange={(e) => setStart(e.target.value)} placeholder="58.0" inputMode="decimal" />
            </div>
            <div className="field">
              <label className="field__label">End (sec)</label>
              <input className="field__input" value={end} onChange={(e) => setEnd(e.target.value)} placeholder="74.0" inputMode="decimal" />
            </div>
          </div>
        )}

        <details className="advanced">
          <summary>Advanced</summary>
          <div className="field">
            <label className="field__label">VOD ID override</label>
            <input
              className="field__input"
              value={vodId}
              onChange={(e) => setVodId(e.target.value)}
              placeholder="defaults to the filename stem"
            />
          </div>
          {mode === "auto" && (
            <>
              <div className="field">
                <label className="field__label">Chat log path (.jsonl)</label>
                <input
                  className="field__input"
                  value={chatLogPath}
                  onChange={(e) => setChatLogPath(e.target.value)}
                  placeholder="omit for chat-less content"
                />
              </div>
              <div className="field">
                <label className="field__label">Subtitle track path (.vtt / .srt)</label>
                <input
                  className="field__input"
                  value={subtitleTrackPath}
                  onChange={(e) => setSubtitleTrackPath(e.target.value)}
                  placeholder="required if no chat log"
                />
              </div>
            </>
          )}
          <div className="field">
            <label className="field__label">Drain timeout (sec)</label>
            <input
              className="field__input"
              value={drainTimeout}
              onChange={(e) => setDrainTimeout(e.target.value)}
              placeholder="1800"
              inputMode="numeric"
            />
          </div>
        </details>

        <div className="run-row">
          <button className="run-btn" data-pulsing={isBusy} disabled={submitting} onClick={handleRun}>
            <span className="run-btn__lamp" />
            {submitting ? "STARTING…" : "RUN"}
          </button>
        </div>
        {submitError && <div className="run-error">{submitError}</div>}
      </div>

      <SignalChain job={selectedJob} />

      <JobHistory jobs={jobs} selectedId={selectedJobId} onSelect={handleSelect} />
    </div>
  );
}
