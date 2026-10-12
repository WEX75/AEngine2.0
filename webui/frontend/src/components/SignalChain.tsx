// SignalChain — the panel's signature element: a patch-bay style strip of
// LED "lamps" for each real pipeline stage, wired up from the job's own
// milestone events (EventBus payload names), plus a scrolling log feed
// underneath fed by the same log/milestone events. Nothing here invents
// progress that isn't real — a stage lamp only lights once its actual
// EventBus event (or, for --render-manual-window's stage-less middle,
// the job simply being "running") has been observed.
import { downloadUrl, type JobEvent, type JobStatusResponse } from "../api";

interface StageDef {
  key: string;
  label: string;
}

// Mirrors CLAUDE.md's event contracts exactly (STREAM_VOD_INGESTED ->
// MOMENT_DETECTED -> CLIP_SCRIPT_READY -> CLIP_RENDER_READY).
const INGEST_STAGES: StageDef[] = [
  { key: "STREAM_VOD_INGESTED", label: "Ingest" },
  { key: "MOMENT_DETECTED", label: "Detect" },
  { key: "CLIP_SCRIPT_READY", label: "Script" },
  { key: "CLIP_RENDER_READY", label: "Render" },
];

// --render-manual-window skips Sentinel/Brain entirely (apex.py's own
// header comment above _resolve_manual_video_path) — no
// STREAM_VOD_INGESTED / MOMENT_DETECTED / CLIP_SCRIPT_READY ever fires on
// this path, only CLIP_RENDER_READY at the end. "PROCESSING" is a
// synthetic node (no EventBus event backs it) standing in for the real
// extract+transcribe work apex.py narrates only through log lines.
const MANUAL_STAGES: StageDef[] = [
  { key: "PROCESSING", label: "Extract + Transcribe" },
  { key: "CLIP_RENDER_READY", label: "Render" },
];

function stagesFor(kind: JobStatusResponse["kind"]): StageDef[] {
  return kind === "ingest" ? INGEST_STAGES : MANUAL_STAGES;
}

function touchedKeys(events: JobEvent[]): Set<string> {
  return new Set(events.filter((e) => e.type === "milestone").map((e) => e.stage));
}

type LampState = "idle" | "active" | "done" | "error";

function nodeState(
  stage: StageDef,
  index: number,
  stages: StageDef[],
  touched: Set<string>,
  status: JobStatusResponse["status"],
): LampState {
  if (stage.key === "PROCESSING") {
    if (touched.has("CLIP_RENDER_READY")) return "done";
    if (status === "running") return "active";
    if (status === "failed") return "error";
    return "idle";
  }
  if (touched.has(stage.key)) return "done";
  const priorReady = stages.slice(0, index).every((s) => s.key === "PROCESSING" || touched.has(s.key));
  if (!priorReady) return "idle";
  if (status === "running") return "active";
  if (status === "failed") return "error";
  return "idle";
}

function formatTime(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

export function SignalChain({ job }: { job: JobStatusResponse | null }) {
  if (!job) {
    return (
      <div className="module">
        <div className="module__eyebrow">Signal chain</div>
        <p className="logfeed__empty">Configure a source below and hit RUN — live pipeline progress lands here.</p>
      </div>
    );
  }

  const stages = stagesFor(job.kind);
  const touched = touchedKeys(job.events);
  const visibleEvents = job.events.filter((e) => e.stage !== "queued");

  return (
    <div className="module">
      <div className="module__eyebrow">Signal chain — {job.job_id}</div>

      <div className="chain">
        {stages.map((stage, i) => {
          const state = nodeState(stage, i, stages, touched, job.status);
          const nextState = i < stages.length - 1 ? nodeState(stages[i + 1], i + 1, stages, touched, job.status) : "idle";
          return (
            <div className="chain__node" data-touched={state !== "idle"} key={stage.key}>
              <div className="chain__lamp-track">
                {i > 0 && <div className="chain__connector" data-lit={state !== "idle"} />}
                <div className="chain__lamp" data-state={state} />
                {i < stages.length - 1 && <div className="chain__connector" data-lit={nextState !== "idle"} />}
              </div>
              <div className="chain__label">{stage.label}</div>
            </div>
          );
        })}
      </div>

      <div className="logfeed">
        {visibleEvents.length === 0 ? (
          <div className="logfeed__empty">Waiting for the first event…</div>
        ) : (
          visibleEvents.map((event, i) => (
            <div className="logfeed__line" data-type={event.type} key={i}>
              <span className="logfeed__ts">{formatTime(event.ts)}</span>
              <span className="logfeed__msg">{event.message}</span>
            </div>
          ))
        )}
      </div>

      <div className="chain-footer">
        <span className="chain-status" data-status={job.status}>
          {job.status}
        </span>
        {job.status === "succeeded" && job.output_paths.length > 0 && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {job.output_paths.map((_, i) => (
              <a key={i} className="download-btn" href={downloadUrl(job.job_id, i)}>
                Download{job.output_paths.length > 1 ? ` #${i + 1}` : ""}
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
