// JobHistory — the "reel log": a session-only (no persistence, per the
// v1 spec) list of jobs created from this tab. Clicking a row selects it
// for the SignalChain panel above.
import type { JobStatusResponse } from "../api";

interface JobHistoryProps {
  jobs: JobStatusResponse[];
  selectedId: string | null;
  onSelect: (jobId: string) => void;
}

function shortPath(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) return "—";
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}

function formatTime(ts: string): string {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

export function JobHistory({ jobs, selectedId, onSelect }: JobHistoryProps) {
  return (
    <div className="module">
      <div className="module__eyebrow">Reel log — this session</div>
      {jobs.length === 0 ? (
        <p className="reel__empty">No jobs run yet.</p>
      ) : (
        <table className="reel">
          <thead>
            <tr>
              <th>Status</th>
              <th>Source</th>
              <th>Streamer</th>
              <th>Mode</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {[...jobs].reverse().map((job) => (
              <tr key={job.job_id} data-selected={job.job_id === selectedId} onClick={() => onSelect(job.job_id)}>
                <td>
                  <span className="reel__tally" data-status={job.status} />
                  {job.status}
                </td>
                <td>{shortPath(job.params.video_path)}</td>
                <td>{typeof job.params.streamer_id === "string" ? job.params.streamer_id : "—"}</td>
                <td>{job.kind === "ingest" ? "auto" : "manual"}</td>
                <td>{formatTime(job.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
