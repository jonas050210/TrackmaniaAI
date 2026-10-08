/** Overview: headline stats, live system resources, recent runs, quick actions. */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type RunRow, type SystemInfo } from "../api";
import { Badge, Empty, ErrorBox, Loading, formatBytes, formatNumber, timeAgo } from "../components/ui";

export function Overview() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [models, setModels] = useState<number>(0);
  const [tracks, setTracks] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [sys, runsData, modelsData, tracksData] = await Promise.all([
          api.system(),
          api.runs(),
          api.models(),
          api.tracks(),
        ]);
        if (cancelled) return;
        setSystem(sys);
        setRuns(runsData.runs);
        setModels(modelsData.models.length);
        setTracks(tracksData.report?.num_tracks ?? 0);
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const interval = setInterval(load, 15_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  const activeRuns = runs.filter((r) => !r.ended);
  const finishedRuns = runs.filter((r) => r.ended);
  const best = [...runs].sort((a, b) => b.progress_fraction - a.progress_fraction)[0];
  const resources = system?.resources;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Overview</h1>
          <p className="subtitle">Training, evaluation and system status at a glance.</p>
        </div>
        <div className="btn-row">
          <Link className="btn primary" to="/training">▶ Start training</Link>
          <Link className="btn" to="/evaluate">◎ Evaluate</Link>
        </div>
      </div>

      <div className="grid cols-4">
        <div className="stat">
          <div className="label">Runs</div>
          <div className="value">{runs.length}</div>
          <div className="hint">{activeRuns.length} still running</div>
        </div>
        <div className="stat">
          <div className="label">Models registered</div>
          <div className="value">{models}</div>
          <div className="hint">in the model registry</div>
        </div>
        <div className="stat">
          <div className="label">Tracks</div>
          <div className="value">{tracks}</div>
          <div className="hint">in the library</div>
        </div>
        <div className="stat">
          <div className="label">Best progress</div>
          <div className="value">{formatNumber((best?.progress_fraction ?? 0) * 100, 1)}%</div>
          <div className="hint">{best?.run_name ?? "no runs yet"}</div>
        </div>
      </div>

      <div className="section" style={{ marginTop: 24 }}>
        <h3>System</h3>
        <div className="grid cols-4">
          <div className="stat">
            <div className="label">CPU load (1m)</div>
            <div className="value">
              {resources?.load_average?.[0] !== undefined && resources.load_average[0] !== null
                ? formatNumber(resources.load_average[0], 2)
                : "—"}
            </div>
            <div className="hint">{system?.resources?.cpu_count ?? "?"} cores</div>
          </div>
          <div className="stat">
            <div className="label">Memory used</div>
            <div className="value">
              {resources?.memory_used_fraction !== null && resources?.memory_used_fraction !== undefined
                ? formatNumber(resources.memory_used_fraction * 100, 0) + "%"
                : "—"}
            </div>
            <div className="hint">
              {formatBytes(resources?.memory_available_bytes ?? null)} available
            </div>
          </div>
          <div className="stat">
            <div className="label">Disk free</div>
            <div className="value">{formatBytes(resources?.disk_free_bytes ?? null)}</div>
            <div className="hint">of {formatBytes(resources?.disk_total_bytes ?? null)}</div>
          </div>
          <div className="stat">
            <div className="label">Real game</div>
            <div className="value" style={{ fontSize: 18, paddingTop: 6 }}>
              {system?.game_integration_possible ? (
                <Badge tone="green">TMInterface ready</Badge>
              ) : (
                <Badge tone="amber">not on a game host</Badge>
              )}
            </div>
            <div className="hint">
              {system?.windows_host ? "Windows host" : `host: ${system?.platform ?? "?"}`}
            </div>
          </div>
        </div>
      </div>

      <div className="section">
        <h3>Recent runs</h3>
        {runs.length === 0 ? (
          <Empty icon="🏁">
            No training runs yet. <Link to="/training">Start one</Link> — a smoke run takes seconds.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Step</th>
                  <th>Progress</th>
                  <th>Driver</th>
                  <th>Status</th>
                  <th>Checkpoints</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 12).map((run) => (
                  <tr key={run.run_dir}>
                    <td>
                      <Link to={`/runs/${encodeURIComponent(run.name)}`}>{run.run_name || run.name}</Link>
                    </td>
                    <td>
                      {run.step} / {run.total_steps}
                    </td>
                    <td>{formatNumber(run.progress_fraction * 100, 1)}%</td>
                    <td>
                      {run.simulated ? (
                        <Badge tone="amber" title="Toy model, not the real game">simulated</Badge>
                      ) : (
                        <Badge tone="green">{run.driver}</Badge>
                      )}
                    </td>
                    <td>
                      {run.ended ? <Badge tone="neutral">ended</Badge> : <Badge tone="blue">running</Badge>}
                    </td>
                    <td className="dim">{run.checkpoints}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {finishedRuns.length > 0 && (
          <p className="faint" style={{ marginTop: 8, fontSize: 12 }}>
            Showing {Math.min(12, runs.length)} of {runs.length} runs · {finishedRuns.length} finished
          </p>
        )}
      </div>

      <div className="section">
        <h3>Quick start</h3>
        <div className="card">
          <ul className="tight">
            <li>
              <Link to="/training">Train</Link> a policy (start from <code className="mono">tmai/configs/smoke.yaml</code> for a
              seconds-long smoke run, or <code className="mono">default.yaml</code> for the real game).
            </li>
            <li>
              <Link to="/evaluate">Evaluate</Link> a checkpoint, or <Link to="/evaluate">benchmark</Link> several
              models across splits.
            </li>
            <li>
              <Link to="/models">Register</Link> the resulting checkpoint as a named model.
            </li>
            <li>
              <Link to="/tracks">Inspect tracks</Link> in 3D, and <Link to="/replays">compare replays</Link> against a
              human ghost.
            </li>
            <li>
              <Link to="/diagnostics">Run the doctor</Link> to verify the environment and the real-game integration.
            </li>
          </ul>
        </div>
      </div>

      <p className="faint" style={{ fontSize: 12 }}>
        Last refreshed {timeAgo(new Date().toISOString())} · auto-refresh every 15s
      </p>
    </div>
  );
}
