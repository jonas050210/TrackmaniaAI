/** Runs list page. */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type RunRow } from "../api";
import { Badge, Empty, ErrorBox, Loading, formatNumber } from "../components/ui";

export function Runs() {
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await api.runs();
        if (!cancelled) {
          setRuns(data.runs);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const interval = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  const needle = filter.trim().toLowerCase();
  const visible = needle
    ? runs.filter((r) => (r.run_name || r.name).toLowerCase().includes(needle) || r.driver.includes(needle))
    : runs;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Runs</h1>
          <p className="subtitle">{runs.length} training runs · click one for metrics, episodes and logs.</p>
        </div>
        <input
          type="text"
          placeholder="Filter runs…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ maxWidth: 280 }}
        />
      </div>
      {runs.length === 0 ? (
        <Empty icon="🏁">
          No runs yet. <Link to="/training">Start a training run</Link>.
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
              {visible.map((run) => (
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
    </div>
  );
}
