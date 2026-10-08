/** Run detail: metric charts, episodes, evaluations, checkpoints, log tail. */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type RunSnapshot, type Series } from "../api";
import { LineChart, toChartSeries } from "../components/Chart";
import { LogViewer } from "../components/LogViewer";
import { Badge, Empty, ErrorBox, Loading, formatBytes, formatNumber, formatPercent, timeAgo } from "../components/ui";

const REWARD_METRICS = ["env/episode_reward", "reward/episode_reward", "env/mean_reward_100"];
const PROGRESS_METRICS = ["env/progress_fraction", "env/episode_progress_fraction"];
const LEARNER_METRICS = ["sac/critic_loss", "sac/actor_loss", "learner/temperature"];
const SYSTEM_METRICS = ["system/memory_used_fraction", "system/load1"];

function pickSeries(history: { series: Series[] } | null, names: string[]): string[] {
  if (!history) return [];
  const available = new Set(history.series.map((s) => s.name));
  return names.filter((n) => available.has(n));
}

export function RunDetail() {
  const { name = "" } = useParams();
  const [snapshot, setSnapshot] = useState<RunSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await api.run(name);
      setSnapshot(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    setLoading(true);
    load();
    const interval = setInterval(load, 10_000);
    return () => clearInterval(interval);
  }, [load]);

  const rewardChart = useMemo(
    () => toChartSeries(snapshot?.history ?? null, pickSeries(snapshot?.history ?? null, REWARD_METRICS)),
    [snapshot]
  );
  const progressChart = useMemo(
    () => toChartSeries(snapshot?.history ?? null, pickSeries(snapshot?.history ?? null, PROGRESS_METRICS)),
    [snapshot]
  );
  const learnerChart = useMemo(
    () =>
      toChartSeries(
        snapshot?.history ?? null,
        pickSeries(snapshot?.history ?? null, LEARNER_METRICS),
        ["#f87171", "#a78bfa", "#38bdf8"]
      ),
    [snapshot]
  );
  const systemChart = useMemo(
    () =>
      toChartSeries(
        snapshot?.history ?? null,
        pickSeries(snapshot?.history ?? null, SYSTEM_METRICS),
        ["#34d399", "#fbbf24"]
      ),
    [snapshot]
  );

  if (loading && !snapshot) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;
  if (!snapshot) return <Empty>No data</Empty>;

  const { status, history, episodes, evaluations, checkpoints, manifest, log_tail } = snapshot;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>{status.run_name}</h1>
          <p className="subtitle">
            {status.driver} · step {status.step} / {status.total_steps} ·{" "}
            {status.ended ? "ended" : "running"} · last update {timeAgo(manifest.created_utc as string)}
          </p>
        </div>
        <div className="btn-row">
          <Link className="btn" to="/replays">Replays</Link>
          <Link className="btn primary" to="/evaluate">Evaluate</Link>
        </div>
      </div>

      <div className="grid cols-4">
        <div className="stat">
          <div className="label">Progress</div>
          <div className="value">{formatPercent(status.progress_fraction)}</div>
          <div className="hint">of the training budget</div>
        </div>
        <div className="stat">
          <div className="label">Episodes</div>
          <div className="value">{episodes.length}</div>
          <div className="hint">recorded in this run</div>
        </div>
        <div className="stat">
          <div className="label">Evaluations</div>
          <div className="value">{evaluations.length}</div>
          <div className="hint">mid-run + held-out</div>
        </div>
        <div className="stat">
          <div className="label">Checkpoints</div>
          <div className="value">{checkpoints.length}</div>
          <div className="hint">{status.checkpoints} on disk</div>
        </div>
      </div>

      <div className="section" style={{ marginTop: 24 }}>
        <h3>Reward</h3>
        {rewardChart.length ? (
          <LineChart series={rewardChart} title="Episode reward" yLabel="reward" />
        ) : (
          <Empty>No reward metrics recorded</Empty>
        )}
      </div>

      <div className="section">
        <h3>Progress</h3>
        {progressChart.length ? (
          <LineChart series={progressChart} title="Progress fraction" yLabel="fraction" />
        ) : (
          <Empty>No progress metrics recorded</Empty>
        )}
      </div>

      <div className="grid cols-2">
        <div className="section">
          <h3>Learner</h3>
          {learnerChart.length ? (
            <LineChart series={learnerChart} title="SAC losses & temperature" />
          ) : (
            <Empty>No learner metrics</Empty>
          )}
        </div>
        <div className="section">
          <h3>System</h3>
          {systemChart.length ? (
            <LineChart series={systemChart} title="Resources during training" yLabel="value" />
          ) : (
            <Empty>No system metrics</Empty>
          )}
        </div>
      </div>

      <div className="section">
        <h3>Episodes</h3>
        {episodes.length === 0 ? (
          <Empty>No episodes yet</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Step</th>
                  <th>End reason</th>
                  <th>Reward</th>
                  <th>Progress</th>
                </tr>
              </thead>
              <tbody>
                {episodes.map((ep) => (
                  <tr key={ep.episode}>
                    <td>{ep.episode}</td>
                    <td>{ep.step}</td>
                    <td>
                      {ep.end_reason === "finish" ? (
                        <Badge tone="green">finish</Badge>
                      ) : ep.end_reason === "crash" || ep.end_reason === "out_of_bounds" ? (
                        <Badge tone="red">{ep.end_reason}</Badge>
                      ) : (
                        <Badge tone="neutral">{ep.end_reason}</Badge>
                      )}
                    </td>
                    <td>{formatNumber(ep.reward)}</td>
                    <td>{formatNumber(ep.progress, 1)} m</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="section">
        <h3>Evaluations</h3>
        {evaluations.length === 0 ? (
          <Empty>No evaluations recorded</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Step</th>
                  <th>Finish rate</th>
                  <th>Mean progress</th>
                  <th>Crash rate</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {evaluations.map((ev, i) => (
                  <tr key={i}>
                    <td>{String(ev.label ?? "—")}</td>
                    <td>{String(ev.step ?? "—")}</td>
                    <td>{formatPercent(ev.finish_rate as number)}</td>
                    <td>{formatPercent(ev.mean_progress_fraction as number)}</td>
                    <td>{formatPercent(ev.crash_rate as number)}</td>
                    <td>{formatNumber(ev.score as number, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="section">
        <h3>Checkpoints</h3>
        {checkpoints.length === 0 ? (
          <Empty>No checkpoints yet</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((ckpt) => (
                  <tr key={ckpt.name}>
                    <td className="mono">{ckpt.name}</td>
                    <td className="dim">{formatBytes(ckpt.size)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="section">
        <h3>Log tail</h3>
        <LogViewer lines={log_tail} />
      </div>

      <p className="faint" style={{ fontSize: 12 }}>
        {history.total_records} metric records · auto-refresh every 10s
      </p>
    </div>
  );
}
