/** Evaluate & benchmark page: run evaluations, benchmark models, view reports. */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, waitForJob, type BenchmarkRow, type Job, type RunRow } from "../api";
import { LogViewer } from "../components/LogViewer";
import { Badge, Empty, ErrorBox, JobBadge, Loading, Spinner, formatPercent, useToast } from "../components/ui";

type BootstrapEstimate = {
  ci95: [number, number] | null;
  resampling_unit: "families" | "tracks" | "episodes" | "episodes_within_family" | null;
  sample_count: number;
};

type BenchmarkSplitSummary = {
  finish_rate: number;
  mean_progress_fraction: number;
  crash_rate: number;
  num_tracks: number;
  num_episodes: number;
};

type BenchmarkModelResult = {
  label: string;
  score: number;
  splits: Record<string, BenchmarkSplitSummary>;
  confidence_intervals: Record<string, Record<string, BootstrapEstimate>>;
};

type BenchmarkPairResult = {
  split: string;
  model_a: string;
  model_b: string;
  episodes: number;
  wins_a: number;
  wins_b: number;
  ties: number;
  win_rate_a: number;
  mean_progress_delta: number;
  confidence_intervals: {
    win_rate_a?: BootstrapEstimate;
    mean_progress_delta?: BootstrapEstimate;
  };
};

type BenchmarkReportResult = {
  seed_repeats: number;
  evaluation_seeds: number[];
  models: BenchmarkModelResult[];
  head_to_head: BenchmarkPairResult[];
};

function formatCi(estimate: BootstrapEstimate | undefined): string {
  const interval = estimate?.ci95;
  if (!interval || interval.length !== 2) return "not enough independent samples";
  return `${formatPercent(interval[0])}–${formatPercent(interval[1])}`;
}

export function Evaluate() {
  const { toast } = useToast();
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [models, setModels] = useState<{ name: string; source_checkpoint: string; step: number }[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // eval form
  const [evalTarget, setEvalTarget] = useState("");
  const [evalSplits, setEvalSplits] = useState("validation");
  const [evalEpisodes, setEvalEpisodes] = useState(3);
  const [evalJob, setEvalJob] = useState<Job | null>(null);
  const [evalRunning, setEvalRunning] = useState(false);

  // benchmark form
  const [benchRows, setBenchRows] = useState<{ label: string; checkpoint: string }[]>([
    { label: "", checkpoint: "" },
    { label: "", checkpoint: "" },
  ]);
  const [benchSplits, setBenchSplits] = useState("validation,test");
  const [benchEpisodes, setBenchEpisodes] = useState(3);
  const [benchSeedRepeats, setBenchSeedRepeats] = useState(3);
  const [benchName, setBenchName] = useState("benchmark");
  const [benchJob, setBenchJob] = useState<Job | null>(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const benchmarkReport = (benchJob?.result?.report ?? null) as unknown as BenchmarkReportResult | null;

  async function load() {
    try {
      const [runsData, modelsData, benchmarksData] = await Promise.all([
        api.runs(),
        api.models(),
        api.benchmarks(),
      ]);
      setRuns(runsData.runs);
      setModels(modelsData.models);
      setBenchmarks(benchmarksData.benchmarks);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 15_000);
    return () => clearInterval(interval);
  }, []);

  // follow running jobs
  useEffect(() => {
    const job = evalJob ?? benchJob;
    if (!job || ["done", "failed", "cancelled", "interrupted"].includes(job.state)) return;
    const interval = setInterval(async () => {
      try {
        const fresh = await api.job(job.id);
        if (evalJob) setEvalJob(fresh);
        if (benchJob) setBenchJob(fresh);
      } catch {
        /* ignore */
      }
    }, 1_500);
    return () => clearInterval(interval);
  }, [evalJob, benchJob]);

  async function startEval() {
    setEvalRunning(true);
    try {
      const payload: Record<string, unknown> = {
        splits: evalSplits.split(",").map((s) => s.trim()).filter(Boolean),
        episodes: evalEpisodes,
      };
      if (evalTarget.includes("/")) payload.checkpoint = evalTarget;
      else payload.run = evalTarget;
      const { job } = await api.evaluate(payload as never);
      setEvalJob(job);
      toast(`Evaluation job started: ${job.id}`, "success");
      waitForJob(job.id, (j) => setEvalJob(j)).then((finished) => {
        setEvalRunning(false);
        load();
        toast(
          finished.state === "done" ? "Evaluation finished" : `Evaluation ${finished.state}`,
          finished.state === "done" ? "success" : "error"
        );
      });
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
      setEvalRunning(false);
    }
  }

  async function startBenchmark() {
    setBenchRunning(true);
    try {
      const payload = {
        models: benchRows.filter((r) => r.label && r.checkpoint),
        splits: benchSplits.split(",").map((s) => s.trim()).filter(Boolean),
        episodes: benchEpisodes,
        seed_repeats: benchSeedRepeats,
        name: benchName || "benchmark",
      };
      const { job } = await api.benchmark(payload);
      setBenchJob(job);
      toast(`Benchmark job started: ${job.id}`, "success");
      waitForJob(job.id, (j) => setBenchJob(j)).then((finished) => {
        setBenchRunning(false);
        load();
        toast(
          finished.state === "done" ? "Benchmark finished" : `Benchmark ${finished.state}`,
          finished.state === "done" ? "success" : "error"
        );
      });
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
      setBenchRunning(false);
    }
  }

  if (loading) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Evaluate & Benchmark</h1>
          <p className="subtitle">
            Evaluate a checkpoint on held-out splits, or benchmark several models under one
            identical protocol.
          </p>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card-title">
            <h2>Evaluation</h2>
            {evalJob && <JobBadge state={evalJob.state} />}
          </div>
          <div className="field">
            <label>Checkpoint or run</label>
            <select value={evalTarget} onChange={(e) => setEvalTarget(e.target.value)}>
              <option value="">— select a run —</option>
              {runs.map((run) => (
                <option key={run.run_dir} value={run.name}>
                  {run.run_name || run.name} (step {run.step})
                </option>
              ))}
            </select>
            <span className="hint">Runs resolve to their latest checkpoint; a path to a .pt file works too.</span>
          </div>
          <div className="grid cols-2" style={{ gap: 12 }}>
            <div className="field">
              <label>Splits (comma separated)</label>
              <input value={evalSplits} onChange={(e) => setEvalSplits(e.target.value)} />
            </div>
            <div className="field">
              <label>Episodes per track</label>
              <input
                type="number"
                min={1}
                value={evalEpisodes}
                onChange={(e) => setEvalEpisodes(Number(e.target.value))}
              />
            </div>
          </div>
          <button className="btn primary" onClick={startEval} disabled={evalRunning || !evalTarget}>
            {evalRunning ? <Spinner size={12} /> : "◎"} Run evaluation
          </button>
          {evalJob?.result && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ marginBottom: 8 }}>Result</h3>
              <pre className="log" style={{ maxHeight: 260 }}>
                {JSON.stringify(evalJob.result.reports ?? evalJob.result, null, 2)}
              </pre>
            </div>
          )}
          {evalJob && <div style={{ marginTop: 12 }}><LogViewer lines={evalJob.log_tail} maxHeight={200} /></div>}
        </div>

        <div className="card">
          <div className="card-title">
            <h2>Benchmark</h2>
            {benchJob && <JobBadge state={benchJob.state} />}
          </div>
          <div className="field">
            <label>Models & baselines (label → checkpoint, run, or heuristic)</label>
            {benchRows.map((row, i) => (
              <div key={i} className="grid cols-2" style={{ gap: 8, marginBottom: 8 }}>
                <input
                  placeholder="label"
                  value={row.label}
                  onChange={(e) =>
                    setBenchRows((rows) => rows.map((r, j) => (j === i ? { ...r, label: e.target.value } : r)))
                  }
                />
                <select
                  value={row.checkpoint}
                  onChange={(e) =>
                    setBenchRows((rows) => rows.map((r, j) => (j === i ? { ...r, checkpoint: e.target.value } : r)))
                  }
                >
                  <option value="">— pick a model or run —</option>
                  <option value="baseline:curvature">
                    CurvaturePilot heuristic (not real-game validated)
                  </option>
                  {models.map((m) => (
                    <option key={m.name} value={m.source_checkpoint || m.name}>
                      {m.name} (step {m.step})
                    </option>
                  ))}
                  {runs.map((run) => (
                    <option key={run.run_dir} value={run.name}>
                      run: {run.run_name || run.name}
                    </option>
                  ))}
                </select>
              </div>
            ))}
            <button
              className="btn sm"
              onClick={() => setBenchRows((rows) => [...rows, { label: "", checkpoint: "" }])}
            >
              + add model
            </button>
            <p className="hint" style={{ marginTop: 8 }}>
              CurvaturePilot is a geometry-based heuristic for comparison, not a trained checkpoint or a live-game-validated controller.
            </p>
          </div>
          <div className="grid cols-4" style={{ gap: 12 }}>
            <div className="field">
              <label>Splits</label>
              <input value={benchSplits} onChange={(e) => setBenchSplits(e.target.value)} />
            </div>
            <div className="field">
              <label>Episodes / track / seed</label>
              <input
                type="number"
                min={1}
                step={1}
                value={benchEpisodes}
                onChange={(e) => setBenchEpisodes(Math.max(1, Number(e.target.value) || 1))}
              />
            </div>
            <div className="field">
              <label>Seed repeats</label>
              <input
                type="number"
                min={1}
                max={100}
                step={1}
                value={benchSeedRepeats}
                onChange={(e) => setBenchSeedRepeats(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
              />
            </div>
            <div className="field">
              <label>Name</label>
              <input value={benchName} onChange={(e) => setBenchName(e.target.value)} />
            </div>
          </div>
          <p className="hint" style={{ marginTop: -4, marginBottom: 12 }}>
            Repeats keep models paired on the same seeds. Intervals resample family clusters when labeled, otherwise tracks; seeds only vary starts when the configured driver supports random starts.
          </p>
          <button
            className="btn primary"
            onClick={startBenchmark}
            disabled={benchRunning || benchRows.filter((r) => r.label && r.checkpoint).length < 1}
          >
            {benchRunning ? <Spinner size={12} /> : "⚖"} Run benchmark
          </button>
          {benchJob?.result && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ marginBottom: 8 }}>Ranking</h3>
              <p>
                {(benchJob.result.ranking as string[] | undefined)?.map((label, i) => (
                  <span key={label} style={{ marginRight: 12 }}>
                    <Badge tone={i === 0 ? "green" : "neutral"}>
                      {i + 1}. {label}
                    </Badge>
                  </span>
                ))}
              </p>
              <pre className="log" style={{ maxHeight: 220 }}>
                {(benchJob.result.table as string) ?? JSON.stringify(benchJob.result, null, 2)}
              </pre>
              {benchmarkReport && (
                <div style={{ marginTop: 16 }}>
                  <h3 style={{ marginBottom: 6 }}>Approximate 95% bootstrap intervals</h3>
                  <p className="hint" style={{ marginBottom: 10 }}>
                    {benchmarkReport.seed_repeats} paired seed repeats: {benchmarkReport.evaluation_seeds?.join(", ")}. Explicit family clusters are resampled together, otherwise tracks; a single cluster falls back to episodes.
                  </p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Split</th>
                          <th>Finish rate</th>
                          <th>Mean progress</th>
                          <th>Crash rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchmarkReport.models.flatMap((model) =>
                          Object.entries(model.splits).map(([split, metrics]) => {
                            const intervals = model.confidence_intervals?.[split];
                            return (
                              <tr key={`${model.label}-${split}`}>
                                <td>{model.label}</td>
                                <td className="dim">{split} · {metrics.num_tracks} tracks / {metrics.num_episodes} episodes</td>
                                <td>{formatPercent(metrics.finish_rate)}<div className="dim">{formatCi(intervals?.finish_rate)}</div></td>
                                <td>{formatPercent(metrics.mean_progress_fraction)}<div className="dim">{formatCi(intervals?.mean_progress_fraction)}</div></td>
                                <td>{formatPercent(metrics.crash_rate)}<div className="dim">{formatCi(intervals?.crash_rate)}</div></td>
                              </tr>
                            );
                          })
                        )}
                      </tbody>
                    </table>
                  </div>
                  {benchmarkReport.head_to_head.length > 0 && (
                    <>
                      <h3 style={{ margin: "16px 0 8px" }}>Paired comparison intervals</h3>
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Split</th><th>Pair</th><th>W–L–T</th><th>A win share · 95% CI</th><th>Progress delta · 95% CI</th></tr>
                          </thead>
                          <tbody>
                            {benchmarkReport.head_to_head.map((pair) => (
                              <tr key={`${pair.split}-${pair.model_a}-${pair.model_b}`}>
                                <td>{pair.split}</td>
                                <td>{pair.model_a} vs {pair.model_b}</td>
                                <td>{pair.wins_a}–{pair.wins_b}–{pair.ties} · {pair.episodes} pairs</td>
                                <td>{formatPercent(pair.win_rate_a)}<div className="dim">{formatCi(pair.confidence_intervals.win_rate_a)}</div></td>
                                <td>{formatPercent(pair.mean_progress_delta)}<div className="dim">{formatCi(pair.confidence_intervals.mean_progress_delta)}</div></td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
          {benchJob && <div style={{ marginTop: 12 }}><LogViewer lines={benchJob.log_tail} maxHeight={200} /></div>}
        </div>
      </div>

      <div className="section">
        <h3>Benchmark reports</h3>
        {benchmarks.length === 0 ? (
          <Empty icon="⚖">No benchmark reports yet.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Created</th>
                  <th>Splits</th>
                  <th>Seeds</th>
                  <th>Ranking</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((b) => (
                  <tr key={b.path}>
                    <td className="mono">{b.benchmark}</td>
                    <td className="dim">{new Date(b.created_utc).toLocaleString()}</td>
                    <td className="dim">{b.splits.join(", ")}</td>
                    <td className="dim">{b.seed_repeats}</td>
                    <td>
                      {b.ranking.map((label, i) => (
                        <span key={label} style={{ marginRight: 8 }}>
                          <Badge tone={i === 0 ? "green" : "neutral"}>{label}</Badge>
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="section">
        <h3>Tips</h3>
        <div className="card">
          <ul className="tight">
            <li>Evaluations are deterministic by default (policy mean); stochastic mode samples actions.</li>
            <li>Held-out splits measure generalisation; a big train/validation gap means the policy is memorising.</li>
            <li>Register interesting checkpoints in <Link to="/models">Models</Link>, then benchmark them here.</li>
            <li>Head-to-head results pair the same track and episode index for each model.</li>
            <li>CurvaturePilot is a transparent geometric heuristic; it is not validated on a live Trackmania session.</li>
            <li>All numbers here come from the same evaluation code as <code className="mono">tmai eval</code>.</li>
          </ul>
        </div>
      </div>

      <p className="faint" style={{ fontSize: 12 }}>
        {formatPercent(0)} · reports are also written to the benchmarks directory on disk
      </p>
    </div>
  );
}
