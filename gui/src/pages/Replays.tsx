/** Replays & ghosts: pick a run, inspect its replays in 3D, compare against a human ghost. */

import { useEffect, useMemo, useState } from "react";
import { api, type DemoRow, type EpisodeReplay, type ReplayComparison, type ReplayRow, type RunRow, type TrackGeometry } from "../api";
import { LineChart, type ChartSeries } from "../components/Chart";
import { TrackViewer3D, type Trajectory } from "../components/TrackViewer3D";
import { Badge, Empty, ErrorBox, Loading, formatNumber, formatPercent, useToast } from "../components/ui";

export function Replays() {
  const { toast } = useToast();
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [run, setRun] = useState("");
  const [replays, setReplays] = useState<ReplayRow[]>([]);
  const [selected, setSelected] = useState<ReplayRow | null>(null);
  const [replay, setReplay] = useState<EpisodeReplay | null>(null);
  const [geometry, setGeometry] = useState<TrackGeometry | null>(null);
  const [demos, setDemos] = useState<DemoRow[]>([]);
  const [ghost, setGhost] = useState("");
  const [comparison, setComparison] = useState<ReplayComparison | null>(null);
  const [carIndex, setCarIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadRuns() {
    const data = await api.runs();
    setRuns(data.runs);
    if (!run && data.runs.length) setRun(data.runs[0].name);
  }

  async function loadReplays(runName: string) {
    try {
      const data = await api.replays(runName);
      setReplays(data.replays);
      if (data.replays.length && !selected) setSelected(data.replays[data.replays.length - 1]);
    } catch (err) {
      setReplays([]);
      toast(err instanceof Error ? err.message : String(err), "error");
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function initial() {
      try {
        await loadRuns();
        const demosData = await api.demos();
        if (cancelled) return;
        setDemos(demosData.demos);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    initial();
    const interval = setInterval(() => {
      loadRuns().catch(() => {});
    }, 15_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (run) loadReplays(run);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  // load the selected replay + its track geometry
  useEffect(() => {
    if (!run || !selected) return;
    const selectedName = selected.name;
    const selectedTrack = selected.track;
    let cancelled = false;
    async function load() {
      try {
        const [replayData, geometryData] = await Promise.all([
          api.replay(run, selectedName),
          api
            .trackGeometry(selectedTrack ? { name: selectedTrack } : { synthetic: "straight" })
            .catch(() => null),
        ]);
        if (cancelled) return;
        setReplay(replayData);
        setGeometry(geometryData);
        setCarIndex(0);
        setComparison(null);
      } catch (err) {
        if (!cancelled) toast(err instanceof Error ? err.message : String(err), "error");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, selected?.name]);

  const trajectory: Trajectory[] = useMemo(
    () =>
      replay
        ? [
            {
              positions: replay.positions,
              label: `episode ${replay.episode}`,
              color: replay.finished ? 0x34d399 : 0x38bdf8,
            },
          ]
        : [],
    [replay]
  );

  const speedChart: ChartSeries[] = useMemo(() => {
    if (!replay) return [];
    return [
      {
        name: "speed (m/s)",
        points: replay.speeds.map((v, i) => ({ step: i, value: v })),
        color: "#38bdf8",
      },
    ];
  }, [replay]);

  const rewardChart: ChartSeries[] = useMemo(() => {
    if (!replay) return [];
    return [
      {
        name: "reward",
        points: replay.rewards.map((v, i) => ({ step: i, value: v })),
        color: "#a78bfa",
      },
    ];
  }, [replay]);

  const gapChart: ChartSeries[] = useMemo(() => {
    if (!comparison) return [];
    return [
      {
        name: "AI time at station (s)",
        points: comparison.stations.map((s, i) => ({ step: s, value: comparison.ai_times[i] })),
        color: "#38bdf8",
      },
      {
        name: "ghost time at station (s)",
        points: comparison.stations.map((s, i) => ({ step: s, value: comparison.ghost_times[i] })),
        color: "#fbbf24",
      },
    ];
  }, [comparison]);

  async function compare() {
    if (!run || !selected || !ghost) return;
    try {
      const result = await api.replayCompare({ run, a: selected.name, b: ghost });
      setComparison(result);
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
    }
  }

  if (loading) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Replays & Ghosts</h1>
          <p className="subtitle">
            Every training episode can be replayed in 3D. Compare an AI replay against a human
            demonstration (a ghost) to see exactly where time is lost.
          </p>
        </div>
      </div>

      <div className="grid cols-2" style={{ gridTemplateColumns: "280px 1fr" }}>
        <div className="card" style={{ padding: 16 }}>
          <div className="field">
            <label>Run</label>
            <select value={run} onChange={(e) => { setRun(e.target.value); setSelected(null); setReplay(null); }}>
              {runs.map((r) => (
                <option key={r.run_dir} value={r.name}>{r.run_name || r.name}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Replay (episode)</label>
            <select
              value={selected?.name ?? ""}
              onChange={(e) => setSelected(replays.find((r) => r.name === e.target.value) ?? null)}
            >
              {replays.length === 0 && <option value="">no replays in this run</option>}
              {replays.map((r) => (
                <option key={r.name} value={r.name}>
                  ep {r.episode} · {r.track} · {r.end_reason || "running"}
                </option>
              ))}
            </select>
          </div>
          {selected && (
            <div style={{ fontSize: 13 }}>
              <p><span className="dim">episode</span> {selected.episode}</p>
              <p><span className="dim">track</span> {selected.track}</p>
              <p>
                <span className="dim">outcome</span>{" "}
                {selected.finished ? <Badge tone="green">finished</Badge> : <Badge tone="neutral">{selected.end_reason}</Badge>}
              </p>
              <p><span className="dim">race time</span> {formatNumber(selected.race_time, 3)} s</p>
              <p><span className="dim">reward</span> {formatNumber(selected.total_reward)}</p>
              <p><span className="dim">progress</span> {formatPercent(selected.progress_fraction)}</p>
              <p><span className="dim">samples</span> {selected.num_samples}</p>
            </div>
          )}
          <hr className="divider" />
          <div className="field">
            <label>Ghost (human demonstration)</label>
            <select value={ghost} onChange={(e) => setGhost(e.target.value)}>
              <option value="">— none —</option>
              {demos.map((d) => (
                <option key={d.path} value={d.path}>
                  {d.name} ({d.steps} steps)
                </option>
              ))}
            </select>
          </div>
          <button className="btn primary" onClick={compare} disabled={!selected || !ghost}>
            ⚖ Compare vs ghost
          </button>
        </div>

        <div>
          {replay ? (
            <TrackViewer3D
              geometry={geometry}
              trajectories={trajectory}
              carIndex={carIndex}
              onCarIndexChange={setCarIndex}
              height={440}
            />
          ) : (
            <Empty icon="◉">
              {replays.length === 0
                ? "This run has no replays. Enable train.record_replays in the config to record them."
                : "Select a replay."}
            </Empty>
          )}
          {replay && (
            <div className="btn-row dim" style={{ marginTop: 8, fontSize: 12 }}>
              <span className="dim">
                sample {carIndex} / {replay.num_samples} · speed{" "}
                {formatNumber(replay.speeds[Math.min(carIndex, replay.speeds.length - 1)], 1)} m/s
              </span>
            </div>
          )}
        </div>
      </div>

      {replay && (
        <div className="grid cols-2" style={{ marginTop: 24 }}>
          <div className="section">
            <h3>Speed profile</h3>
            <LineChart series={speedChart} title="Forward speed" yLabel="m/s" />
          </div>
          <div className="section">
            <h3>Reward profile</h3>
            <LineChart series={rewardChart} title="Per-step reward" yLabel="reward" />
          </div>
        </div>
      )}

      {comparison && (
        <div className="section">
          <h3>Ghost comparison — {comparison.track}</h3>
          <div className="grid cols-4">
            <div className="stat">
              <div className="label">Mean segment gap</div>
              <div className="value">{formatNumber(comparison.mean_gap, 3)} s</div>
              <div className="hint">positive = AI loses time</div>
            </div>
            <div className="stat">
              <div className="label">Worst segment</div>
              <div className="value">{formatNumber(comparison.max_gap, 3)} s</div>
            </div>
            <div className="stat">
              <div className="label">Best segment</div>
              <div className="value">{formatNumber(comparison.min_gap, 3)} s</div>
            </div>
            <div className="stat">
              <div className="label">Race time delta</div>
              <div className="value">
                {comparison.race_time_delta === null ? "—" : `${formatNumber(comparison.race_time_delta, 3)} s`}
              </div>
              <div className="hint">
                AI {formatNumber(comparison.ai_race_time, 2)} s vs ghost {formatNumber(comparison.ghost_race_time, 2)} s
              </div>
            </div>
          </div>
          <div style={{ marginTop: 16 }}>
            <LineChart series={gapChart} title="Race time at each station (arc length)" yLabel="seconds" />
          </div>
          <p className="dim" style={{ fontSize: 12, marginTop: 8 }}>
            Stations: {comparison.stations.length} · the comparison covers only the arc length both
            replays drove ({formatNumber(comparison.stations[0], 0)}–{formatNumber(comparison.stations[comparison.stations.length - 1], 0)} m).
            Segment gaps are invariant to where each replay started.
          </p>
        </div>
      )}
    </div>
  );
}
