/** Replays & ghosts: pick a run, inspect its replays in 3D, compare against a human ghost. */

import { useEffect, useMemo, useState } from "react";
import { api, type DemoRow, type EpisodeReplay, type ReplayAnalysis, type ReplayComparison, type ReplayRow, type RunRow, type TrackGeometry } from "../api";
import { LineChart, type ChartSeries } from "../components/Chart";
import { LazyTrackViewer3D, type Trajectory } from "../components/LazyTrackViewer3D";
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
  const [ghostSource, setGhostSource] = useState<"human" | "replay">("human");
  const [ghostRun, setGhostRun] = useState("");
  const [ghostReplays, setGhostReplays] = useState<ReplayRow[]>([]);
  const [ghost, setGhost] = useState("");
  const [comparison, setComparison] = useState<ReplayComparison | null>(null);
  const [analysis, setAnalysis] = useState<ReplayAnalysis | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
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

  useEffect(() => {
    if (!ghostRun && runs.length) setGhostRun(runs[0].name);
  }, [ghostRun, runs]);

  useEffect(() => {
    if (ghostSource !== "replay" || !ghostRun) {
      setGhostReplays([]);
      return;
    }
    let cancelled = false;
    api.replays(ghostRun).then((data) => {
      if (!cancelled) setGhostReplays(data.replays);
    }).catch(() => {
      if (!cancelled) setGhostReplays([]);
    });
    return () => {
      cancelled = true;
    };
  }, [ghostSource, ghostRun]);

  const availableGhostReplays = useMemo(
    () => ghostReplays.filter((candidate) => (
      Boolean(selected?.track)
      && candidate.track === selected?.track
      && !(ghostRun === run && candidate.name === selected?.name)
    )),
    [ghostReplays, ghostRun, run, selected?.name, selected?.track]
  );

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
          selectedTrack
            ? api.trackGeometry({ name: selectedTrack }).catch(() => null)
            : Promise.resolve(null),
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

  useEffect(() => {
    if (!run || !selected?.track) {
      setAnalysis(null);
      setAnalysisError(null);
      return;
    }
    let cancelled = false;
    setAnalysis(null);
    setAnalysisError(null);
    api
      .runAnalysis(run, selected.track, 20)
      .then((result) => {
        if (!cancelled) setAnalysis(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setAnalysisError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [run, selected?.track]);

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
        name: ghostSource === "human" ? "ghost time at station (s)" : "reference AI time at station (s)",
        points: comparison.stations.map((s, i) => ({ step: s, value: comparison.ghost_times[i] })),
        color: "#fbbf24",
      },
    ];
  }, [comparison, ghostSource]);

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
            Inspect recorded episodes in 3D, then compare an AI run against a human ghost or
            another AI replay on the same track to find where pace is gained or lost.
          </p>
        </div>
      </div>

      <div className="grid cols-2" style={{ gridTemplateColumns: "280px 1fr" }}>
        <div className="card" style={{ padding: 16 }}>
          <div className="field">
            <label>Run</label>
            <select
              value={run}
              onChange={(e) => {
                setRun(e.target.value);
                setSelected(null);
                setReplay(null);
                setGhost("");
                setComparison(null);
              }}
            >
              {runs.map((r) => (
                <option key={r.run_dir} value={r.name}>{r.run_name || r.name}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Replay (episode)</label>
            <select
              value={selected?.name ?? ""}
              onChange={(e) => {
                setSelected(replays.find((r) => r.name === e.target.value) ?? null);
                setGhost("");
                setComparison(null);
              }}
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
                {selected.end_reason === "invalid_finish" ? (
                  <Badge tone="red">invalid finish</Badge>
                ) : selected.finished ? (
                  <Badge tone="green">finished</Badge>
                ) : (
                  <Badge tone="neutral">{selected.end_reason}</Badge>
                )}
              </p>
              <p><span className="dim">race time</span> {formatNumber(selected.race_time, 3)} s</p>
              <p><span className="dim">reward</span> {formatNumber(selected.total_reward)}</p>
              <p><span className="dim">progress</span> {formatPercent(selected.progress_fraction)}</p>
              <p><span className="dim">samples</span> {selected.num_samples}</p>
            </div>
          )}
          <hr className="divider" />
          <div className="field">
            <label>Comparison reference</label>
            <select
              value={ghostSource}
              onChange={(e) => {
                setGhostSource(e.target.value as "human" | "replay");
                setGhost("");
                setComparison(null);
              }}
            >
              <option value="human">Human demonstration</option>
              <option value="replay">Another AI replay</option>
            </select>
          </div>
          {ghostSource === "human" ? (
            <div className="field">
              <label>Ghost (human demonstration)</label>
              <select
                value={ghost}
                onChange={(e) => {
                  setGhost(e.target.value);
                  setComparison(null);
                }}
              >
                <option value="">— none —</option>
                {demos.map((demo) => (
                  <option key={demo.path} value={demo.path}>
                    {demo.name} ({demo.steps} steps)
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <>
              <div className="field">
                <label>Reference run</label>
                <select
                  value={ghostRun}
                  onChange={(e) => {
                    setGhostRun(e.target.value);
                    setGhost("");
                    setComparison(null);
                  }}
                >
                  {runs.map((candidate) => (
                    <option key={candidate.run_dir} value={candidate.name}>
                      {candidate.run_name || candidate.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>AI replay · {selected?.track || "same track required"}</label>
                <select
                  value={ghost}
                  onChange={(e) => {
                    setGhost(e.target.value);
                    setComparison(null);
                  }}
                  disabled={!selected?.track || availableGhostReplays.length === 0}
                >
                  <option value="">
                    {availableGhostReplays.length ? "— select a reference replay —" : "no matching replay on this track"}
                  </option>
                  {availableGhostReplays.map((candidate) => (
                    <option key={candidate.path} value={candidate.path}>
                      episode {candidate.episode} · {candidate.end_reason || "running"} · {formatPercent(candidate.progress_fraction)}
                    </option>
                  ))}
                </select>
                <span className="hint">
                  Only replays from the same track are offered, keeping timing comparisons meaningful.
                </span>
              </div>
            </>
          )}
          <button
            className="btn primary"
            onClick={compare}
            disabled={!selected || !ghost}
          >
            ⚖ {ghostSource === "human" ? "Compare vs ghost" : "Compare AI replays"}
          </button>
        </div>

        <div>
          {replay ? (
            <LazyTrackViewer3D
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

      {selected && (
        <div className="section">
          <div className="page-head" style={{ marginBottom: 12 }}>
            <div>
              <h3>Sector analysis & failure heatmap</h3>
              <p className="subtitle">
                Summarizes all matching replays for {selected.track || "this track"}. Pace and
                lateral position are descriptive diagnostics, not a proof of optimality.
              </p>
            </div>
            {analysis && <Badge tone="blue">{analysis.num_replays} replays · {analysis.num_samples} samples</Badge>}
          </div>
          {analysisError ? (
            <ErrorBox>{analysisError}</ErrorBox>
          ) : !analysis ? (
            <Loading label="Analyzing replay trajectories…" />
          ) : (
            <>
              <div className="grid cols-4">
                <div className="stat">
                  <div className="label">Located failures</div>
                  <div className="value">{analysis.failure_heatmap.events_with_location}</div>
                  <div className="hint">of {Object.values(analysis.failure_reasons).reduce((sum, count) => sum + count, 0)} recorded failures</div>
                </div>
                <div className="stat">
                  <div className="label">Failure reasons</div>
                  <div className="value">{Object.keys(analysis.failure_reasons).length}</div>
                  <div className="hint">distinct end reasons</div>
                </div>
                <div className="stat">
                  <div className="label">Slowest sectors</div>
                  <div className="value">{analysis.slowest_sectors.map((index) => index + 1).join(", ") || "—"}</div>
                  <div className="hint">ranked by average sector time</div>
                </div>
                <div className="stat">
                  <div className="label">Sector coverage</div>
                  <div className="value">{analysis.sectors.filter((sector) => sector.speed_samples > 0).length}/{analysis.sector_count}</div>
                  <div className="hint">sectors with speed samples</div>
                </div>
              </div>
              <div className="table-wrap" style={{ marginTop: 16 }}>
                <table>
                  <thead>
                    <tr>
                      <th>Sector</th>
                      <th>Mean speed</th>
                      <th>Mean |lateral|</th>
                      <th>Mean sector time</th>
                      <th>Failures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.sectors.map((sector) => (
                      <tr key={sector.index}>
                        <td>{sector.index + 1} · {formatNumber(sector.start_m, 0)}–{formatNumber(sector.end_m, 0)} m</td>
                        <td>{formatNumber(sector.mean_speed_mps, 2)} m/s</td>
                        <td>{formatNumber(sector.mean_abs_lateral_m, 2)} m</td>
                        <td>{formatNumber(sector.mean_sector_time_s, 3)} s <span className="faint">({sector.sector_time_samples} samples)</span></td>
                        <td>{sector.failure_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ marginTop: 20 }}>
                <h4>Failure locations · station × normalized lateral offset</h4>
                <p className="faint" style={{ fontSize: 12 }}>
                  Columns run left-to-right across the corridor and beyond it; only known failure end reasons are counted.
                </p>
                <div className="table-wrap">
                  <table aria-label="Spatial failure heatmap">
                    <thead>
                      <tr>
                        <th>Station</th>
                        {analysis.failure_heatmap.lateral_labels.map((label) => <th key={label}>{label.replace(/_/g, " ")}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.failure_heatmap.counts.map((row, sector) => {
                        const maximum = Math.max(1, ...row);
                        return (
                          <tr key={sector}>
                            <th>{formatNumber(analysis.failure_heatmap.station_edges_m[sector], 0)}–{formatNumber(analysis.failure_heatmap.station_edges_m[sector + 1], 0)} m</th>
                            {row.map((count, bin) => (
                              <td
                                key={bin}
                                title={`${count} failure(s) · ${analysis.failure_heatmap.lateral_labels[bin]}`}
                                style={{
                                  textAlign: "center",
                                  minWidth: 42,
                                  background: `rgba(248, 113, 113, ${count ? 0.22 + 0.68 * count / maximum : 0.035})`,
                                }}
                              >
                                {count || "·"}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {Object.keys(analysis.failure_reasons).length > 0 && (
                  <p className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                    Reasons: {Object.entries(analysis.failure_reasons).map(([reason, count]) => `${reason} ${count}`).join(" · ")}
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {comparison && (
        <div className="section">
          <h3>
            {ghostSource === "human" ? "Ghost comparison" : "AI replay comparison"} — {comparison.track}
          </h3>
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
                AI {formatNumber(comparison.ai_race_time, 2)} s vs reference {formatNumber(comparison.ghost_race_time, 2)} s
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
