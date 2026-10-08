/** Tracks page: library table + interactive 3D viewer with curvature colouring. */

import { useEffect, useState } from "react";
import { api, type TrackGeometry, type TrackLibraryReport } from "../api";
import { TrackViewer3D } from "../components/TrackViewer3D";
import { Badge, Empty, ErrorBox, Loading, formatNumber, formatPercent, useToast } from "../components/ui";

export function Tracks() {
  const { toast } = useToast();
  const [report, setReport] = useState<TrackLibraryReport | null>(null);
  const [synthetic, setSynthetic] = useState<string[]>([]);
  const [directory, setDirectory] = useState("");
  const [geometry, setGeometry] = useState<TrackGeometry | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCorridor, setShowCorridor] = useState(true);
  const [showCurvature, setShowCurvature] = useState(true);

  async function load() {
    try {
      const data = await api.tracks();
      setReport(data.report);
      setSynthetic(data.synthetic);
      setDirectory(data.directory);
      setError(data.report ? null : data.error ?? "no tracks");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function view(name: string, isSynthetic = false) {
    try {
      const geo = await api.trackGeometry(
        isSynthetic ? { synthetic: name } : { name, directory }
      );
      setGeometry(geo);
      setSelected(name);
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
    }
  }

  if (loading) return <Loading />;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Tracks</h1>
          <p className="subtitle">
            The track library with its split assignment, and an interactive 3D view of the
            geometry the agent actually drives on.
          </p>
        </div>
        <div className="btn-row">
          <label className="dim" style={{ fontSize: 12, display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={showCorridor} onChange={(e) => setShowCorridor(e.target.checked)} />
            corridor
          </label>
          <label className="dim" style={{ fontSize: 12, display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={showCurvature} onChange={(e) => setShowCurvature(e.target.checked)} />
            curvature colours
          </label>
        </div>
      </div>

      <div className="section">
        <h3>3D view {selected ? `— ${selected}` : ""}</h3>
        {error && !geometry ? (
          <ErrorBox>{error}</ErrorBox>
        ) : (
          <TrackViewer3D
            geometry={geometry}
            showCorridor={showCorridor}
            showCurvature={showCurvature}
            height={480}
          />
        )}
        <div className="toolbar" style={{ marginTop: 12 }}>
          {report?.tracks.map((track) => (
            <button
              key={track.name}
              className={`btn sm${selected === track.name ? " primary" : ""}`}
              onClick={() => view(track.name)}
            >
              {track.name}
            </button>
          ))}
          <span className="sep" />
          {synthetic.map((name) => (
            <button
              key={name}
              className={`btn sm${selected === name ? " primary" : ""}`}
              onClick={() => view(name, true)}
              title="built-in synthetic track"
            >
              {name} <span className="faint">(synthetic)</span>
            </button>
          ))}
        </div>
      </div>

      {geometry && (
        <div className="section">
          <h3>Geometry</h3>
          <div className="grid cols-4">
            <div className="stat">
              <div className="label">Length</div>
              <div className="value">{formatNumber(geometry.length, 0)} m</div>
            </div>
            <div className="stat">
              <div className="label">Samples</div>
              <div className="value">{geometry.num_points}</div>
            </div>
            <div className="stat">
              <div className="label">Mean curvature</div>
              <div className="value">{formatNumber(geometry.stats.curvature_mean as number, 4)}</div>
              <div className="hint">1/m</div>
            </div>
            <div className="stat">
              <div className="label">Corners</div>
              <div className="value">{String(geometry.stats.corner_count ?? "—")}</div>
              <div className="hint">tightest {formatNumber(geometry.stats.min_corner_radius as number, 0)} m</div>
            </div>
          </div>
        </div>
      )}

      <div className="section">
        <h3>Library</h3>
        {!report || report.num_tracks === 0 ? (
          <Empty icon="⬡">
            No tracks in <code className="mono">{directory}</code>. Record one with{" "}
            <code className="mono">tmai record-track</code> on the game host, or use a synthetic suite.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Track</th>
                  <th>Split</th>
                  <th>Length</th>
                  <th>Corners</th>
                  <th>Mean curvature</th>
                  <th>Straight %</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {report.tracks.map((track) => (
                  <tr key={track.identity} onClick={() => view(track.name)}>
                    <td><strong>{track.name}</strong></td>
                    <td>
                      <Badge
                        tone={
                          track.split === "train"
                            ? "blue"
                            : track.split === "validation"
                              ? "violet"
                              : "amber"
                        }
                      >
                        {track.split}
                      </Badge>
                    </td>
                    <td>{formatNumber(track.length, 0)} m</td>
                    <td>{(track.stats?.corner_count as number | undefined) ?? "—"}</td>
                    <td>{formatNumber(track.stats?.curvature_mean as number, 4)}</td>
                    <td>{formatPercent(track.stats?.straight_fraction as number, 0)}</td>
                    <td className="dim mono">{track.source ?? "synthetic"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {report && (
          <p className="faint" style={{ marginTop: 8, fontSize: 12 }}>
            Split weights:{" "}
            {Object.entries(report.split_weights)
              .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
              .join(" · ")}
            {" "}· counts:{" "}
            {Object.entries(report.counts)
              .map(([k, v]) => `${k} ${v}`)
              .join(" · ")}
          </p>
        )}
      </div>
    </div>
  );
}
