/** Diagnostics page: live system metrics, doctor report, calibration. */

import { useEffect, useState } from "react";
import { api, waitForJob, type Job, type SystemInfo } from "../api";
import { LogViewer } from "../components/LogViewer";
import { Badge, Empty, ErrorBox, JobBadge, Loading, Spinner, formatBytes, formatNumber, useToast } from "../components/ui";

export function Diagnostics() {
  const { toast } = useToast();
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [doctorJob, setDoctorJob] = useState<Job | null>(null);
  const [calibrateJob, setCalibrateJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);

  async function load() {
    try {
      setSystem(await api.system());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 5_000);
    return () => clearInterval(interval);
  }, []);

  // follow the doctor/calibrate jobs
  useEffect(() => {
    const job = doctorJob ?? calibrateJob;
    if (!job || ["done", "failed", "cancelled", "interrupted"].includes(job.state)) return;
    const interval = setInterval(async () => {
      try {
        const fresh = await api.job(job.id);
        if (doctorJob) setDoctorJob(fresh);
        if (calibrateJob) setCalibrateJob(fresh);
      } catch {
        /* ignore */
      }
    }, 1_500);
    return () => clearInterval(interval);
  }, [doctorJob, calibrateJob]);

  async function startDoctor(calibrate: boolean) {
    setRunning(true);
    try {
      const payload: Record<string, unknown> = {};
      if (calibrate) payload.calibrate = true;
      const { job } = calibrate ? await api.calibrate(payload) : await api.doctor(payload);
      if (calibrate) setCalibrateJob(job);
      else setDoctorJob(job);
      waitForJob(job.id, (j) => {
        if (calibrate) setCalibrateJob(j);
        else setDoctorJob(j);
      }).finally(() => setRunning(false));
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
      setRunning(false);
    }
  }

  if (!system && !error) return <Loading />;
  if (error && !system) return <ErrorBox>{error}</ErrorBox>;

  const r = system?.resources;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Diagnostics</h1>
          <p className="subtitle">
            Host resources, environment health, and the real-game integration check.
          </p>
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => startDoctor(false)} disabled={running}>
            {running ? <Spinner size={12} /> : "♥"} Run doctor
          </button>
          <button
            className="btn"
            onClick={() => startDoctor(true)}
            disabled={running}
            title="Measures the real game's telemetry conventions; needs Trackmania running on a Windows host"
          >
            ⚗ Calibrate telemetry
          </button>
        </div>
      </div>

      <div className="grid cols-4">
        <div className="stat">
          <div className="label">CPU</div>
          <div className="value">{r?.cpu_count ?? "—"}</div>
          <div className="hint">
            load {r?.load_average?.map((l) => formatNumber(l, 2)).join(" / ") ?? "—"}
          </div>
        </div>
        <div className="stat">
          <div className="label">Memory</div>
          <div className="value">
            {r?.memory_used_fraction !== null && r?.memory_used_fraction !== undefined
              ? formatNumber(r.memory_used_fraction * 100, 0) + "%"
              : "—"}
          </div>
          <div className="hint">
            {formatBytes(r?.memory_available_bytes ?? null)} of {formatBytes(r?.memory_total_bytes ?? null)} free
          </div>
        </div>
        <div className="stat">
          <div className="label">Process</div>
          <div className="value" style={{ fontSize: 18 }}>{formatBytes(r?.process_memory_bytes ?? null)}</div>
          <div className="hint">RSS of this server</div>
        </div>
        <div className="stat">
          <div className="label">Disk</div>
          <div className="value">{formatBytes(r?.disk_free_bytes ?? null)}</div>
          <div className="hint">free of {formatBytes(r?.disk_total_bytes ?? null)}</div>
        </div>
      </div>

      <div className="section" style={{ marginTop: 24 }}>
        <h3>Environment</h3>
        <div className="table-wrap">
          <table>
            <tbody>
              <tr>
                <td className="dim">trackmania-ai</td>
                <td>{system?.version}</td>
                <td><Badge tone="green">ok</Badge></td>
              </tr>
              <tr>
                <td className="dim">python</td>
                <td className="mono">{system?.python}</td>
                <td><Badge tone="green">ok</Badge></td>
              </tr>
              <tr>
                <td className="dim">platform</td>
                <td className="mono">{system?.platform}</td>
                <td>
                  <Badge tone={system?.windows_host ? "green" : "amber"}>
                    {system?.windows_host ? "windows" : "not windows"}
                  </Badge>
                </td>
              </tr>
              <tr>
                <td className="dim">torch</td>
                <td>{system?.torch_installed ? "installed" : "missing"}</td>
                <td>
                  <Badge tone={system?.torch_installed ? "green" : "red"}>
                    {system?.torch_installed ? "ok" : "needed for training"}
                  </Badge>
                </td>
              </tr>
              <tr>
                <td className="dim">tminterface</td>
                <td>{system?.tminterface_installed ? "installed" : "not installed"}</td>
                <td>
                  <Badge tone={system?.tminterface_installed ? "green" : "amber"}>
                    {system?.tminterface_installed ? "ok" : "game integration unavailable"}
                  </Badge>
                </td>
              </tr>
              <tr>
                <td className="dim">real game integration</td>
                <td>
                  {system?.game_integration_possible
                    ? "possible (Windows + tminterface)"
                    : "not possible on this host"}
                </td>
                <td>
                  <Badge tone={system?.game_integration_possible ? "green" : "amber"}>
                    {system?.game_integration_possible ? "ready" : "unavailable"}
                  </Badge>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        {!system?.game_integration_possible && (
          <p className="faint" style={{ fontSize: 12, marginTop: 8 }}>
            The real-game integration needs Trackmania running on Windows with the{" "}
            <code className="mono">tminterface</code> package installed. This server can still train,
            evaluate and analyse against the simulated driver — those runs are labelled as simulated
            everywhere.
          </p>
        )}
      </div>

      <div className="section">
        <h3>Paths</h3>
        <div className="table-wrap">
          <table>
            <tbody>
              {system?.paths &&
                Object.entries(system.paths).map(([key, value]) => (
                  <tr key={key}>
                    <td className="dim">{key}</td>
                    <td className="mono">{value}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <h3>Doctor report</h3>
        {!doctorJob ? (
          <Empty icon="♥">Run the doctor for a full environment + game-integration report.</Empty>
        ) : (
          <div className="card">
            <div className="card-title">
              <h2>{doctorJob.description}</h2>
              <JobBadge state={doctorJob.state} />
            </div>
            {doctorJob.error && <ErrorBox>{doctorJob.error}</ErrorBox>}
            <LogViewer lines={doctorJob.log_tail} maxHeight={420} />
          </div>
        )}
      </div>

      <div className="section">
        <h3>Calibration</h3>
        {!calibrateJob ? (
          <Empty icon="⚗">
            Telemetry calibration measures how the real game's telemetry maps to metres and seconds.
            It only works on the game host.
          </Empty>
        ) : (
          <div className="card">
            <div className="card-title">
              <h2>{calibrateJob.description}</h2>
              <JobBadge state={calibrateJob.state} />
            </div>
            {calibrateJob.error && <ErrorBox>{calibrateJob.error}</ErrorBox>}
            <LogViewer lines={calibrateJob.log_tail} maxHeight={420} />
          </div>
        )}
      </div>
    </div>
  );
}
