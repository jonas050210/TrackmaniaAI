/** Training page: start a run (config picker or YAML editor) and watch jobs live. */

import { useEffect, useState } from "react";
import { api, waitForJob, type ConfigDoc, type Job } from "../api";
import { LogViewer } from "../components/LogViewer";
import { Badge, ErrorBox, JobBadge, Loading, OkBox, Spinner, useToast } from "../components/ui";

const CONFIG_PRESETS = [
  { label: "Smoke (simulated, seconds)", path: "tmai/configs/smoke.yaml" },
  { label: "Multi-track smoke (simulated)", path: "tmai/configs/multitrack_smoke.yaml" },
  { label: "Pipeline smoke (BC + curriculum + replays)", path: "tmai/configs/pipeline_smoke.yaml" },
  { label: "Default (real game)", path: "tmai/configs/default.yaml" },
];

export function Training() {
  const { toast } = useToast();
  const [config, setConfig] = useState<ConfigDoc | null>(null);
  const [yaml, setYaml] = useState("");
  const [preset, setPreset] = useState(CONFIG_PRESETS[0].path);
  const [runName, setRunName] = useState("");
  const [steps, setSteps] = useState<number | "">("");
  const [resume, setResume] = useState("");
  const [allowSimulated, setAllowSimulated] = useState(true);
  const [validating, setValidating] = useState(false);
  const [validation, setValidation] = useState<{ ok: boolean; problems: string[] } | null>(null);
  const [starting, setStarting] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadConfig() {
    try {
      const doc = await api.config();
      setConfig(doc);
      setYaml(doc.yaml);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadJobs() {
    try {
      const data = await api.jobs();
      setJobs(data.jobs.filter((j) => ["train", "eval", "benchmark", "calibrate", "doctor"].includes(j.kind)));
    } catch {
      /* keep old list */
    }
  }

  useEffect(() => {
    setLoading(true);
    Promise.all([loadConfig(), loadJobs()]).finally(() => setLoading(false));
    const interval = setInterval(loadJobs, 3_000);
    return () => clearInterval(interval);
  }, []);

  // follow the selected job live
  useEffect(() => {
    if (!selectedJob) return;
    if (["done", "failed", "cancelled", "interrupted"].includes(selectedJob.state)) return;
    const interval = setInterval(async () => {
      try {
        const job = await api.job(selectedJob.id);
        setSelectedJob(job);
      } catch {
        /* job may be gone */
      }
    }, 1_500);
    return () => clearInterval(interval);
  }, [selectedJob?.id, selectedJob?.state]);

  async function onValidate() {
    setValidating(true);
    try {
      const result = await api.configValidate(yaml);
      setValidation({ ok: result.ok, problems: result.problems });
    } catch (err) {
      setValidation({ ok: false, problems: [err instanceof Error ? err.message : String(err)] });
    } finally {
      setValidating(false);
    }
  }

  async function onStart() {
    setStarting(true);
    setValidation(null);
    try {
      const payload: Record<string, unknown> = {
        config_yaml: yaml,
        allow_simulated_driver: allowSimulated,
      };
      if (runName.trim()) payload.run_name = runName.trim();
      if (steps !== "") payload.steps = Number(steps);
      if (resume.trim()) payload.resume = resume.trim();
      const { job } = await api.train(payload);
      toast(`Training job started: ${job.id}`, "success");
      setSelectedJob(job);
      loadJobs();
      waitForJob(job.id, (j) => setSelectedJob(j)).then((finished) => {
        loadJobs();
        if (finished.state === "done") toast("Training finished", "success");
        else if (finished.state === "failed") toast(`Training failed: ${finished.error}`, "error");
      });
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
    } finally {
      setStarting(false);
    }
  }

  async function onCancel(id: string) {
    try {
      await api.jobCancel(id);
      loadJobs();
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
    }
  }

  if (loading && !config) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Training</h1>
          <p className="subtitle">
            Start a run from the default configuration or paste your own YAML. Jobs run in the
            background; this page follows them live.
          </p>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card-title">
            <h2>New training run</h2>
          </div>

          <div className="field">
            <label>Configuration preset</label>
            <select
              value={preset}
              onChange={async (e) => {
                setPreset(e.target.value);
                // load the preset's YAML text through the validator endpoint is not possible;
                // the backend stores whatever YAML we send, so fetch the file via the config
                // endpoint only for the default. For presets, send the path instead.
              }}
            >
              {CONFIG_PRESETS.map((p) => (
                <option key={p.path} value={p.path}>{p.label}</option>
              ))}
              <option value="__custom__">Custom YAML (edit below)</option>
            </select>
            <span className="hint">
              {preset === "__custom__"
                ? "Editing the YAML below; it is validated before training starts."
                : `Will train from ${preset}. Switch to "Custom YAML" to edit.`}
            </span>
          </div>

          {preset !== "__custom__" ? (
            <div className="field">
              <label>Run options</label>
              <div className="grid cols-2" style={{ gap: 12 }}>
                <input
                  type="text"
                  placeholder="run name (optional)"
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                />
                <input
                  type="number"
                  placeholder="total steps (optional)"
                  value={steps}
                  onChange={(e) => setSteps(e.target.value === "" ? "" : Number(e.target.value))}
                />
              </div>
              <input
                type="text"
                placeholder="resume from checkpoint or run dir (optional)"
                value={resume}
                onChange={(e) => setResume(e.target.value)}
                style={{ marginTop: 8 }}
              />
            </div>
          ) : (
            <div className="field">
              <label>Configuration YAML</label>
              <textarea
                rows={18}
                value={yaml}
                onChange={(e) => setYaml(e.target.value)}
                spellCheck={false}
              />
              <div className="btn-row" style={{ marginTop: 8 }}>
                <button className="btn sm" onClick={onValidate} disabled={validating}>
                  {validating ? <Spinner size={12} /> : "Validate"}
                </button>
                <label className="dim" style={{ fontSize: 12, display: "flex", gap: 6, alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={allowSimulated}
                    onChange={(e) => setAllowSimulated(e.target.checked)}
                  />
                  allow simulated driver
                </label>
              </div>
              {validation && (
                <div style={{ marginTop: 8 }}>
                  {validation.ok ? (
                    <OkBox>Configuration is valid.</OkBox>
                  ) : (
                    <ErrorBox>
                      {validation.problems.map((p, i) => (
                        <div key={i}>{p}</div>
                      ))}
                    </ErrorBox>
                  )}
                </div>
              )}
            </div>
          )}

          {preset === "__custom__" && (
            <div className="field">
              <label>Run options</label>
              <div className="grid cols-2" style={{ gap: 12 }}>
                <input
                  type="text"
                  placeholder="run name (optional)"
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                />
                <input
                  type="number"
                  placeholder="total steps (optional)"
                  value={steps}
                  onChange={(e) => setSteps(e.target.value === "" ? "" : Number(e.target.value))}
                />
              </div>
            </div>
          )}

          <div className="btn-row">
            <button
              className="btn primary"
              onClick={async () => {
                if (preset !== "__custom__") {
                  // send the preset path directly
                  setStarting(true);
                  try {
                    const payload: Record<string, unknown> = { config_path: preset };
                    if (runName.trim()) payload.run_name = runName.trim();
                    if (steps !== "") payload.steps = Number(steps);
                    if (resume.trim()) payload.resume = resume.trim();
                    const { job } = await api.train(payload);
                    toast(`Training job started: ${job.id}`, "success");
                    setSelectedJob(job);
                    loadJobs();
                    waitForJob(job.id, (j) => setSelectedJob(j)).then((finished) => {
                      loadJobs();
                      toast(
                        finished.state === "done"
                          ? "Training finished"
                          : `Training ${finished.state}: ${finished.error ?? ""}`,
                        finished.state === "done" ? "success" : "error"
                      );
                    });
                  } catch (err) {
                    toast(err instanceof Error ? err.message : String(err), "error");
                  } finally {
                    setStarting(false);
                  }
                } else {
                  onStart();
                }
              }}
              disabled={starting}
            >
              {starting ? <Spinner size={12} /> : "▶"} Start training
            </button>
          </div>
          <p className="hint" style={{ marginTop: 12, fontSize: 12 }}>
            Training runs as a background job on the server host. Simulated-driver runs are labelled
            as such everywhere — they exercise the pipeline, not the real game.
          </p>
        </div>

        <div className="card">
          <div className="card-title">
            <h2>Job</h2>
            {selectedJob && <JobBadge state={selectedJob.state} />}
          </div>
          {!selectedJob ? (
            <p className="dim">Select a job below to follow it.</p>
          ) : (
            <div>
              <p className="dim" style={{ fontSize: 12 }}>
                <Badge tone="violet">{selectedJob.kind}</Badge> {selectedJob.description || selectedJob.id}
              </p>
              {selectedJob.error && <ErrorBox>{selectedJob.error}</ErrorBox>}
              {selectedJob.result && (
                <pre className="log" style={{ maxHeight: 140 }}>
                  {JSON.stringify(selectedJob.result, null, 2)}
                </pre>
              )}
              <LogViewer lines={selectedJob.log_tail} maxHeight={320} />
              {["queued", "running"].includes(selectedJob.state) && (
                <div className="btn-row" style={{ marginTop: 12 }}>
                  <button className="btn danger sm" onClick={() => onCancel(selectedJob.id)}>
                    Cancel job
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <h3>Jobs</h3>
        {jobs.length === 0 ? (
          <p className="dim">No jobs yet.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Kind</th>
                  <th>Description</th>
                  <th>State</th>
                  <th>Created</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    key={job.id}
                    onClick={() => setSelectedJob(job)}
                    style={{ cursor: "pointer" }}
                  >
                    <td><Badge tone="violet">{job.kind}</Badge></td>
                    <td>{job.description || job.id}</td>
                    <td><JobBadge state={job.state} /></td>
                    <td className="dim">{new Date(job.created_utc).toLocaleString()}</td>
                    <td>
                      {["queued", "running"].includes(job.state) && (
                        <button
                          className="btn danger sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            onCancel(job.id);
                          }}
                        >
                          cancel
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
