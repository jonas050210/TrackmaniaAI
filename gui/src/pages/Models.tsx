/** Models page: the model registry — list, register, inspect, delete. */

import { useEffect, useState } from "react";
import { api, type ModelInfo, type RunRow } from "../api";
import { Badge, Empty, ErrorBox, Loading, Spinner, formatNumber, formatPercent, useToast } from "../components/ui";

export function Models() {
  const { toast } = useToast();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  const [notes, setNotes] = useState("");
  const [tag, setTag] = useState("");
  const [registering, setRegistering] = useState(false);
  const [selected, setSelected] = useState<ModelInfo | null>(null);

  async function load() {
    try {
      const [modelsData, runsData] = await Promise.all([api.models(), api.runs()]);
      setModels(modelsData.models);
      setRuns(runsData.runs);
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

  async function onRegister() {
    setRegistering(true);
    try {
      await api.modelRegister({
        name: name.trim(),
        checkpoint: checkpoint.trim(),
        notes: notes.trim() || undefined,
        tags: tag.trim() ? [tag.trim()] : undefined,
      });
      toast(`Model ${name.trim()} registered`, "success");
      setName("");
      setCheckpoint("");
      setNotes("");
      setTag("");
      load();
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
    } finally {
      setRegistering(false);
    }
  }

  async function onDelete(modelName: string) {
    if (!window.confirm(`Delete model ${modelName}? This removes its weights from the registry.`)) return;
    try {
      await api.modelDelete(modelName);
      toast(`Model ${modelName} deleted`, "success");
      if (selected?.name === modelName) setSelected(null);
      load();
    } catch (err) {
      toast(err instanceof Error ? err.message : String(err), "error");
    }
  }

  async function onSelect(model: ModelInfo) {
    try {
      setSelected(await api.model(model.name));
    } catch {
      setSelected(model);
    }
  }

  if (loading) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Models</h1>
          <p className="subtitle">
            Named, self-contained policies. Register a checkpoint, evaluate it, benchmark it, or
            delete it — without touching the run that produced it.
          </p>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card-title">
            <h2>Register a model</h2>
          </div>
          <div className="field">
            <label>Name</label>
            <input placeholder="e.g. sac-2m-steps" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="field">
            <label>Checkpoint</label>
            <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
              <option value="">— pick a run's latest checkpoint —</option>
              {runs.map((run) => (
                <option key={run.run_dir} value={run.name}>
                  {run.run_name || run.name} (step {run.step})
                </option>
              ))}
            </select>
            <span className="hint">A run directory resolves to its latest checkpoint; a .pt path is used directly.</span>
          </div>
          <div className="grid cols-2" style={{ gap: 12 }}>
            <div className="field">
              <label>Tag (optional)</label>
              <input placeholder="e.g. curriculum" value={tag} onChange={(e) => setTag(e.target.value)} />
            </div>
            <div className="field">
              <label>Notes (optional)</label>
              <input placeholder="what is special about this one?" value={notes} onChange={(e) => setNotes(e.target.value)} />
            </div>
          </div>
          <button
            className="btn primary"
            onClick={onRegister}
            disabled={registering || !name.trim() || !checkpoint.trim()}
          >
            {registering ? <Spinner size={12} /> : "◆"} Register
          </button>
        </div>

        <div className="card">
          <div className="card-title">
            <h2>Selected model</h2>
          </div>
          {!selected ? (
            <p className="dim">Select a model from the table below.</p>
          ) : (
            <div>
              <h2 style={{ marginBottom: 8 }}>{selected.name}</h2>
              <div className="btn-row" style={{ marginBottom: 12 }}>
                {selected.tags.map((t) => (
                  <Badge key={t} tone="violet">{t}</Badge>
                ))}
              </div>
              <table style={{ fontSize: 13 }}>
                <tbody>
                  <tr><td className="dim">step</td><td>{selected.step}</td></tr>
                  <tr><td className="dim">gradient steps</td><td>{selected.gradient_steps}</td></tr>
                  <tr><td className="dim">best score</td><td>{formatNumber(selected.best_score, 3)}</td></tr>
                  <tr><td className="dim">observation dim</td><td>{selected.observation_dim ?? "—"}</td></tr>
                  <tr><td className="dim">action dim</td><td>{selected.action_dim ?? "—"}</td></tr>
                  <tr><td className="dim">algorithm</td><td>{selected.algorithm}</td></tr>
                  <tr><td className="dim">created</td><td>{new Date(selected.created_utc).toLocaleString()}</td></tr>
                  <tr><td className="dim">source</td><td className="mono" style={{ wordBreak: "break-all" }}>{selected.source_checkpoint || "—"}</td></tr>
                  <tr><td className="dim">notes</td><td>{selected.notes || "—"}</td></tr>
                </tbody>
              </table>
              {selected.evaluation && Object.keys(selected.evaluation).length > 0 && (
                <>
                  <h3 style={{ margin: "16px 0 8px" }}>Latest evaluation</h3>
                  <pre className="log" style={{ maxHeight: 180 }}>
                    {JSON.stringify(selected.evaluation, null, 2)}
                  </pre>
                </>
              )}
              <div className="btn-row" style={{ marginTop: 12 }}>
                <button className="btn danger sm" onClick={() => onDelete(selected.name)}>Delete</button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <h3>Registry</h3>
        {models.length === 0 ? (
          <Empty icon="◆">
            No models registered. Train a run, then register its checkpoint here.
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Step</th>
                  <th>Grad steps</th>
                  <th>Best score</th>
                  <th>Dims</th>
                  <th>Tags</th>
                  <th>Created</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr
                    key={model.name}
                    onClick={() => onSelect(model)}
                    style={{ cursor: "pointer" }}
                  >
                    <td><strong>{model.name}</strong></td>
                    <td>{model.step}</td>
                    <td>{model.gradient_steps}</td>
                    <td>{formatNumber(model.best_score, 3)}</td>
                    <td className="dim">
                      {model.observation_dim ?? "?"} → {model.action_dim ?? "?"}
                    </td>
                    <td>
                      {model.tags.length ? (
                        model.tags.map((t) => <Badge key={t} tone="violet">{t}</Badge>)
                      ) : (
                        <span className="faint">—</span>
                      )}
                    </td>
                    <td className="dim">{new Date(model.created_utc).toLocaleDateString()}</td>
                    <td>
                      <button
                        className="btn danger sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDelete(model.name);
                        }}
                      >
                        delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <p className="faint" style={{ fontSize: 12 }}>
        Registered models are copies — deleting a run never breaks a model. Scores shown are the
        training-time best, not an evaluation; use Evaluate & Benchmark for that.
        {formatPercent(0)}
      </p>
    </div>
  );
}
