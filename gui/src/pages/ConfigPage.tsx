/** Configuration page: view/edit the default config YAML with live validation. */

import { useEffect, useState } from "react";
import { api, type ConfigDoc } from "../api";
import { ErrorBox, Loading, OkBox, Spinner, useToast } from "../components/ui";

export function ConfigPage() {
  const { toast } = useToast();
  const [doc, setDoc] = useState<ConfigDoc | null>(null);
  const [yaml, setYaml] = useState("");
  const [loading, setLoading] = useState(true);
  const [validating, setValidating] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; problems: string[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .config()
      .then((d) => {
        setDoc(d);
        setYaml(d.yaml);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  async function validate() {
    setValidating(true);
    try {
      const res = await api.configValidate(yaml);
      setResult({ ok: res.ok, problems: res.problems });
      toast(res.ok ? "Configuration is valid" : "Configuration has problems", res.ok ? "success" : "error");
    } catch (err) {
      setResult({ ok: false, problems: [err instanceof Error ? err.message : String(err)] });
    } finally {
      setValidating(false);
    }
  }

  if (loading) return <Loading />;
  if (error) return <ErrorBox>{error}</ErrorBox>;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Configuration</h1>
          <p className="subtitle">
            The default configuration the server uses. Validate it here, or copy it as the starting
            point for a training run.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <h2>{doc?.path ?? "config.yaml"}</h2>
          <button className="btn primary" onClick={validate} disabled={validating}>
            {validating ? <Spinner size={12} /> : "⚙"} Validate
          </button>
        </div>
        <textarea rows={28} value={yaml} onChange={(e) => setYaml(e.target.value)} spellCheck={false} />
        <div style={{ marginTop: 12 }}>
          {result &&
            (result.ok ? (
              <OkBox>Configuration is valid.</OkBox>
            ) : (
              <ErrorBox>
                {result.problems.map((p, i) => (
                  <div key={i}>{p}</div>
                ))}
              </ErrorBox>
            ))}
        </div>
      </div>

      <div className="section">
        <h3>How configuration works</h3>
        <div className="card">
          <ul className="tight">
            <li>
              Every run saves the exact configuration it used as <code className="mono">config.yaml</code> in
              its run directory, and <code className="mono">tmai eval --checkpoint &lt;run&gt;</code> picks
              that up automatically — you never evaluate against a mismatched observation layout.
            </li>
            <li>
              CLI overrides use dotted keys: <code className="mono">--set sac.gamma=0.98 --set train.total_steps=100000</code>.
            </li>
            <li>
              The training page sends YAML text to the server, which validates it before starting a job.
            </li>
            <li>
              Sections: <code className="mono">driver</code>, <code className="mono">track</code>,{" "}
              <code className="mono">env</code> (observation/reward/termination), <code className="mono">sac</code>,{" "}
              <code className="mono">curriculum</code>, <code className="mono">bc</code>,{" "}
              <code className="mono">train</code>.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
