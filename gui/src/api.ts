/**
 * Typed client for the local TrackmaniaAI backend (`tmai serve`).
 *
 * Every function maps to one route in `tmai/server/app.py`. The dev server proxies `/api`
 * to the backend, so all URLs here are relative -- the browser never needs to know where
 * the backend runs.
 */

// -- types -----------------------------------------------------------------------

export interface Health {
  status: string;
  version: string;
  started_utc: string;
}

export interface Resources {
  python?: string;
  platform?: string;
  cpu_count?: number;
  load_average?: number[] | null;
  memory_total_bytes?: number | null;
  memory_available_bytes?: number | null;
  memory_used_fraction?: number | null;
  process_memory_bytes?: number | null;
  disk_free_bytes?: number | null;
  disk_total_bytes?: number | null;
  [key: string]: unknown;
}

export interface SystemInfo {
  version: string;
  python?: string;
  platform?: string;
  windows_host: boolean;
  torch_installed: boolean;
  tminterface_installed: boolean;
  game_integration_possible: boolean;
  resources: Resources;
  paths: Record<string, string>;
}

export interface RunRow {
  /** Canonical id: the run directory name. All run endpoints are keyed by this. */
  name: string;
  run_dir: string;
  run_name: string;
  step: number;
  total_steps: number;
  progress_fraction: number;
  ended: boolean;
  simulated: boolean;
  driver: string;
  checkpoints: number;
}

export interface SeriesPoint {
  step: number;
  value: number;
}

export interface Series {
  name: string;
  group: string;
  steps: number[];
  values: number[];
}

export interface RunHistory {
  run_dir: string;
  max_points: number;
  total_records: number;
  available: Record<string, string[]>;
  series: Series[];
}

export interface EpisodeRow {
  step: number;
  episode: number;
  end_reason: string;
  finished?: boolean;
  game_finished?: boolean;
  invalid_finish?: boolean;
  track?: string;
  reward: number;
  progress: number;
  [key: string]: unknown;
}

export interface RunSnapshot {
  status: {
    run_name: string;
    step: number;
    total_steps: number;
    progress_fraction: number;
    ended: boolean;
    running: boolean;
    simulated: boolean;
    driver: string;
    checkpoints: number;
    [key: string]: unknown;
  };
  history: RunHistory;
  episodes: EpisodeRow[];
  evaluations: Record<string, unknown>[];
  checkpoints: { name: string; size?: number; mtime?: number }[];
  manifest: Record<string, unknown>;
  config_yaml: string | null;
  log_tail: string[];
}

export interface TrackEntryRow {
  name: string;
  identity: string;
  family: string | null;
  split_group: string;
  split: string;
  source: string | null;
  length: number;
  stats: Record<string, unknown> | null;
}

export interface TrackLibraryReport {
  num_tracks: number;
  num_split_groups: number;
  counts: Record<string, number>;
  families_by_split: Record<string, number>;
  split_groups_by_split: Record<string, number>;
  split_weights: Record<string, number>;
  tracks: TrackEntryRow[];
  geometry_by_split: Record<string, Record<string, number>>;
}

export interface TrackGeometry {
  name: string;
  length: number;
  closed: boolean;
  num_points: number;
  points: number[][];
  edges: { left: number[][]; right: number[][] };
  curvature: number[];
  corridor_half_width: number[];
  stats: Record<string, unknown>;
}

export interface ModelInfo {
  name: string;
  directory: string;
  source_checkpoint: string;
  source_run: string;
  step: number;
  gradient_steps: number;
  best_score: number | null;
  created_utc: string;
  algorithm: string;
  observation_dim: number | null;
  action_dim: number | null;
  tags: string[];
  notes: string;
  evaluation: Record<string, unknown>;
  weights: string;
}

export interface Job {
  id: string;
  kind: string;
  description: string;
  state: "queued" | "running" | "cancelling" | "done" | "failed" | "cancelled" | "interrupted";
  created_utc: string;
  started_utc: string | null;
  finished_utc: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  log_tail: string[];
  params: Record<string, unknown>;
}

export interface ReplayRow {
  path: string;
  name: string;
  episode: number;
  step: number;
  track: string;
  split: string;
  end_reason: string;
  finished: boolean;
  race_time: number;
  total_reward: number;
  progress_fraction: number;
  source: string;
  num_samples: number;
}

export interface EpisodeReplay {
  episode: number;
  step: number;
  track: string;
  split: string;
  end_reason: string;
  finished: boolean;
  race_time: number;
  total_reward: number;
  progress_fraction: number;
  source: string;
  num_samples: number;
  positions: number[][];
  speeds: number[];
  actions: number[][];
  rewards: number[];
  progress: number[];
  race_times: number[];
  metadata: Record<string, unknown>;
}

export interface SectorSummary {
  index: number;
  start_m: number;
  end_m: number;
  episodes_with_samples: number;
  speed_samples: number;
  mean_speed_mps: number | null;
  mean_abs_lateral_m: number | null;
  mean_lateral_fraction_of_half_width: number | null;
  sector_time_samples: number;
  mean_sector_time_s: number | null;
  median_sector_time_s: number | null;
  failure_count: number;
}

export interface ReplayAnalysis {
  schema_version: number;
  track: string;
  track_uid: string | null;
  track_length_m: number;
  closed: boolean;
  num_replays: number;
  num_samples: number;
  sector_count: number;
  sectors: SectorSummary[];
  slowest_sectors: number[];
  failure_reasons: Record<string, number>;
  failure_heatmap: {
    station_edges_m: number[];
    normalized_lateral_edges: number[];
    lateral_labels: string[];
    counts: number[][];
    events_with_location: number;
  };
}

export interface ReplayComparison {
  track: string;
  stations: number[];
  ai_times: number[];
  ghost_times: number[];
  gaps: number[];
  segment_gaps: number[];
  mean_gap: number;
  max_gap: number;
  min_gap: number;
  ai_finished: boolean;
  ghost_finished: boolean;
  ai_race_time: number;
  ghost_race_time: number;
  race_time_delta: number | null;
}

export interface DemoRow {
  name: string;
  path: string;
  steps: number;
  size: number;
}

export interface BenchmarkRow {
  name: string;
  path: string;
  benchmark: string;
  created_utc: string;
  ranking: string[];
  splits: string[];
  seed_repeats: number;
}

export interface ConfigDoc {
  path: string;
  yaml: string;
  config: Record<string, unknown>;
  problems: string[];
}

export interface ValidateResult {
  ok: boolean;
  problems: string[];
  config?: Record<string, unknown>;
}

export interface Tick {
  type: "tick";
  utc: string;
  system: Resources;
  runs: { name: string; step: number; updated_utc: string; running: boolean }[];
  jobs: { id: string; kind: string; state: string; description: string }[];
}

// -- helpers ---------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

const get = <T>(path: string) => request<T>(path);
const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });
const del = <T>(path: string) => request<T>(path, { method: "DELETE" });

// -- endpoints -------------------------------------------------------------------

export const api = {
  health: () => get<Health>("/api/health"),
  system: () => get<SystemInfo>("/api/system"),

  runs: () => get<{ runs: RunRow[]; runs_dir: string }>("/api/runs"),
  run: (name: string) => get<RunSnapshot>(`/api/runs/${encodeURIComponent(name)}`),
  runMetrics: (name: string, metrics?: string, maxPoints = 400) =>
    get<{ run: string; history: RunHistory }>(
      `/api/runs/${encodeURIComponent(name)}/metrics?max_points=${maxPoints}` +
        (metrics ? `&metrics=${encodeURIComponent(metrics)}` : "")
    ),
  runLog: (name: string, lines = 200) =>
    get<{ run: string; log: string[] }>(
      `/api/runs/${encodeURIComponent(name)}/log?lines=${lines}`
    ),
  runCheckpoints: (name: string) =>
    get<{ run: string; checkpoints: { name: string; size: number }[]; best: { name: string } | null }>(
      `/api/runs/${encodeURIComponent(name)}/checkpoints`
    ),

  tracks: (directory?: string) =>
    get<{ directory: string; report: TrackLibraryReport | null; synthetic: string[]; error?: string }>(
      "/api/tracks" + (directory ? `?directory=${encodeURIComponent(directory)}` : "")
    ),
  trackGeometry: (params: { name?: string; synthetic?: string; directory?: string }) => {
    const query = new URLSearchParams();
    if (params.name) query.set("name", params.name);
    if (params.synthetic) query.set("synthetic", params.synthetic);
    if (params.directory) query.set("directory", params.directory);
    return get<TrackGeometry>(`/api/tracks/geometry?${query.toString()}`);
  },

  models: () => get<{ models: ModelInfo[]; directory: string }>("/api/models"),
  model: (name: string) => get<ModelInfo>(`/api/models/${encodeURIComponent(name)}`),
  modelRegister: (payload: {
    name: string;
    checkpoint: string;
    tags?: string[];
    notes?: string;
    overwrite?: boolean;
  }) => post<ModelInfo>("/api/models/register", payload),
  modelDelete: (name: string) => del<{ deleted: string }>(`/api/models/${encodeURIComponent(name)}`),

  benchmarks: () => get<{ benchmarks: BenchmarkRow[]; directory: string }>("/api/benchmarks"),
  benchmark: (payload: {
    config_path?: string;
    models: { label: string; checkpoint: string }[];
    splits: string[];
    episodes: number;
    name: string;
  }) => post<{ job: Job }>("/api/benchmark", payload),

  demos: () => get<{ demos: DemoRow[]; directory: string }>("/api/demos"),

  replays: (run: string) =>
    get<{ run: string; replays: ReplayRow[]; directory: string }>(
      `/api/replays?run=${encodeURIComponent(run)}`
    ),
  replay: (run: string, file: string) =>
    get<EpisodeReplay>(`/api/replays/${encodeURIComponent(run)}/${encodeURIComponent(file)}`),
  replayCompare: (payload: { run: string; a: string; b: string; track?: string }) =>
    post<ReplayComparison>("/api/replays/compare", payload),
  runAnalysis: (run: string, track: string, sectors = 20) =>
    get<ReplayAnalysis>(
      `/api/runs/${encodeURIComponent(run)}/analysis?track=${encodeURIComponent(track)}&sectors=${sectors}`
    ),

  config: () => get<ConfigDoc>("/api/config"),
  configValidate: (yaml: string) =>
    post<ValidateResult>("/api/config/validate", { yaml }),

  jobs: (kind?: string) =>
    get<{ jobs: Job[]; workers: number }>("/api/jobs" + (kind ? `?kind=${kind}` : "")),
  job: (id: string) => get<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  jobCancel: (id: string) => post<{ job: Job }>(`/api/jobs/${encodeURIComponent(id)}/cancel`),

  train: (payload: {
    config_path?: string;
    config_yaml?: string;
    overrides?: Record<string, string>;
    run_name?: string;
    steps?: number;
    resume?: string;
    allow_simulated_driver?: boolean;
  }) => post<{ job: Job }>("/api/train", payload),
  evaluate: (payload: {
    config_path?: string;
    run?: string;
    checkpoint?: string;
    splits: string[];
    episodes: number;
    stochastic?: boolean;
  }) => post<{ job: Job }>("/api/eval", payload),
  calibrate: (payload: { config_path?: string; steps?: number }) =>
    post<{ job: Job }>("/api/calibrate", payload),
  doctor: (payload: { config_path?: string; calibrate?: boolean; steps?: number }) =>
    post<{ job: Job }>("/api/doctor", payload),
};

/** Open the live-update WebSocket with a bounded exponential reconnect policy. */
export function connectWebSocket(onTick: (tick: Tick) => void): () => void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${window.location.host}/api/ws`;
  let socket: WebSocket | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let retryDelay = 750;
  let closed = false;

  const connect = () => {
    if (closed) return;
    const candidate = new WebSocket(url);
    socket = candidate;
    candidate.onopen = () => {
      if (socket === candidate) retryDelay = 750;
    };
    candidate.onmessage = (event) => {
      if (socket !== candidate || closed) return;
      try {
        onTick(JSON.parse(event.data) as Tick);
      } catch {
        /* ignore malformed frames */
      }
    };
    candidate.onclose = () => {
      if (socket !== candidate || closed) return;
      socket = null;
      retryTimer = setTimeout(connect, retryDelay);
      retryDelay = Math.min(15_000, Math.round(retryDelay * 1.8));
    };
    candidate.onerror = () => candidate.close();
  };

  connect();
  return () => {
    closed = true;
    if (retryTimer !== null) clearTimeout(retryTimer);
    retryTimer = null;
    const current = socket;
    socket = null;
    current?.close();
  };
}

/** Poll a job until it reaches a terminal state. Returns the final job. */
export async function waitForJob(
  id: string,
  onUpdate?: (job: Job) => void,
  timeoutMs = 600_000
): Promise<Job> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await api.job(id);
    onUpdate?.(job);
    if (["done", "failed", "cancelled", "interrupted"].includes(job.state)) {
      return job;
    }
    if (Date.now() > deadline) {
      throw new Error(`job ${id} did not finish in time (state: ${job.state})`);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}
