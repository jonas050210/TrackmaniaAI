"""The GUI backend: every route over HTTP, jobs end to end, and the WebSocket tick.

Uses FastAPI's TestClient, so no server process is needed. The one heavyweight test starts
a real training subprocess through ``POST /api/train`` -- the same path the GUI's "start
training" button uses -- and waits for the job to finish.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tmai.config import RunConfig
from tmai.server.app import ServerConfig, create_app
from tmai.server.jobs import JobManager, JobState

# -- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def server_dir(tmp_path):
    """An isolated server data layout: runs/, models/, demos/, benchmarks/, state/."""
    layout = {
        "runs": tmp_path / "runs",
        "tracks": tmp_path / "tracks",
        "models": tmp_path / "models",
        "demos": tmp_path / "demos",
        "benchmarks": tmp_path / "benchmarks",
        "state": tmp_path / "state",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


@pytest.fixture
def client(server_dir):
    config = ServerConfig(
        runs_dir=str(server_dir["runs"]),
        tracks_dir=str(server_dir["tracks"]),
        models_dir=str(server_dir["models"]),
        demos_dir=str(server_dir["demos"]),
        benchmarks_dir=str(server_dir["benchmarks"]),
        static_dir=str(server_dir["state"] / "no-frontend-here"),
        state_dir=str(server_dir["state"]),
    )
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


def _write_run(runs_dir: Path, name: str, *, steps: int = 3, episodes: int = 2) -> Path:
    """Create a minimal but realistic run directory."""
    run_dir = runs_dir / name
    (run_dir / "replays").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_name": name,
                "created_utc": "2026-10-08T00:00:00+00:00",
                "step": steps * 100,
                "config": {"driver": {"kind": "simulated"}},
                "driver": "simulated",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text("driver:\n  kind: simulated\n", encoding="utf-8")
    metrics = [
        {
            "step": (i + 1) * 100,
            "env/episode_reward": 1.0 + i,
            "env/progress_fraction": 0.1 * (i + 1),
            "sac/critic_loss": 0.5,
            "system/cpu_count": 8.0,
        }
        for i in range(steps)
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics) + "\n", encoding="utf-8"
    )
    events = [
        {
            "event": "episode_end",
            "t": float(i),
            "step": (i + 1) * 100,
            "episode": i + 1,
            "end_reason": "time_limit",
            "reward": 2.0,
            "progress": 20.0,
        }
        for i in range(episodes)
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    (run_dir / "run.log").write_text("line one\nline two\nline three\n", encoding="utf-8")
    return run_dir


def _write_track(tracks_dir: Path, name: str = "oval") -> Path:
    from tmai.tracks.synthetic import build_synthetic

    track = build_synthetic(name)
    path = tracks_dir / f"{name}.json"
    track.save(path)
    return path


def _write_demo(demos_dir: Path, name: str = "lap.jsonl", steps: int = 5) -> Path:
    import numpy as np

    from tmai.training.demos import Demonstration

    demo = Demonstration(
        observations=np.zeros((steps, 4), dtype=np.float32),
        actions=np.tile([0.1, 0.9, 0.0], (steps, 1)).astype(np.float32),
    )
    path = demos_dir / name
    demo.save(path)
    return path


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in ("done", "failed", "cancelled", "interrupted"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


# -- health & system ----------------------------------------------------------------------


class TestHealthAndSystem:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"]

    def test_system_reports_resources_and_capabilities(self, client):
        data = client.get("/api/system").json()
        assert data["python"]
        assert "resources" in data
        assert data["resources"]["cpu_count"] >= 1
        assert "windows_host" in data
        assert "game_integration_possible" in data
        assert data["paths"]["runs"].endswith("runs")

    def test_root_reports_missing_frontend(self, client):
        response = client.get("/")
        assert response.status_code == 404
        assert "frontend not built" in response.json()["error"]


# -- runs ----------------------------------------------------------------------------------


class TestRuns:
    def test_list_runs(self, client, server_dir):
        _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_alpha")
        _write_run(server_dir["runs"], "2026-01-02T00-00-00Z_beta")
        data = client.get("/api/runs").json()
        names = [r["run_name"] for r in data["runs"]]
        assert "2026-01-01T00-00-00Z_alpha" in names
        assert "2026-01-02T00-00-00Z_beta" in names

    def test_run_detail(self, client, server_dir):
        _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_alpha")
        data = client.get("/api/runs/2026-01-01T00-00-00Z_alpha").json()
        assert data["manifest"]["run_name"] == "2026-01-01T00-00-00Z_alpha"
        assert data["status"]["step"] == 300
        assert data["history"]["total_records"] == 3
        assert data["log_tail"][-1] == "line three"
        assert data["config_yaml"].startswith("driver:")

    def test_run_metrics_filter(self, client, server_dir):
        _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_alpha")
        data = client.get(
            "/api/runs/2026-01-01T00-00-00Z_alpha/metrics",
            params={"metrics": "env/episode_reward,sac/critic_loss"},
        ).json()
        names = {s["name"] for s in data["history"]["series"]}
        assert names == {"env/episode_reward", "sac/critic_loss"}

    def test_run_log_tail(self, client, server_dir):
        _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_alpha")
        data = client.get(
            "/api/runs/2026-01-01T00-00-00Z_alpha/log", params={"lines": 2}
        ).json()
        assert data["log"] == ["line two", "line three"]

    def test_run_checkpoints(self, client, server_dir):
        run_dir = _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_alpha")
        (run_dir / "checkpoint_000000100.pt").write_bytes(b"fake")
        (run_dir / "best.pt").write_bytes(b"fake")
        data = client.get("/api/runs/2026-01-01T00-00-00Z_alpha/checkpoints").json()
        assert [c["name"] for c in data["checkpoints"]] == ["checkpoint_000000100.pt"]
        assert data["best"]["name"] == "best.pt"

    def test_unknown_run_is_404(self, client):
        assert client.get("/api/runs/nope").status_code == 404

    def test_run_resolves_by_directory_name_and_by_manifest_name(self, client, server_dir):
        """The canonical id is the directory name; the manifest run_name also works.

        ``--run-name`` can make the manifest label differ from the directory name, and a UI
        that links by the wrong one 404s. Both must resolve.
        """
        import json as _json

        run_dir = _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_alpha")
        manifest = _json.loads((run_dir / "manifest.json").read_text())
        manifest["run_name"] = "alpha-label"
        (run_dir / "manifest.json").write_text(_json.dumps(manifest))

        by_dir = client.get("/api/runs/2026-01-01T00-00-00Z_alpha")
        by_label = client.get("/api/runs/alpha-label")
        assert by_dir.status_code == 200
        assert by_label.status_code == 200
        assert by_dir.json()["status"]["run_dir"] == by_label.json()["status"]["run_dir"]

        # the list reports both identifiers
        rows = client.get("/api/runs").json()["runs"]
        assert rows[0]["name"] == "2026-01-01T00-00-00Z_alpha"
        assert rows[0]["run_name"] == "alpha-label"

    def test_path_traversal_rejected(self, client):
        # ".." and "a/b" must never resolve outside the runs directory; the encoded form
        # (%2F) is the interesting one, because it arrives as a literal in the path param.
        for bad in ("..", "a/b", "..%2Fstate", "..%2f..%2fetc"):
            response = client.get(f"/api/runs/{bad}")
            assert response.status_code in (400, 404), bad


# -- tracks --------------------------------------------------------------------------------


class TestTracks:
    def test_library_report(self, client, server_dir):
        _write_track(server_dir["tracks"], "oval")
        _write_track(server_dir["tracks"], "straight")
        data = client.get("/api/tracks").json()
        assert data["report"] is not None
        assert data["report"]["num_tracks"] == 2
        assert set(data["synthetic"]) == {"straight", "oval", "s_curve", "figure_eight"}

    def test_missing_directory_is_reported_not_raised(self, client):
        data = client.get("/api/tracks", params={"directory": "/nonexistent/dir"}).json()
        assert data["report"] is None
        assert "not found" in data["error"]

    def test_geometry_for_recorded_track(self, client, server_dir):
        _write_track(server_dir["tracks"], "oval")
        data = client.get(
            "/api/tracks/geometry", params={"name": "oval"}
        ).json()
        assert data["name"] == "oval"
        assert data["length"] > 0
        assert len(data["points"]) == len(data["edges"]["left"])
        assert len(data["curvature"]) == len(data["points"])
        assert "curvature_mean" in data["stats"]

    def test_geometry_for_synthetic_track(self, client):
        data = client.get(
            "/api/tracks/geometry", params={"synthetic": "straight"}
        ).json()
        assert data["name"] == "straight"
        assert data["length"] == pytest.approx(200.0)

    def test_geometry_unknown_track_404(self, client):
        assert client.get("/api/tracks/geometry", params={"name": "nope"}).status_code == 404
        assert (
            client.get("/api/tracks/geometry", params={"synthetic": "nope"}).status_code == 404
        )
        assert client.get("/api/tracks/geometry").status_code == 400


# -- models --------------------------------------------------------------------------------


class TestModels:
    def _checkpoint(self, server_dir, step: int = 10) -> Path:
        import numpy as np

        from tmai.agents.sac import SACLearner
        from tmai.models.networks import NetworkConfig
        from tmai.training.checkpoint import save_checkpoint

        config = RunConfig().sac
        config.network = NetworkConfig(hidden_sizes=[16, 16])
        learner = SACLearner(
            observation_dim=4,
            action_dim=3,
            config=config,
            action_low=np.array([-1.0, 0.0, 0.0]),
            action_high=np.array([1.0, 1.0, 1.0]),
            seed=0,
        )
        return save_checkpoint(
            server_dir["runs"] / "src", step=step, learner=learner, config={"sac": {}}
        )

    def test_register_list_info_delete(self, client, server_dir):
        checkpoint = self._checkpoint(server_dir)
        response = client.post(
            "/api/models/register",
            json={"name": "api-model", "checkpoint": str(checkpoint), "tags": ["t"]},
        )
        assert response.status_code == 200
        info = response.json()
        assert info["name"] == "api-model"
        assert info["observation_dim"] == 4

        models = client.get("/api/models").json()["models"]
        assert [m["name"] for m in models] == ["api-model"]

        detail = client.get("/api/models/api-model").json()
        assert detail["step"] == 10

        assert client.delete("/api/models/api-model").status_code == 200
        assert client.get("/api/models").json()["models"] == []
        assert client.get("/api/models/api-model").status_code == 404

    def test_register_validation(self, client, server_dir):
        checkpoint = self._checkpoint(server_dir)
        assert (
            client.post("/api/models/register", json={"name": "", "checkpoint": str(checkpoint)}).status_code
            == 400
        )
        assert (
            client.post(
                "/api/models/register", json={"name": "bad/name", "checkpoint": str(checkpoint)}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/models/register", json={"name": "x", "checkpoint": "/nope.pt"}
            ).status_code
            == 400
        )

    def test_duplicate_rejected(self, client, server_dir):
        checkpoint = self._checkpoint(server_dir)
        client.post("/api/models/register", json={"name": "dup", "checkpoint": str(checkpoint)})
        response = client.post(
            "/api/models/register", json={"name": "dup", "checkpoint": str(checkpoint)}
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]


# -- config --------------------------------------------------------------------------------


class TestConfig:
    def test_get_default_config(self, client):
        data = client.get("/api/config").json()
        assert data["yaml"]
        assert data["config"]
        assert data["problems"] == []

    def test_validate_ok(self, client):
        yaml_text = "driver:\n  kind: simulated\n  allow_simulated: true\ntrack:\n  synthetic: straight\n"
        data = client.post("/api/config/validate", json={"yaml": yaml_text}).json()
        assert data["ok"] is True
        assert data["problems"] == []

    def test_validate_reports_problems(self, client):
        data = client.post("/api/config/validate", json={"yaml": "driver:\n  kind: nope\n"}).json()
        assert data["ok"] is False
        assert data["problems"]

    def test_validate_requires_yaml(self, client):
        assert client.post("/api/config/validate", json={"yaml": ""}).status_code == 400


# -- replays & demos & benchmarks -----------------------------------------------------------


class TestReplays:
    def _run_with_replays(self, server_dir) -> str:
        import numpy as np

        from tmai.replay import EpisodeReplay, ReplayStore

        run_dir = _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_replays")
        store = ReplayStore(run_dir / "replays")
        n = 20
        store.save(
            EpisodeReplay(
                episode=1,
                step=100,
                track="straight",
                end_reason="time_limit",
                finished=False,
                race_time=1.0,
                positions=np.stack([np.zeros(n), np.zeros(n), np.arange(n, dtype=float)], axis=1),
                speeds=np.full(n, 10.0),
                actions=np.zeros((n, 3)),
                rewards=np.zeros(n),
                progress=np.arange(n, dtype=float),
                race_times=np.arange(n, dtype=float) * 0.05,
            )
        )
        return run_dir.name

    def test_list_and_detail(self, client, server_dir):
        name = self._run_with_replays(server_dir)
        listing = client.get("/api/replays", params={"run": name}).json()
        assert len(listing["replays"]) == 1
        assert listing["replays"][0]["episode"] == 1

        detail = client.get(f"/api/replays/{name}/episode_000001.json").json()
        assert detail["num_samples"] == 20
        assert detail["track"] == "straight"
        assert len(detail["positions"]) == 20

    def test_detail_missing_is_404(self, client, server_dir):
        name = self._run_with_replays(server_dir)
        assert client.get(f"/api/replays/{name}/nope.json").status_code == 404

    def test_compare(self, client, server_dir):
        import numpy as np

        from tmai.replay import EpisodeReplay

        name = self._run_with_replays(server_dir)
        # A ghost: same trajectory, half the time.
        n = 20
        ghost_path = server_dir["demos"] / "ghost.json"
        EpisodeReplay(
            episode=0,
            step=0,
            track="straight",
            finished=True,
            race_time=0.5,
            positions=np.stack([np.zeros(n), np.zeros(n), np.arange(n, dtype=float)], axis=1),
            speeds=np.full(n, 20.0),
            actions=np.zeros((n, 3)),
            rewards=np.zeros(n),
            progress=np.arange(n, dtype=float),
            race_times=np.arange(n, dtype=float) * 0.025,
            source="human",
        ).save(ghost_path)
        data = client.post(
            "/api/replays/compare",
            json={"run": name, "a": "episode_000001.json", "b": str(ghost_path)},
        ).json()
        assert data["ai_finished"] is False
        assert data["ghost_finished"] is True
        # The ghost is twice as fast, so every segment gap is positive: the AI is behind.
        assert all(g > 0 for g in data["segment_gaps"])
        assert data["mean_gap"] > 0

    def test_compare_requires_run(self, client):
        assert client.post("/api/replays/compare", json={"a": "x", "b": "y"}).status_code == 400

    def test_compare_accepts_a_demonstration_as_ghost(self, client, server_dir):
        """The ghost can be a human demonstration (JSONL), not just another replay file."""
        import numpy as np

        from tmai.training.demos import Demonstration

        name = self._run_with_replays(server_dir)
        demo_path = server_dir["demos"] / "human.jsonl"
        n = 20
        Demonstration(
            observations=np.zeros((n, 4), dtype=np.float32),
            actions=np.zeros((n, 3), dtype=np.float32),
            positions=np.stack([np.zeros(n), np.zeros(n), np.arange(n, dtype=float)], axis=1),
            speeds=np.full(n, 20.0),
            rewards=np.zeros(n),
            race_times=np.arange(n, dtype=float) * 0.025,
            metadata={"track": "straight", "finished": True},
        ).save(demo_path)

        data = client.post(
            "/api/replays/compare",
            json={"run": name, "a": "episode_000001.json", "b": str(demo_path)},
        ).json()
        assert data["ghost_finished"] is True
        assert data["ai_finished"] is False
        # The ghost is twice as fast: every segment gap is positive (the AI loses time).
        assert all(g > 0 for g in data["segment_gaps"])


class TestDemos:
    def test_list_demos(self, client, server_dir):
        _write_demo(server_dir["demos"], steps=7)
        data = client.get("/api/demos").json()
        assert len(data["demos"]) == 1
        assert data["demos"][0]["steps"] == 7

    def test_empty_demos(self, client):
        assert client.get("/api/demos").json()["demos"] == []


class TestBenchmarks:
    def test_list_benchmarks(self, client, server_dir):
        report = {
            "schema_version": 1,
            "name": "b1",
            "created_utc": "2026-10-08T00:00:00+00:00",
            "ranking": ["a", "b"],
            "splits": ["validation"],
        }
        (server_dir["benchmarks"] / "b1_20261008.json").write_text(json.dumps(report))
        data = client.get("/api/benchmarks").json()
        assert len(data["benchmarks"]) == 1
        assert data["benchmarks"][0]["ranking"] == ["a", "b"]

    def test_empty_benchmarks(self, client):
        assert client.get("/api/benchmarks").json()["benchmarks"] == []


# -- jobs ------------------------------------------------------------------------------------


class TestJobs:
    def test_list_and_detail(self, client):
        data = client.get("/api/jobs").json()
        assert data["jobs"] == []
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.post("/api/jobs/nope/cancel").status_code == 404

    def test_doctor_job_runs_and_reports(self, client):
        response = client.post("/api/doctor", json={})
        job = response.json()["job"]
        finished = _wait_for_job(client, job["id"], timeout=120)
        assert finished["state"] == "done", finished["error"]
        assert finished["result"]["exit_code"] == 0
        assert any("environment" in line for line in finished["log_tail"])

    def test_train_job_end_to_end(self, client, server_dir, tmp_path):
        """The GUI's start-training path: a real subprocess training run."""
        config = RunConfig.from_yaml("tmai/configs/smoke.yaml")
        config.train.output_dir = str(server_dir["runs"])
        config.train.run_name = "api-train"
        config.train.total_steps = 200
        config.train.warmup_steps = 50
        config.train.eval_interval = 100
        config.train.held_out_eval_interval = 0
        config.train.checkpoint_interval = 100
        config_path = tmp_path / "train.yaml"
        config.save(config_path)

        response = client.post(
            "/api/train",
            json={"config_path": str(config_path), "run_name": "api-train"},
        )
        assert response.status_code == 200
        job = response.json()["job"]
        assert job["kind"] == "train"

        finished = _wait_for_job(client, job["id"], timeout=300)
        assert finished["state"] == "done", finished["error"]
        assert finished["result"]["exit_code"] == 0
        # The job reports the run it created, and the run is real.
        run_name = finished["result"]["run"]
        assert run_name and run_name.startswith("api-train") or "api-train" in run_name
        run_dir = server_dir["runs"] / run_name
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "metrics.jsonl").is_file()
        detail = client.get(f"/api/runs/{run_name}").json()
        assert detail["status"]["step"] >= 200

    def test_train_job_with_config_yaml_text(self, client, server_dir, tmp_path):
        """The config editor posts YAML text; the server stores it and trains from it."""
        config = RunConfig.from_yaml("tmai/configs/smoke.yaml")
        config.train.output_dir = str(server_dir["runs"])
        config.train.run_name = "api-yaml"
        config.train.total_steps = 200
        config.train.warmup_steps = 50
        config.train.eval_interval = 100
        config.train.held_out_eval_interval = 0
        config.train.checkpoint_interval = 100

        response = client.post(
            "/api/train",
            json={"config_yaml": config.to_yaml(), "run_name": "api-yaml"},
        )
        job = response.json()["job"]
        finished = _wait_for_job(client, job["id"], timeout=300)
        assert finished["state"] == "done", finished["error"]
        assert finished["result"]["exit_code"] == 0

    def test_eval_job(self, client, server_dir, tmp_path):
        """Evaluate a checkpoint through the API: report JSON comes back."""

        from tmai.training.checkpoint import save_checkpoint
        from tmai.training.factory import build_learner, build_library, build_multi_track_env

        # The multitrack config has a real validation split (the single-track smoke config
        # assigns its only track to train, so "validation" would be empty there).
        config = RunConfig.from_yaml("tmai/configs/multitrack_smoke.yaml")
        config.train.output_dir = str(server_dir["runs"])
        config.env.termination.max_steps = 60
        library = build_library(config)
        env = build_multi_track_env(config, library, split="train", seed=0)
        try:
            learner = build_learner(env, config)
        finally:
            env.close()
        checkpoint = save_checkpoint(
            tmp_path / "ckpt", step=10, learner=learner, config=config.to_dict()
        )
        config_path = tmp_path / "eval.yaml"
        config.save(config_path)

        response = client.post(
            "/api/eval",
            json={
                "config_path": str(config_path),
                "checkpoint": str(checkpoint),
                "splits": ["validation"],
                "episodes": 1,
            },
        )
        job = response.json()["job"]
        finished = _wait_for_job(client, job["id"], timeout=300)
        assert finished["state"] == "done", finished["error"]
        result = finished["result"]
        assert result["splits"] == ["validation"]
        assert result["reports"][0]["num_episodes"] >= 1
        assert Path(result["path"]).is_file()

    def test_benchmark_job(self, client, server_dir, tmp_path):

        from tmai.training.checkpoint import save_checkpoint
        from tmai.training.factory import build_learner, build_library, build_multi_track_env

        config = RunConfig.from_yaml("tmai/configs/multitrack_smoke.yaml")
        config.train.output_dir = str(server_dir["runs"])
        config.env.termination.max_steps = 40
        library = build_library(config)
        env = build_multi_track_env(config, library, split="train", seed=0)
        try:
            learner = build_learner(env, config)
        finally:
            env.close()
        checkpoint = save_checkpoint(
            tmp_path / "ckpt", step=10, learner=learner, config=config.to_dict()
        )
        config_path = tmp_path / "bench.yaml"
        config.save(config_path)

        response = client.post(
            "/api/benchmark",
            json={
                "config_path": str(config_path),
                "models": [{"label": "m1", "checkpoint": str(checkpoint)}],
                "splits": ["validation"],
                "episodes": 1,
                "name": "api-bench",
            },
        )
        job = response.json()["job"]
        finished = _wait_for_job(client, job["id"], timeout=300)
        assert finished["state"] == "done", finished["error"]
        result = finished["result"]
        assert result["ranking"] == ["m1"]
        assert Path(result["path"]).is_file()
        # And the report shows up in the benchmarks listing.
        listing = client.get("/api/benchmarks").json()
        assert any(b["benchmark"] == "api-bench" for b in listing["benchmarks"])

    def test_jobs_are_listed_newest_first(self, client):
        first = client.post("/api/doctor", json={}).json()["job"]
        _wait_for_job(client, first["id"], timeout=120)
        second = client.post("/api/doctor", json={}).json()["job"]
        _wait_for_job(client, second["id"], timeout=120)
        jobs = client.get("/api/jobs").json()["jobs"]
        assert jobs[0]["id"] == second["id"]
        assert jobs[1]["id"] == first["id"]


class TestJobManager:
    """Unit-level job semantics: states, cancellation, persistence recovery."""

    def test_successful_job(self, tmp_path):
        manager = JobManager(tmp_path / "state")
        job = manager.submit("test", "does a thing", lambda j: {"answer": 42})
        finished = manager.wait(job.id, timeout=10)
        assert finished.state is JobState.DONE
        assert finished.result == {"answer": 42}

    def test_failing_job_captures_the_error(self, tmp_path):
        manager = JobManager(tmp_path / "state")

        def boom(job):
            raise RuntimeError("kaboom")

        job = manager.submit("test", "fails", boom)
        finished = manager.wait(job.id, timeout=10)
        assert finished.state is JobState.FAILED
        assert "kaboom" in finished.error
        assert any("FAILED" in line for line in finished.log_tail)

    def test_cancelled_job(self, tmp_path):
        manager = JobManager(tmp_path / "state")
        started = []

        def slow(job):
            started.append(True)
            while not manager.is_cancelled(job):
                time.sleep(0.01)
            return {"stopped": True}

        job = manager.submit("test", "slow", slow)
        while not started:
            time.sleep(0.01)
        assert manager.cancel(job.id)
        finished = manager.wait(job.id, timeout=10)
        assert finished.state is JobState.CANCELLED

    def test_interrupted_jobs_recovered_on_restart(self, tmp_path):
        state_dir = tmp_path / "state"
        manager = JobManager(state_dir)
        job = manager.submit("test", "never finishes", lambda j: time.sleep(30))
        while job.state is not JobState.RUNNING:
            time.sleep(0.01)
        # Simulate a server crash: a new manager over the same state directory.
        recovered = JobManager(state_dir)
        jobs = {j.id: j for j in recovered.list()}
        assert job.id in jobs
        assert jobs[job.id].state is JobState.INTERRUPTED
        assert "restarted" in (jobs[job.id].error or "")


# -- websocket -------------------------------------------------------------------------------


class TestWebSocket:
    def test_tick(self, client, server_dir):
        _write_run(server_dir["runs"], "2026-01-01T00-00-00Z_ws")
        with client.websocket_connect("/api/ws") as websocket:
            tick = websocket.receive_json()
        assert tick["type"] == "tick"
        assert "system" in tick
        assert "runs" in tick
        assert "jobs" in tick
        assert any(r["name"] == "2026-01-01T00-00-00Z_ws" for r in tick["runs"])
