"""Background job management for the GUI backend.

Training, evaluation, benchmarking and calibration are long-running operations. The GUI
starts them as *jobs*: a thread running a callable, with a state machine
(``queued -> running -> done | failed | cancelled``), a captured log, and a JSON result.

Jobs are persisted to ``jobs.jsonl`` so a server restart can show what happened to a job
that was running when the server went away (it reports ``interrupted`` rather than
pretending it finished).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_FILE_NAME = "jobs.jsonl"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


#: What a job callable receives and returns.
JobFn = Callable[["Job"], "dict[str, Any] | None"]


@dataclass
class Job:
    """One background job."""

    id: str
    kind: str
    description: str = ""
    state: JobState = JobState.QUEUED
    created_utc: str = ""
    started_utc: str | None = None
    finished_utc: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    fn: JobFn | None = field(default=None, repr=False, compare=False)
    _log_limit: int = 2000

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()

    # -- log -------------------------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        self.log_lines.append(line)
        if len(self.log_lines) > self._log_limit:
            del self.log_lines[: len(self.log_lines) - self._log_limit]
        logger.info("job %s (%s): %s", self.id[:8], self.kind, message)

    @property
    def log_tail(self) -> list[str]:
        return list(self.log_lines)

    # -- serialisation ----------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "state": self.state.value,
            "created_utc": self.created_utc,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "result": self.result,
            "error": self.error,
            "log_tail": self.log_lines[-200:],
            "params": self.params,
        }


class JobManager:
    """Runs callables in background threads and tracks them."""

    def __init__(self, state_dir: str | Path | None = None, *, workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._state_dir = Path(state_dir) if state_dir else None
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._workers = [
            threading.Thread(target=self._worker, daemon=True, name=f"tmai-job-{i}")
            for i in range(max(1, workers))
        ]
        for worker in self._workers:
            worker.start()
        self._recover_interrupted()

    # -- public API -------------------------------------------------------------------

    def submit(
        self,
        kind: str,
        description: str,
        fn: JobFn,
        *,
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Queue a job; returns it immediately in the ``queued`` state."""
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            description=description,
            params=dict(params or {}),
            fn=fn,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._cancel[job.id] = threading.Event()
        self._persist(job)
        self._queue.put(job)
        logger.info("queued job %s (%s): %s", job.id[:8], kind, description)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, kind: str | None = None, limit: int = 100) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_utc, reverse=True)
        return jobs[:limit]

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. Returns False when the job is unknown or already finished."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state not in (JobState.QUEUED, JobState.RUNNING):
                return False
            queued = job.state is JobState.QUEUED
            job.state = JobState.CANCELLED if queued else JobState.CANCELLING
            event = self._cancel.get(job_id)
            if event is not None:
                event.set()
            if queued:
                job.finished_utc = datetime.now(timezone.utc).isoformat()
        job.log("cancelled before start" if queued else "cancellation requested by operator")
        self._persist(job)
        return True

    def is_cancelled(self, job: Job) -> bool:
        event = self._cancel.get(job.id)
        return bool(event is not None and event.is_set())

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        """Block until the job reaches a terminal state (or the timeout expires)."""
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job is None:
                return None
            if job.state in (
                JobState.DONE,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.INTERRUPTED,
            ):
                return job
            if deadline is not None and time.monotonic() >= deadline:
                return job
            time.sleep(0.01)

    # -- internals ----------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:  # shutdown sentinel
                return
            with self._lock:
                if job.state is JobState.CANCELLED:
                    continue
                job.state = JobState.RUNNING
                job.started_utc = datetime.now(timezone.utc).isoformat()
                self._threads[job.id] = threading.current_thread()
            self._persist(job)
            job.log(f"started: {job.description}")
            try:
                result = job.fn(job) if job.fn is not None else None
            except Exception as exc:  # noqa: BLE001 - a job's failure is data, not a crash
                if self.is_cancelled(job):
                    job.state = JobState.CANCELLED
                    job.log("stopped after cancellation request")
                else:
                    job.state = JobState.FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.log(f"FAILED: {job.error}")
                    job.log(traceback.format_exc(limit=5))
                    logger.exception("job %s failed", job.id[:8])
            else:
                if self.is_cancelled(job):
                    job.state = JobState.CANCELLED
                    job.log("stopped after cancellation request")
                else:
                    job.state = JobState.DONE
                    job.result = result
                    job.log("done")
            finally:
                job.finished_utc = datetime.now(timezone.utc).isoformat()
                self._persist(job)

    # -- persistence --------------------------------------------------------------------

    @property
    def _jobs_file(self) -> Path | None:
        return self._state_dir / JOBS_FILE_NAME if self._state_dir else None

    def _persist(self, job: Job) -> None:
        path = self._jobs_file
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(job.as_dict(), default=str) + "\n")
        except OSError:  # pragma: no cover - persistence must never break a job
            logger.warning("could not persist job %s", job.id, exc_info=True)

    def _recover_interrupted(self) -> None:
        """Jobs left ``running`` by a previous server process are marked interrupted."""
        path = self._jobs_file
        if path is None or not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        latest: dict[str, dict[str, Any]] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[record.get("id", "")] = record
        for job_id, record in latest.items():
            if record.get("state") in (JobState.RUNNING.value, JobState.CANCELLING.value, JobState.QUEUED.value):
                job = Job(
                    id=job_id,
                    kind=str(record.get("kind", "unknown")),
                    description=str(record.get("description", "")),
                    state=JobState.INTERRUPTED,
                    created_utc=str(record.get("created_utc", "")),
                    started_utc=record.get("started_utc"),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    error="the server was restarted while this job was running",
                    log_lines=list(record.get("log_tail") or []),
                    params=dict(record.get("params") or {}),
                )
                with self._lock:
                    self._jobs[job.id] = job
                self._persist(job)
                logger.warning("job %s was interrupted by a server restart", job_id[:8])


__all__ = ["Job", "JobManager", "JobState"]
