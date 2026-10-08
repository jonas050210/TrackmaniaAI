"""Local backend for the TrackmaniaAI GUI.

A small FastAPI application that exposes everything a command-center UI needs -- runs,
metrics, tracks, models, replays, benchmarks, configuration, calibration, diagnostics and
training/evaluation *jobs* -- as plain JSON over HTTP plus a WebSocket for live updates,
and serves the built web frontend as static files.

The backend is deliberately thin: every piece of data comes from the same modules the CLI
uses (``tmai.api.status``, ``tmai.registry``, ``tmai.replay``, ...), so the GUI can never
disagree with the command line. Long-running work (training, evaluation, benchmarking,
calibration) runs as a managed background job so the UI stays responsive and a job's log
survives a page reload.
"""

__all__ = ["jobs", "app"]
