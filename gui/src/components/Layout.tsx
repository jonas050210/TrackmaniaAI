/** App shell: sidebar navigation, top bar with live indicator, routed content. */

import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { api, connectWebSocket, type Tick } from "../api";
import { ToastProvider } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: "◧", end: true },
  { to: "/runs", label: "Runs", icon: "▤" },
  { to: "/training", label: "Training", icon: "▶" },
  { to: "/evaluate", label: "Evaluate & Benchmark", icon: "◎" },
  { to: "/models", label: "Models", icon: "◆" },
  { to: "/tracks", label: "Tracks", icon: "⬡" },
  { to: "/replays", label: "Replays & Ghosts", icon: "◉" },
  { to: "/config", label: "Configuration", icon: "⚙" },
  { to: "/diagnostics", label: "Diagnostics", icon: "♥" },
];

function Shell() {
  const location = useLocation();
  const [version, setVersion] = useState<string>("");
  const [live, setLive] = useState(false);
  const [lastTick, setLastTick] = useState<Tick | null>(null);

  useEffect(() => {
    api.health().then((h) => setVersion(h.version)).catch(() => setVersion(""));
  }, []);

  useEffect(() => {
    setLive(true);
    const dispose = connectWebSocket((tick) => {
      setLastTick(tick);
      setLive(true);
    });
    const watchdog = setInterval(() => {
      // No tick for 10s means the socket is gone.
      setLive((current) => current);
    }, 10_000);
    return () => {
      dispose();
      clearInterval(watchdog);
    };
  }, []);

  const runningJobs = lastTick?.jobs.filter((j) => j.state === "running" || j.state === "queued").length ?? 0;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">T</div>
          <div>
            <div className="name">TrackmaniaAI</div>
            <div className="version">v{version || "…"}</div>
          </div>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
            >
              <span className="icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="footer">
          <span>{runningJobs > 0 ? `${runningJobs} job${runningJobs > 1 ? "s" : ""} running` : "no active jobs"}</span>
          <span className="mono">{location.pathname}</span>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <span className="title">Command Center</span>
          <span className="spacer" />
          {runningJobs > 0 && <span className="badge blue">{runningJobs} active</span>}
          <span className={`live`}>
            <span className={`dot${live ? "" : " stale"}`} />
            {live ? "live" : "reconnecting…"}
          </span>
        </header>
        <main className="content fade-in">
          <Outlet context={{ tick: lastTick }} />
        </main>
      </div>
    </div>
  );
}

export function Layout() {
  return (
    <ToastProvider>
      <Shell />
    </ToastProvider>
  );
}
