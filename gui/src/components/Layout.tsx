/** App shell: glass navigation, responsive workspace chrome, and live job status. */

import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { api, connectWebSocket, type Tick } from "../api";
import { ToastProvider } from "./ui";

const NAV = [
  { to: "/", label: "Overview", short: "Home", icon: "◧", end: true },
  { to: "/runs", label: "Runs", short: "Runs", icon: "▤" },
  { to: "/training", label: "Training", short: "Train", icon: "▶" },
  { to: "/evaluate", label: "Evaluate & Benchmark", short: "Evaluate", icon: "◎" },
  { to: "/models", label: "Models", short: "Models", icon: "◆" },
  { to: "/tracks", label: "Tracks", short: "Tracks", icon: "⬡" },
  { to: "/replays", label: "Replays & Ghosts", short: "Replays", icon: "◉" },
  { to: "/config", label: "Configuration", short: "Config", icon: "⚙" },
  { to: "/diagnostics", label: "Diagnostics", short: "Health", icon: "♥" },
];

function Shell() {
  const location = useLocation();
  const [version, setVersion] = useState<string>("");
  const [live, setLive] = useState(false);
  const [lastTick, setLastTick] = useState<Tick | null>(null);
  const lastTickAt = useRef<number | null>(null);

  useEffect(() => {
    let mounted = true;
    api.health().then((health) => {
      if (mounted) setVersion(health.version);
    }).catch(() => {
      if (mounted) setVersion("");
    });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    lastTickAt.current = null;
    const dispose = connectWebSocket((tick) => {
      lastTickAt.current = Date.now();
      setLastTick(tick);
      setLive(true);
    });
    const watchdog = setInterval(() => {
      const last = lastTickAt.current;
      setLive(last !== null && Date.now() - last < 10_000);
    }, 2_500);
    return () => {
      dispose();
      clearInterval(watchdog);
    };
  }, []);

  const runningJobs = lastTick?.jobs.filter(
    (job) => job.state === "running" || job.state === "queued"
  ).length ?? 0;
  const currentPage = location.pathname.startsWith("/runs/")
    ? "Run detail"
    : NAV.find((item) => item.to === "/"
      ? location.pathname === "/"
      : location.pathname.startsWith(item.to))?.label ?? "Command Center";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo" aria-hidden="true">T</div>
          <div>
            <div className="name">TrackmaniaAI</div>
            <div className="version">v{version || "…"}</div>
          </div>
        </div>
        <nav aria-label="Primary navigation">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
            >
              <span className="icon" aria-hidden="true">{item.icon}</span>
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
          <div className="topbar-heading">
            <span className="topbar-kicker">TRACKMANIAAI / COMMAND CENTER</span>
            <span className="title">{currentPage}</span>
          </div>
          <span className="spacer" />
          {runningJobs > 0 && <span className="badge blue">{runningJobs} active</span>}
          <span className={`live${live ? " is-live" : " is-stale"}`} role="status" aria-live="polite">
            <span className={`dot${live ? "" : " stale"}`} />
            {live ? "Live" : "Reconnecting"}
          </span>
        </header>
        <main className="content fade-in" key={location.pathname}>
          <Outlet context={{ tick: lastTick }} />
        </main>
        <nav className="mobile-nav" aria-label="Primary navigation">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              title={item.label}
              aria-label={item.label}
              className={({ isActive }) => `mobile-nav-item${isActive ? " active" : ""}`}
            >
              <span className="icon" aria-hidden="true">{item.icon}</span>
              <span>{item.short}</span>
            </NavLink>
          ))}
        </nav>
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
