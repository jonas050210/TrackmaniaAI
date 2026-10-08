/** Small shared UI primitives: badges, spinners, empty states, toasts, formatting. */

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

// -- formatting helpers -----------------------------------------------------------

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(digits > 1 ? 1 : digits);
  return value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = "B";
  for (const next of units) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value.toFixed(1)} ${unit}`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const seconds = (Date.now() - date.getTime()) / 1000;
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

// -- badge ------------------------------------------------------------------------

type BadgeTone = "green" | "amber" | "red" | "blue" | "violet" | "neutral";

export function Badge({
  tone = "neutral",
  children,
  title,
}: {
  tone?: BadgeTone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {children}
    </span>
  );
}

/** A coloured dot + label, for live/running indicators. */
export function StatusDot({ on, label }: { on: boolean; label?: string }) {
  return (
    <span className="live">
      <span className={`dot${on ? "" : " stale"}`} />
      {label ?? (on ? "live" : "stale")}
    </span>
  );
}

// -- spinner / empty ----------------------------------------------------------------

export function Spinner({ size = 16 }: { size?: number }) {
  return <span className="spinner" style={{ width: size, height: size }} />;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="empty">
      <Spinner size={20} /> <span className="dim">{label}</span>
    </div>
  );
}

export function Empty({ icon = "∅", children }: { icon?: string; children: ReactNode }) {
  return (
    <div className="empty">
      <div className="big">{icon}</div>
      <div>{children}</div>
    </div>
  );
}

export function ErrorBox({ children }: { children: ReactNode }) {
  return <div className="error-box">{children}</div>;
}

export function OkBox({ children }: { children: ReactNode }) {
  return <div className="ok-box">{children}</div>;
}

// -- toasts -------------------------------------------------------------------------

interface Toast {
  id: number;
  kind: "info" | "success" | "error";
  message: string;
}

interface ToastContextValue {
  toast: (message: string, kind?: "info" | "success" | "error") => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toast = useCallback((message: string, kind: Toast["kind"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, kind, message }]);
    setTimeout(() => {
      setToasts((current) => current.filter((t) => t.id !== id));
    }, 5000);
  }, []);
  const value = useMemo(() => ({ toast }), [toast]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-stack">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind === "error" ? "error" : t.kind === "success" ? "success" : ""}`}>
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used inside ToastProvider");
  return context;
}

// -- job state badge ------------------------------------------------------------------

import type { Job } from "../api";

const JOB_TONES: Record<Job["state"], BadgeTone> = {
  queued: "neutral",
  running: "blue",
  cancelling: "amber",
  done: "green",
  failed: "red",
  cancelled: "amber",
  interrupted: "amber",
};

export function JobBadge({ state }: { state: Job["state"] }) {
  return <Badge tone={JOB_TONES[state]}>{state}</Badge>;
}
