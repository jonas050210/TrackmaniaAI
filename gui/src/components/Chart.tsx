/** Hand-rolled SVG line charts — no chart library, so the bundle stays small. */

import { useMemo, useState } from "react";
import type { Series } from "../api";

export interface ChartSeries {
  name: string;
  points: { step: number; value: number }[];
  color: string;
}

const PALETTE = [
  "#38bdf8",
  "#34d399",
  "#fbbf24",
  "#a78bfa",
  "#f87171",
  "#22d3ee",
  "#fb923c",
  "#4ade80",
];

export function colorFor(index: number): string {
  return PALETTE[index % PALETTE.length];
}

interface ChartProps {
  series: ChartSeries[];
  height?: number;
  title?: string;
  yLabel?: string;
  /** Render the points as a step line (good for counters/stages). */
  stepped?: boolean;
}

interface ViewBox {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [min];
  const span = max - min;
  const step = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(step)));
  const normalized = step / magnitude;
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  const tick = nice * magnitude;
  const ticks: number[] = [];
  for (let value = Math.ceil(min / tick) * tick; value <= max + 1e-9; value += tick) {
    ticks.push(value);
  }
  return ticks.length ? ticks : [min];
}

export function LineChart({ series, height = 180, title, yLabel, stepped }: ChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const view = useMemo<ViewBox>(() => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const s of series) {
      for (const p of s.points) {
        if (!Number.isFinite(p.value)) continue;
        minX = Math.min(minX, p.step);
        maxX = Math.max(maxX, p.step);
        minY = Math.min(minY, p.value);
        maxY = Math.max(maxY, p.value);
      }
    }
    if (!Number.isFinite(minX)) {
      minX = 0;
      maxX = 1;
    }
    if (maxX === minX) maxX = minX + 1;
    if (!Number.isFinite(minY)) {
      minY = 0;
      maxY = 1;
    }
    if (maxY === minY) {
      maxY = minY + 1;
    } else {
      // a little headroom so lines do not hug the edges
      const pad = (maxY - minY) * 0.08;
      minY -= pad;
      maxY += pad;
    }
    return { minX, maxX, minY, maxY };
  }, [series]);

  const width = 100; // viewBox units; scaled by CSS
  const padLeft = 8;
  const padRight = 8;
  const padTop = 8;
  const padBottom = 18;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  const x = (step: number) => padLeft + ((step - view.minX) / (view.maxX - view.minX)) * innerW;
  const y = (value: number) => padTop + innerH - ((value - view.minY) / (view.maxY - view.minY)) * innerH;

  const yTicks = niceTicks(view.minY, view.maxY, 4);

  const pathFor = (points: { step: number; value: number }[]) => {
    const finite = points.filter((p) => Number.isFinite(p.value));
    if (!finite.length) return "";
    if (stepped) {
      let d = `M ${x(finite[0].step)} ${y(finite[0].value)}`;
      for (let i = 1; i < finite.length; i++) {
        d += ` L ${x(finite[i].step)} ${y(finite[i - 1].value)} L ${x(finite[i].step)} ${y(finite[i].value)}`;
      }
      return d;
    }
    return finite.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.step)} ${y(p.value)}`).join(" ");
  };

  // hover crosshair: nearest step across all series
  const allSteps = useMemo(() => {
    const steps = new Set<number>();
    for (const s of series) for (const p of s.points) steps.add(p.step);
    return [...steps].sort((a, b) => a - b);
  }, [series]);
  const hoverIndex =
    hover === null ? null : Math.max(0, Math.min(allSteps.length - 1, Math.round(hover)));
  const hoverStep = hoverIndex === null ? null : allSteps[hoverIndex];

  return (
    <div className="chart">
      <div className="chart-head">
        <span className="chart-title">{title}</span>
        <div className="legend">
          {series.map((s) => (
            <span className="item" key={s.name}>
              <span className="swatch" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const fraction = (event.clientX - rect.left) / rect.width;
          const step = view.minX + fraction * (view.maxX - view.minX);
          let nearest = 0;
          let best = Infinity;
          allSteps.forEach((s, i) => {
            const d = Math.abs(s - step);
            if (d < best) {
              best = d;
              nearest = i;
            }
          });
          setHover(nearest);
        }}
        onMouseLeave={() => setHover(null)}
      >
        {/* gridlines + y labels */}
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={padLeft}
              x2={width - padRight}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--border)"
              strokeWidth={0.5}
              strokeDasharray="3 3"
            />
            <text
              x={padLeft - 3}
              y={y(tick) + 2.5}
              fill="var(--text-faint)"
              fontSize={6}
              textAnchor="end"
            >
              {tick >= 1000 ? tick.toFixed(0) : tick.toFixed(tick < 10 ? 2 : 1)}
            </text>
          </g>
        ))}
        {/* x labels: first and last step */}
        <text x={padLeft} y={height - 5} fill="var(--text-faint)" fontSize={6}>
          {Math.round(view.minX)}
        </text>
        <text x={width - padRight} y={height - 5} fill="var(--text-faint)" fontSize={6} textAnchor="end">
          {Math.round(view.maxX)}
        </text>
        {yLabel && (
          <text x={padLeft - 3} y={padTop - 2} fill="var(--text-faint)" fontSize={5.5}>
            {yLabel}
          </text>
        )}
        {/* series */}
        {series.map((s) => (
          <path
            key={s.name}
            d={pathFor(s.points)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.4}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {/* hover crosshair */}
        {hoverStep !== null && (
          <line
            x1={x(hoverStep)}
            x2={x(hoverStep)}
            y1={padTop}
            y2={height - padBottom}
            stroke="var(--text-faint)"
            strokeWidth={0.5}
            strokeDasharray="2 2"
          />
        )}
      </svg>
      {hoverStep !== null && (
        <div className="legend" style={{ marginTop: 6 }}>
          <span className="item">
            <span className="mono faint">step {Math.round(hoverStep)}</span>
          </span>
          {series.map((s) => {
            const point = s.points.reduce<{ step: number; value: number } | null>((best, p) => {
              if (!best || Math.abs(p.step - hoverStep) < Math.abs(best.step - hoverStep)) return p;
              return best;
            }, null);
            if (!point) return null;
            return (
              <span className="item" key={s.name}>
                <span className="swatch" style={{ background: s.color }} />
                <span className="mono">
                  {s.name} {point.value.toFixed(3)}
                </span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Convert API series (filtered by name list) into chart series. */
export function toChartSeries(
  history: { series: Series[] } | null,
  names: string[],
  colors?: string[]
): ChartSeries[] {
  if (!history) return [];
  return names
    .map((name, index) => {
      const found = history.series.find((s) => s.name === name);
      if (!found || !found.steps.length) return null;
      return {
        name,
        points: found.steps.map((step, i) => ({ step, value: found.values[i] })),
        color: colors?.[index] ?? colorFor(index),
      };
    })
    .filter((s): s is ChartSeries => s !== null);
}
