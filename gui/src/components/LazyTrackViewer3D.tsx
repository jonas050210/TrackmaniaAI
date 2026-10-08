import { lazy, Suspense } from "react";
import type { Props as TrackViewerProps } from "./TrackViewer3D";

const TrackViewer = lazy(() =>
  import("./TrackViewer3D").then((module) => ({ default: module.TrackViewer3D }))
);

export type { Trajectory } from "./TrackViewer3D";

export function LazyTrackViewer3D({ height = 520, ...props }: TrackViewerProps) {
  return (
    <Suspense
      fallback={
        <div
          role="status"
          style={{
            height,
            display: "grid",
            placeItems: "center",
            borderRadius: 8,
            background: "#080b10",
            color: "#9aa8b8",
          }}
        >
          Loading 3D viewer…
        </div>
      }
    >
      <TrackViewer {...props} height={height} />
    </Suspense>
  );
}
