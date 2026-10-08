import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Loading } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview").then((module) => ({ default: module.Overview })));
const Runs = lazy(() => import("./pages/Runs").then((module) => ({ default: module.Runs })));
const RunDetail = lazy(() => import("./pages/RunDetail").then((module) => ({ default: module.RunDetail })));
const Training = lazy(() => import("./pages/Training").then((module) => ({ default: module.Training })));
const Evaluate = lazy(() => import("./pages/Evaluate").then((module) => ({ default: module.Evaluate })));
const Models = lazy(() => import("./pages/Models").then((module) => ({ default: module.Models })));
const Tracks = lazy(() => import("./pages/Tracks").then((module) => ({ default: module.Tracks })));
const Replays = lazy(() => import("./pages/Replays").then((module) => ({ default: module.Replays })));
const ConfigPage = lazy(() => import("./pages/ConfigPage").then((module) => ({ default: module.ConfigPage })));
const Diagnostics = lazy(() => import("./pages/Diagnostics").then((module) => ({ default: module.Diagnostics })));

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Overview />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:name" element={<RunDetail />} />
          <Route path="/training" element={<Training />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/models" element={<Models />} />
          <Route path="/tracks" element={<Tracks />} />
          <Route path="/replays" element={<Replays />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/diagnostics" element={<Diagnostics />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
