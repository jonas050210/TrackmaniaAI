import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Overview } from "./pages/Overview";
import { Runs } from "./pages/Runs";
import { RunDetail } from "./pages/RunDetail";
import { Training } from "./pages/Training";
import { Evaluate } from "./pages/Evaluate";
import { Models } from "./pages/Models";
import { Tracks } from "./pages/Tracks";
import { Replays } from "./pages/Replays";
import { ConfigPage } from "./pages/ConfigPage";
import { Diagnostics } from "./pages/Diagnostics";

export default function App() {
  return (
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
  );
}
