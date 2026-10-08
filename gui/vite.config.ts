import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server binds to all interfaces so the sandbox preview proxy can reach it.
// API calls are proxied to the local backend (`tmai serve`, default port 8765), so the
// browser never talks to a different origin.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.TMAI_API ?? "http://127.0.0.1:8765",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
