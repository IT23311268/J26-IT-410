import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands inside the Python package and FastAPI serves it from
// there, so the demo machine needs no Node and no network: `npm run
// build` happens on a development machine and the output is committed.
//
// `base` has to match the mount path or every asset 404s — the bundle
// asks for /assets/… by default, and the app is served at /app/.
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: {
    outDir: "../api/static/app",
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` talks to a uvicorn already running on 8000, so the
    // React app and the real API are the same origin during development
    // and no CORS rule is needed in main.py.
    proxy: {
      "/paper": "http://127.0.0.1:8000",
      "/papers": "http://127.0.0.1:8000",
      "/ingest": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
