// Vite builds the React page into frontend/dist, which server.py serves.
// During development (npm run dev) every /api call is passed to server.py on port 8000.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", emptyOutDir: true },
});
