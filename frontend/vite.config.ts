import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const proxyTarget = process.env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": {
        target: proxyTarget,
        changeOrigin: true,
      },
      "/chat": {
        target: proxyTarget,
        changeOrigin: true,
      },
      "/sessions": {
        target: proxyTarget,
        changeOrigin: true,
      },
    },
  },
});
