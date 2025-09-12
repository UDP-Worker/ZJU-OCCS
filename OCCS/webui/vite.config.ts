import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Basic Vite config to support dev proxy (HTTP + WS)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
