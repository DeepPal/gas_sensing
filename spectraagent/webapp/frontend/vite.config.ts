import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Serve the SPA under /app/ so FastAPI's /docs and /openapi.json are reachable
  base: '/app/',
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    clearMocks: true,
  },
  build: {
    // Output directly into the directory the FastAPI server serves
    outDir: '../static/dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('recharts')) return 'charts'
          if (id.includes('react-markdown')) return 'markdown'
          if (id.includes('lucide-react')) return 'icons'
          if (id.includes('react')) return 'react-vendor'
          return 'vendor'
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8765',
      '/ws': { target: 'ws://localhost:8765', ws: true },
    },
  },
})
