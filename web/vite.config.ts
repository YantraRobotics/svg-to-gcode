import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/image': 'http://127.0.0.1:8000',
      '/gcode': 'http://127.0.0.1:8000',
      '/convert': 'http://127.0.0.1:8000',
      '/draw': 'http://127.0.0.1:8000',
      '/teach': 'http://127.0.0.1:8000',
    },
  },
})
