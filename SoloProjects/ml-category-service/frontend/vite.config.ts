import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Vite + React + Tailwind v4 (@tailwindcss/vite)
export default defineConfig({
  plugins: [react(), tailwindcss()],
})
