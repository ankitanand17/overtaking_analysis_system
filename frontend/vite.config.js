import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Vite config file to enable React fast refresh plugin support
export default defineConfig({
  plugins: [react()],
})
