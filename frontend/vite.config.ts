import { defineConfig } from 'vite'; // cache buster 1
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    host: '127.0.0.1',
    watch: {
      usePolling: true,
    }
  },
});
