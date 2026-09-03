import { defineConfig } from 'vite'

export default defineConfig({
  base: '/vizmlir/',
  build: {
    target: 'esnext'
  },

  worker: {
    format: 'es'
  }
})