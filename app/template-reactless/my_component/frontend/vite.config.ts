import { defineConfig } from "vite";
export default defineConfig({
  base: "./",                // ESSENCIAL pra servir dentro do Streamlit
  server: { port: 5173, open: false },
  build: { outDir: "dist", emptyOutDir: true }
});
