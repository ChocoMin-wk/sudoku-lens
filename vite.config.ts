import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const repository = process.env.GITHUB_REPOSITORY?.split("/")[1];
const githubBase = repository?.endsWith(".github.io") ? "/" : `/${repository ?? "sudoku-lens"}/`;

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? githubBase : "/",
  plugins: [react()],
});
