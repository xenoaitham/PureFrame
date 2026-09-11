import { defineConfig, devices } from "@playwright/test";

/**
 * Smoke E2E for the Tauri webview UI.
 *
 * The Tauri runtime is NOT present in this environment - tests inject a
 * minimal `window.__TAURI_INTERNALS__` shim before app scripts run so
 * `invoke(...)` resolves without a real backend. This catches regressions
 * in render, routing, and the React mount lifecycle without requiring
 * the full Rust shell.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: "http://localhost:4173",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    // `vite preview` serves the production build - closer to the bundle
    // that actually ships inside the Tauri webview than `vite dev`.
    command: "npm run build && npx vite preview --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
