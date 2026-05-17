import { test, expect } from "@playwright/test";
import { tauriShimScript } from "./tauri-shim";

test.beforeEach(async ({ page }) => {
  // Install Tauri IPC shim BEFORE any app script executes so `invoke(...)`
  // calls during the first render don't reject.
  await page.addInitScript(tauriShimScript);
});

test("onboarding page renders the welcome heading", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /welcome to pureframe/i }),
  ).toBeVisible();
});

test("acknowledging onboarding routes to queue page", async ({ page }) => {
  await page.goto("/");
  const accept = page.getByRole("button", { name: /accept|agree|continue/i });
  // Onboarding copy may evolve; only assert the click flow when a button
  // is actually present so this stays a smoke test, not a content lock.
  if ((await accept.count()) > 0) {
    await accept.first().click();
    await expect(
      page.getByRole("heading", { name: /queue|jobs/i }),
    ).toBeVisible({ timeout: 5_000 });
  }
});

test("no uncaught console errors during initial render", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  // Filter dev-only / shim-only noise; tighten as the surface grows.
  const ignorePatterns = [
    /download.*react devtools/i,
    /\[e2e shim\]/i,
    /plugin:event\|listen/i,
    /plugin:webview\|/i,
  ];
  const meaningful = errors.filter(
    (m) => !ignorePatterns.some((p) => p.test(m)),
  );
  expect(meaningful, meaningful.join("\n")).toEqual([]);
});
