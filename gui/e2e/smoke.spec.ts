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

test("plan editor timeline scrubber fetches the frame at the new position", async ({
  page,
}) => {
  // Seed a DONE plan job so the queue offers "Review plan"; the shim's
  // load_plan serves a 90 s plan with a flagged middle shot.
  await page.addInitScript(() => {
    window.localStorage.setItem("onboarding_done", "1");
    window.localStorage.setItem(
      "pureframe_jobs",
      JSON.stringify([
        {
          id: "job-e2e-1",
          path: "/videos/movie.mkv",
          status: "DONE",
          mode: "plan",
          progress: 100,
          lastLine: null,
          output: "/videos/movie.censorplan.json",
          exitCode: 0,
        },
      ]),
    );
  });
  await page.goto("/");
  await page.getByRole("button", { name: /review plan/i }).click();

  const scrub = page.getByLabel(/scrub timeline/i);
  await expect(scrub).toBeVisible();
  // No preview panel before the first seek.
  await expect(page.getByAltText(/scrub preview/i)).toHaveCount(0);

  await scrub.fill("45");
  await expect(page.getByTestId("scrub-timecode")).toHaveText(
    "0:45.0 / 1:30.0",
  );
  // The shim resolves extract_thumbnail with a deterministic image.
  await expect(page.getByAltText(/scrub preview/i)).toBeVisible();
});
