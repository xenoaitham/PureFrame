/**
 * Browser-side shim for `window.__TAURI_INTERNALS__`.
 *
 * `@tauri-apps/api`'s `invoke(...)` reads this object to dispatch IPC
 * calls. Outside the Tauri webview it is undefined, which crashes the
 * app at first render. The shim returns sensible empty responses so the
 * UI can boot for smoke tests.
 *
 * Add new command shims here as the surface grows - keep responses
 * minimal and deterministic.
 */
export const tauriShimScript = `
  (() => {
    const responses = {
      // start_job -> returns a fake job id
      start_job: () => "00000000-0000-0000-0000-000000000000",
      cancel_job: () => null,
      // job_status -> deterministic running state with one progress line
      job_status: () => ({
        state: "running",
        exit_code: null,
        mode: "process",
        output: null,
        log_tail: ["Densifying shot 3 - 42%|██████ | 1337/3180"],
      }),
      // load_plan -> empty plan JSON
      load_plan: () => JSON.stringify({
        pureframe_version: "0.0.0-e2e",
        plan_version: 1,
        input_metadata: { duration_seconds: 0 },
        config_snapshot: {},
        shots: [],
        verdicts: [],
        total_censored_frames: 0,
      }),
      save_plan: () => null,
      // base64 of a 1x1 transparent png
      extract_thumbnail: () =>
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII=",
    };

    window.__TAURI_INTERNALS__ = {
      transformCallback: (cb) => cb,
      invoke: (cmd, args) => {
        const handler = responses[cmd];
        if (handler) {
          return Promise.resolve(handler(args));
        }
        // Tauri plugin commands ("plugin:event|listen", "plugin:webview|...")
        // are called during boot for window/event wiring. Resolve to a
        // no-op so the React tree mounts cleanly instead of surfacing
        // the rejection as a console error.
        if (cmd.startsWith("plugin:")) {
          return Promise.resolve(0);
        }
        // Unknown app-level command - log but don't reject, so a single
        // missing shim entry doesn't cascade into render failure.
        console.warn("[e2e shim] unhandled command:", cmd);
        return Promise.resolve(null);
      },
      ipc: {
        postMessage: () => undefined,
      },
      metadata: {
        currentWindow: { label: "main" },
        currentWebview: { label: "main", windowLabel: "main" },
      },
      plugins: {},
    };
  })();
`;
