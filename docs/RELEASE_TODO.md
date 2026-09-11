# Release Roadmap - Known Limitations

Tracks the gaps that remain in the shipped binaries. Each item lists the
blocker so future maintainers can pick up where this stopped.

## Shipped now

- All 7 Tauri/PyInstaller jobs run on tag push.
- Versions stay in sync across `pyproject.toml`, `gui/package.json`,
  `gui/src-tauri/tauri.conf.json`, and `gui/src-tauri/Cargo.toml` (CI fails
  on drift via `packaging/sync_versions.py --check`).
- Every release gets a `SHA256SUMS.txt` aggregating sums for all uploaded
  assets, generated after the matrix completes.
- GUI ships through Tauri installers; the CLI ships through PyInstaller
  zips/tarballs.
- Code-signing limitations and Gatekeeper/SmartScreen workarounds are
  documented in `README.md`.

## Deferred - Big Restructure

### Tauri sidecar (bundle the backend in the GUI installer)

**Status:** Not done. The Tauri installer is a 3 MB UI shell that calls
`Command::new("pureframe")` in `gui/src-tauri/src/lib.rs`. If a user installs
the Tauri MSI/DMG without also installing the CLI, the buttons silently
fail.

**Fix outline:**

1. In `tauri.conf.json`, add the PyInstaller dist as `bundle.externalBin`,
   namespaced by target triple (e.g., `pureframe-x86_64-pc-windows-msvc.exe`).
2. Reorder `release.yml` so `pyinstaller-build` finishes first, then the
   `tauri-build` matrix downloads the matching artifact, renames it for the
   triple, and points Tauri at it.
3. In `lib.rs`, swap `Command::new("pureframe")` for the Tauri shell
   sidecar API so the spawned binary is the bundled one, not a system
   `pureframe` on PATH.
4. Expect installer size to grow to ~400 MB on Windows.

This is one PR's worth of work and should be done in isolation.

## Deferred - Costs Money

### Code signing

- Windows: $200–500/yr for an OV/EV cert removes the "Windows protected
  your PC" SmartScreen banner.
- macOS: $99/yr Apple Developer Program enables `codesign --options
  runtime` plus notarization, which removes the Gatekeeper "unidentified
  developer" block.
- Once certs exist, populate the matching `tauri-action` env vars in
  `release.yml`; the secret-presence guard already in place means no
  workflow change is needed beyond adding the secrets.

### Auto-updater

Tauri's updater needs a private signing key in `TAURI_SIGNING_PRIVATE_KEY`
and the public counterpart embedded in `tauri.conf.json`. We have neither
generated nor hosted an update endpoint. Adding this is straightforward
once signing exists.

## Deferred - Smaller Polish

### E2E test for the Tauri ↔ backend wiring

**Partially landed (#21).** A Playwright smoke harness now lives in
`gui/e2e/` and runs in CI on every push/PR. It uses a browser-side
`window.__TAURI_INTERNALS__` shim (`gui/e2e/tauri-shim.ts`) so the React
tree boots without the Rust runtime, and asserts:

- onboarding heading renders
- optional accept-button flow
- no uncaught console errors on first paint

**Still open:** a real Tauri ↔ backend E2E that proves `start_job`
actually spawns the `pureframe` process. That requires either
`tauri-driver` (WebDriver against the packaged app) or running the
bundled binary as a sidecar (see "Tauri sidecar" below). The current
shim covers UI regressions only - it does **not** validate IPC wiring.

### ESLint configuration

The `gui/package.json` `lint` script exists but there is no flat config
checked in (the config-protection hook blocks adding one). CI runs
`typecheck` only. Developers running `npm run lint` locally need to
provide their own `eslint.config.js`.

### Offline-installer variant with bundled models

PureFrame downloads ~500 MB of model weights on first run. A "fat"
release variant that bundles the model cache directory would let air-gapped
users avoid the network step. Build it as a separate workflow that runs
`pureframe download-models`, packages the cache, and uploads it as an
additional asset.

### Anti-virus false positives on Windows

The PyInstaller + PyTorch combo gets flagged by some Defender heuristics.
We do not use UPX (which would be worse), but the only reliable fix is
code signing (see above). Submitting the binary to Microsoft Defender for
analysis can also help.

### Intel mac PyInstaller binary

GitHub-hosted `macos-13` runners are EOL'd and queue jobs indefinitely.
Intel mac users currently fall back to `pip install pureframe` or the
Tauri x64 DMG. If GitHub revives the runners, restore the matrix entry
in `release.yml` (the spot is commented).
