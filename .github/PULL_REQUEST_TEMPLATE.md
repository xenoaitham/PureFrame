<!--
Thanks for sending a pull request! Please fill out the sections below to make review faster.
For small fixes (typos, docs), a one-line summary is fine.
-->

## Summary

<!-- What does this PR do? One or two sentences. -->

## Related Issues

<!-- Link issues with "Fixes #123" / "Closes #456" so they auto-close on merge. -->

Fixes #

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Refactoring (no functional change)
- [ ] Documentation update
- [ ] CI / build / dev-tooling change

## Test Plan

<!-- How did you verify this works? Commands run, scenarios exercised, screenshots for UI. -->

- [ ] `pytest -q -m "not slow"` passes locally
- [ ] `ruff check pureframe tests` clean
- [ ] `ruff format --check pureframe tests` clean
- [ ] If GUI changes: `cd gui && npm run typecheck` and `npm run build` pass
- [ ] If Rust changes: `cd gui/src-tauri && cargo clippy -- -D warnings` clean
- [ ] Manual verification: <!-- describe scenario -->

## Backwards Compatibility

<!-- Does this change the CLI surface, config schema, plan format, or Tauri IPC contract? -->

- [ ] No user-visible API change
- [ ] CLI flags / config keys deprecated rather than removed
- [ ] `pureframe_version` / `plan_version` bumped if plan schema changed

## Checklist

- [ ] My changes follow the project's code style (see `CONTRIBUTING.md`).
- [ ] I added tests that prove my fix is effective or my feature works.
- [ ] Existing tests still pass.
- [ ] I updated relevant docs (`README.md`, `docs/`, `CHANGELOG.md`).
- [ ] No secrets, tokens, or credentials are committed.
- [ ] No `print()` / `console.log()` / debug statements left in production code.
