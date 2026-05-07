# Privacy Policy

**Last updated:** 2026-05-07

## Summary

PureFrame processes all video content **locally on your machine**. No frames, audio samples, detection results, or any other data are transmitted to any server, cloud service, or third party. Ever.

---

## Data Processing

### What PureFrame Processes

- Video frames (decoded from your local files)
- Audio samples (for moaning detection, if enabled)
- Video metadata (resolution, duration, codec, frame rate)

### Where Processing Happens

**100% local.** All detection, analysis, and rendering happens on your CPU/GPU. No cloud APIs, no remote inference, no server-side processing.

### What PureFrame Stores

| Data | Location | Purpose | Deletable |
|------|----------|---------|-----------|
| Censor plans (`.censorplan.json`) | Same directory as input video | Detection results for review | Yes, delete the file |
| Job database (`jobs.db`) | `~/.local/share/PureFrame/` (Linux) / `~/Library/Application Support/PureFrame/` (macOS) / `%APPDATA%\PureFrame\` (Windows) | Resume interrupted processing | Yes, run `pureframe jobs cleanup --all` |
| ML models | System cache directory | Inference | Yes, see [Model Deletion Guide](docs/installation.md#deleting-models) |
| Output videos | User-specified path | Censored output | Yes, delete the file |

### What PureFrame Does NOT Store

- No usage logs or analytics
- No crash reports
- No user profiles or accounts
- No video content beyond your specified output files
- No thumbnails or frame captures (except during active processing in memory)

---

## Network Activity

### During Installation

- `pip install pureframe` downloads packages from PyPI over HTTPS
- Standard Python package installation behavior

### During First Run

- NudeNet model (~120 MB) downloaded from PyPI assets
- CLIP model (~350 MB) downloaded from HuggingFace
- PANNs model (~312 MB) downloaded from Zenodo

All downloads use HTTPS. After the first run, **no further network activity occurs**.

### During Video Processing

**None.** Zero network requests. Completely offline.

### Verification

```bash
# After first run, disconnect from the internet and process a video.
# If it works, you're fully offline.
pureframe process test.mp4
```

---

## Third-Party Services

PureFrame uses **no** third-party services:

- ❌ No cloud APIs
- ❌ No analytics (Google Analytics, Mixpanel, etc.)
- ❌ No crash reporting (Sentry, Bugsnag, etc.)
- ❌ No telemetry
- ❌ No user tracking
- ❌ No advertising

---

## Your Rights

Since PureFrame processes everything locally and stores nothing remotely:

- **Right to access:** All your data is already on your machine
- **Right to deletion:** Delete the output files, job database, and cached models
- **Right to portability:** Censor plans are portable JSON files
- **Right to object:** You control every processing decision via the plan/apply workflow

---

## Children's Privacy

PureFrame does not collect any data from any users, including children. The software runs entirely offline on the user's own hardware.

---

## Changes to This Policy

We will update this policy if PureFrame's data handling changes. Check the "Last updated" date above. Any changes that introduce network-dependent features or data collection will be clearly documented in the [CHANGELOG](../CHANGELOG.md) and require explicit user opt-in.

---

## Contact

For privacy questions, open a GitHub issue or email: privacy@pureframe.dev
