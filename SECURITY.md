# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in PureFrame, please report it responsibly:

1. **Email:** security@pureframe.dev (or open a private GitHub security advisory)
2. **Do not** open a public issue for security vulnerabilities
3. **Include:** steps to reproduce, affected versions, and potential impact

### Disclosure Timeline

- We aim to acknowledge reports within **48 hours**
- We aim to release fixes within **7 days** for critical issues
- We will credit reporters unless they prefer to remain anonymous

---

## Security Model

### Threat Model

PureFrame processes potentially sensitive media files. Our security model addresses:

| Threat | Mitigation |
|--------|-----------|
| **Private media exposure** | All processing is 100% local. No frames, audio, or metadata leave your machine. |
| **Malicious video files** | FFmpeg handles decoding; we rely on FFmpeg's security track record. Malformed files may cause crashes but not code execution in PureFrame's Python layer. |
| **Malicious model files** | Models are downloaded from trusted sources (PyPI, HuggingFace, Zenodo) over HTTPS. We recommend verifying checksums. |
| **Supply chain attacks** | Dependencies are pinned to minimum versions. We use GitHub Dependabot for vulnerability scanning. |
| **Data exfiltration** | No network requests are made during processing. PureFrame is fully offline after first model download. |

### Network Request Audit

PureFrame makes **zero** network requests during video processing. The only network activity occurs during:

1. **First-time model download** (NudeNet, CLIP, PANNs) - uses HTTPS from trusted sources
2. **pip install** - standard PyPI package installation

After installation and first run, PureFrame works **completely offline**.

### Telemetry Statement

**PureFrame collects no telemetry, analytics, or usage data. Period.**

No crash reports, no usage statistics, no phone-home checks. Your video processing is entirely private.

### Dependencies That Could Make Network Requests

| Dependency | Network Behavior |
|------------|-----------------|
| `nudenet` | Downloads ONNX model on first use (from PyPI assets) |
| `transformers` | Downloads CLIP model on first use (from HuggingFace) |
| `panns-inference` | Downloads CNN14 model on first use (from Zenodo) |
| `torch` | No network requests during inference |
| `onnxruntime` | No network requests |
| `ffmpeg-python` | No network requests (local FFmpeg wrapper) |

All other dependencies (`pydantic`, `typer`, `rich`, `numpy`, `opencv`, etc.) make **no network requests**.

### Offline Mode Verification

To verify PureFrame is fully offline:

```bash
# 1. Pre-download all models
pureframe process --help
python -c "from nudenet import NudeDetector; NudeDetector()"

# 2. Disconnect from internet

# 3. Process a video
pureframe process test_video.mp4

# If this succeeds, you're fully offline.
```

---

## Dependency Security

### Vulnerability Scanning

- GitHub Dependabot is enabled for automated vulnerability alerts
- CodeQL analysis runs on every push via GitHub Actions
- We review dependency updates weekly

### Pinning Strategy

We use minimum version constraints (`>=`) rather than exact pins to allow security patches while maintaining compatibility. Critical security dependencies:

| Package | Min Version | Purpose |
|---------|------------|---------|
| `torch` | ≥2.4.0 | Neural network inference |
| `onnxruntime` | ≥1.15.0 | ONNX model execution |
| `numpy` | ≥1.24.0 | Array operations |
| `pydantic` | ≥2.0 | Data validation |

### SBOM (Software Bill of Materials)

Generate a full dependency tree:

```bash
pip install pipdeptree
pipdeptree --packages pureframe --json > sbom.json
```

---

## Safe Usage Guidelines

1. **Process your own legal copies** of media only
2. **Review censor plans** before applying - use the `plan` + `plan-edit` workflow
3. **Keep PureFrame updated** to receive security fixes
4. **Store output files securely** - the censored video is still your responsibility
5. **Delete model caches** if you want to remove all traces of PureFrame from your system

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.0b4 | ✅ Current |
| 0.1.0b3 | ⚠️ Upgrade recommended |
| 0.1.0b2 | ⚠️ Upgrade recommended |
| < 0.1.0b2 | ❌ Not supported |
