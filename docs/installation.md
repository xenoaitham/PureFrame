# Installation Guide

## Quick Install (All Platforms)

```bash
pip install pureframe
```

> **Prerequisite:** FFmpeg must be installed and accessible in your PATH. See [FFmpeg Setup](#ffmpeg-setup) below.

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# 1. Install FFmpeg
sudo apt update && sudo apt install -y ffmpeg

# 2. Install Python 3.11+ (usually pre-installed)
python3 --version  # must be 3.11+

# 3. Install PureFrame
pip install pureframe

# 4. Verify
pureframe --version
ffmpeg -version
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install -y ffmpeg python3-pip
pip install pureframe
```

### Linux (Arch)

```bash
sudo pacman -S ffmpeg python-pip
pip install pureframe
```

### macOS

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install FFmpeg and Python
brew install ffmpeg python@3.12

# 3. Install PureFrame
pip3 install pureframe

# 4. Verify
pureframe --version
```

### Windows

```powershell
# 1. Install Python 3.11+ from python.org
# Make sure to check "Add Python to PATH" during installation

# 2. Install FFmpeg
# Option A: Using winget
winget install Gyan.FFmpeg

# Option B: Using chocolatey
choco install ffmpeg

# Option C: Manual download from https://www.gyan.dev/ffmpeg/builds/
# Extract and add the bin/ folder to your PATH

# 3. Install PureFrame
pip install pureframe

# 4. Verify
pureframe --version
ffmpeg -version
```

---

## FFmpeg Setup

PureFrame requires FFmpeg for video decoding and encoding. Minimum version: **4.4**.

| Platform | Command | Notes |
|----------|---------|-------|
| Ubuntu/Debian | `sudo apt install ffmpeg` | Usually 5.x or 6.x |
| Fedora | `sudo dnf install ffmpeg` | Enable RPM Fusion first |
| Arch | `sudo pacman -S ffmpeg` | Always latest |
| macOS | `brew install ffmpeg` | Includes all codecs |
| Windows | `winget install Gyan.FFmpeg` | Full build recommended |

### Verify FFmpeg

```bash
ffmpeg -version
# Should show version 4.4+

ffmpeg -encoders | grep libx264
# Should show at least one h264 encoder
```

---

## GPU Setup

PureFrame automatically detects your hardware and selects the optimal profile.

### NVIDIA GPU (CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

The hardware encoder (`h264_nvenc`) requires NVIDIA drivers 525+ and CUDA toolkit.

### Apple Silicon (M1/M2/M3)

PyTorch on Apple Silicon uses MPS (Metal Performance Shaders) automatically. No extra setup needed.

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### CPU Only

No special setup needed. PureFrame falls back to CPU automatically. Processing will be slower but fully functional.

```bash
# Install CPU-only PyTorch (smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Model Downloads

PureFrame downloads detection models on first use:

| Model | Size | Source | Purpose |
|-------|------|--------|---------|
| NudeNet ONNX | ~120 MB | [nudenet PyPI](https://pypi.org/project/nudenet/) | Nudity/explicit detection |
| CLIP ViT-B/32 | ~350 MB | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) | Scene classification |
| PANNs CNN14 | ~312 MB | [Zenodo](https://zenodo.org/record/3987831) | Audio classification |

Models are stored in:
- Linux: `~/.cache/` (nudenet, huggingface, panns)
- macOS: `~/Library/Caches/`
- Windows: `%LOCALAPPDATA%\cache\`

### Offline Mode

After the first run downloads all models, PureFrame works completely offline. No internet connection is ever needed for video processing.

To pre-download all models:
```bash
pureframe process --help  # Triggers model cache population
python -c "from nudenet import NudeDetector; NudeDetector()"
```

### Deleting Models

To free disk space, remove cached models:
```bash
# Linux/macOS
rm -rf ~/.cache/nudenet
rm -rf ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32
rm -rf ~/panns_data

# Windows
rmdir /s %LOCALAPPDATA%\cache\nudenet
```

---

## Troubleshooting

### "FFmpeg not found"

Make sure FFmpeg is in your system PATH:
```bash
which ffmpeg    # Linux/macOS
where ffmpeg    # Windows
```

### "CUDA out of memory"

Use a lower hardware profile:
```bash
pureframe process video.mp4 --profile cpu
pureframe process video.mp4 --profile low
```

### "ModuleNotFoundError: No module named 'pureframe'"

Make sure you installed with pip in the correct Python environment:
```bash
pip show pureframe
```

### "Permission denied" on output file

Make sure the output directory is writable:
```bash
pureframe process video.mp4 --output ~/Videos/output.mp4
```

### Slow processing on CPU

Expected. CPU-only mode processes at approximately 0.5-2 fps for detection. Use `--no-audio` and `--no-clip` to skip non-essential detectors:
```bash
pureframe process video.mp4 --no-audio --no-clip
```

### Video has no audio after processing

This is a known limitation when using `--no-audio`. Without this flag, audio is preserved automatically.

### Encoding errors / broken output

Try forcing CPU encoding:
```bash
pureframe process video.mp4 --profile cpu
```

If the issue persists, report it with:
```bash
ffmpeg -version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pureframe --version
```
