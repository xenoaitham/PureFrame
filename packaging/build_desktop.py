"""PureFrame Desktop Packaging - PyInstaller spec and build script.

Usage:
    python packaging/build_desktop.py [--platform linux|windows|macos]

This creates a standalone executable that bundles:
- Python runtime
- PyTorch (CPU-only)
- All ONNX/CLIP models
- FFmpeg (must be pre-installed or bundled separately)

Output:
    dist/pureframe           (Linux/macOS executable)
    dist/pureframe.exe       (Windows executable)
"""

import platform
import subprocess
import sys
from pathlib import Path


def get_pyinstaller_args() -> list[str]:
    """Build PyInstaller command-line arguments."""
    root = Path(__file__).parent.parent
    main_script = root / "pureframe" / "cli.py"

    args = [
        "pyinstaller",
        "--onedir",  # onedir is faster to build and start than onefile
        "--name",
        "pureframe",
        "--noconfirm",
        "--clean",
        # Hidden imports that PyInstaller might miss
        "--hidden-import",
        "pureframe",
        "--hidden-import",
        "pureframe.cli",
        "--hidden-import",
        "pureframe.eval",
        "--hidden-import",
        "pureframe.pipeline",
        "--hidden-import",
        "pureframe.pipeline.detect",
        "--hidden-import",
        "pureframe.pipeline.render",
        "--hidden-import",
        "pureframe.tracking",
        "--hidden-import",
        "pureframe.utils",
        "--hidden-import",
        "typer",
        "--hidden-import",
        "rich",
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "numpy",
        "--hidden-import",
        "torch",
        "--hidden-import",
        "transformers",
        "--hidden-import",
        "PIL",
        "--hidden-import",
        "librosa",
        "--hidden-import",
        "scipy",
        "--hidden-import",
        "platformdirs",
        # panns_inference imports matplotlib.pyplot at module level; the
        # audio classifier constructs it lazily on every process/plan run
        # with audio, so excluding matplotlib here made every standalone
        # crash on real work while --version still looked fine.
        "--hidden-import",
        "panns_inference",
        "--hidden-import",
        "matplotlib",
        # NudeNet's detector resolves 320n.onnx next to its own module
        # file; PyInstaller collects .py modules but not package data by
        # default, so without this the frozen app died on first detection
        # with NO_SUCHFILE for the model.
        "--collect-data",
        "nudenet",
        # Exclude unnecessary modules to reduce size
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "IPython",
        "--exclude-module",
        "jupyter",
        "--exclude-module",
        "notebook",
        # Console mode
        "--console",
        str(main_script),
    ]

    return args


def build(target_platform: str = "auto"):
    """Build the desktop executable."""
    if target_platform == "auto":
        target_platform = platform.system().lower()

    print(f"Building PureFrame for {target_platform}...")
    print()

    # Check PyInstaller is installed
    try:
        import PyInstaller

        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Run PyInstaller
    args = get_pyinstaller_args()
    print(f"Running: {' '.join(args[:5])}...")
    subprocess.check_call(args)

    # Post-build info
    dist_dir = Path("dist") / "pureframe"
    if dist_dir.exists():
        total_size = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
        print()
        print("[OK] Build complete!")
        print(f"   Output: {dist_dir}")
        print(f"   Size: {total_size / 1024 / 1024:.1f} MB")
        print()
        print("To test:")
        if target_platform == "windows":
            print("   .\\dist\\pureframe\\pureframe.exe --version")
        else:
            print("   ./dist/pureframe/pureframe --version")
    else:
        print("[FAIL] Build may have failed - check PyInstaller output above.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PureFrame desktop executable")
    parser.add_argument(
        "--platform",
        choices=["linux", "windows", "macos", "auto"],
        default="auto",
        help="Target platform (default: auto-detect)",
    )
    args = parser.parse_args()
    build(args.platform)
