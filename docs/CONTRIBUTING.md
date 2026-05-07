# Contributing to PureFrame

Thanks for wanting to contribute! Here's how to get started.

## Setup

```bash
git clone https://github.com/xenoaitham/PureFrame.git
cd PureFrame
pip install -e ".[dev]"
```

For GUI work:
```bash
cd gui && npm install
```

## Architecture

- **`pureframe/pipeline/`** - Detection, scene analysis, confidence fusion, rendering. The core processing loop.
- **`pureframe/pipeline/render/`** - Plan serialization and frame-by-frame apply logic.
- **`gui/`** - Tauri desktop app (React + Rust). Calls the Python CLI via subprocess - keep it thin.
- **`tests/`** - pytest suite. Run with `pytest`.

## Pull Requests

1. Run `pytest` locally before pushing.
2. Run `ruff check pureframe tests` for linting.
3. If your change touches the CLI interface or plan schema, update the docs.
4. Keep PRs focused - one feature or fix per PR.

## Code Style

- Python: follow PEP 8. We use `ruff` for linting and formatting.
- Rust (GUI): standard `cargo fmt`.
- TypeScript (GUI): Vite defaults.

## Reporting Issues

Open an issue with:
- Your OS, Python version, and GPU (if applicable)
- The command you ran
- The full error traceback

## Community

Join the Discord to discuss larger changes before writing code. Link in the README.
