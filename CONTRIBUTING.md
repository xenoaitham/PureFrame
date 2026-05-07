# Contributing to PureFrame

Thank you for your interest in contributing to PureFrame!

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/PureFrame.git
cd PureFrame
```

### 2. Set Up Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e ".[dev]"
```

### 3. Install FFmpeg

See [Installation Guide](docs/installation.md#ffmpeg-setup).

### 4. Run Tests

```bash
pytest -x -q
```

### 5. Check Code Quality

```bash
ruff check .
ruff format --check .
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest -x -q`
4. Run linting: `ruff check . && ruff format .`
5. Commit with a clear message: `git commit -m "feat: add content-type profiles"`
6. Push and open a Pull Request

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use |
|--------|-----|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding/fixing tests |
| `refactor:` | Code refactoring |
| `chore:` | Maintenance tasks |
| `perf:` | Performance improvement |

## Project Structure

See [Architecture Documentation](docs/architecture.md).

## Good First Issues

Look for issues labeled `good first issue` on GitHub. These are tasks that are well-scoped and don't require deep knowledge of the codebase.

## Testing

- All new features must include tests
- Run `pytest -x -q` before submitting
- CI runs Python 3.11, 3.12, and 3.13
- Golden-file tests are in `tests/fixtures/`

## Code Style

- Python: formatted with `ruff format`, linted with `ruff check`
- Max line length: 100 characters (ruff default)
- Type hints are required for all function signatures
- Docstrings for all public functions

## Detection Quality Reports

If you want to help improve detection accuracy:

1. Test PureFrame on different content types
2. Note false positives and false negatives
3. Open a GitHub issue with the scene description (no explicit screenshots)
4. Include PureFrame version, settings, and content type

## Questions?

Open a GitHub Discussion or issue. We're happy to help!
