#!/usr/bin/env bash
# Install-path smoke test for the example plugin package.
#
# Creates a throwaway virtualenv, installs PureFrame plus
# pureframe-plugins-examples from this repository, and walks the exact
# path a real user takes: `pureframe plugins list` shows the plugin
# through its installed entry point, a tiny `pureframe process` run with
# --enable-plugin motionblob censors a synthetic clip end to end, and
# after pip uninstall the plugin is gone from the listing. Nothing else
# in the suite installs packages; this script is the only place where
# the real entry-point machinery is exercised in CI.
#
# Usage: scripts/plugin_install_smoke.sh   (from the repo root or anywhere)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIR=/tmp/pureframe-plugin-install-smoke
VENV="$SMOKE_DIR/venv"
PY="$VENV/bin/python"
PUREFRAME="$VENV/bin/pureframe"

rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
echo "smoke scratch dir: $SMOKE_DIR"

python3 -m venv "$VENV"

echo "== install cpu torch =="
"$PY" -m pip install --no-input torch --index-url https://download.pytorch.org/whl/cpu
echo "== install pureframe =="
"$PY" -m pip install --no-input "$ROOT"
echo "== install example plugin =="
"$PY" -m pip install --no-input "$ROOT/examples/pureframe-plugins-examples"

echo "== plugins list =="
"$PUREFRAME" plugins list | tee "$SMOKE_DIR/listing.txt"
grep -q motionblob "$SMOKE_DIR/listing.txt"
echo "ok: motionblob discovered through its installed entry point"

# 4 s of always-moving testsrc2 at 240p: small enough to analyze in
# seconds, busy enough for a motion detector to flag throughout.
ffmpeg -nostdin -y -loglevel error \
  -f lavfi -i testsrc2=duration=4:size=320x240:rate=15 \
  -pix_fmt yuv420p -c:v libx264 -crf 28 \
  "$SMOKE_DIR/motion.mp4"

echo "== process with --enable-plugin motionblob =="
"$PUREFRAME" process "$SMOKE_DIR/motion.mp4" \
  --output "$SMOKE_DIR/motion_censored.mp4" \
  --profile cpu --no-clip --no-audio \
  --enable-plugin motionblob \
  | tee "$SMOKE_DIR/process.log"
grep -q "Plugins enabled" "$SMOKE_DIR/process.log"
test -s "$SMOKE_DIR/motion_censored.mp4"
echo "ok: censored output rendered with the plugin enabled"

echo "== uninstall example plugin =="
"$PY" -m pip uninstall -y pureframe-plugins-examples
"$PUREFRAME" plugins list | tee "$SMOKE_DIR/listing_after.txt"
if grep -q motionblob "$SMOKE_DIR/listing_after.txt"; then
  echo "FAILED: motionblob still listed after uninstall" >&2
  exit 1
fi
echo "ok: motionblob gone after uninstall"

if [ "${PUREFRAME_SMOKE_KEEP:-0}" != "1" ]; then
  rm -rf "$SMOKE_DIR"
fi
echo "plugin install smoke: all green"
