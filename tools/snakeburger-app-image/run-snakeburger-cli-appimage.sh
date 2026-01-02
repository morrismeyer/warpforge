#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DIST_DIR="snakeburger-cli/build/distributions"
ZIP="$(ls -1t "$DIST_DIR"/*appimage*.zip 2>/dev/null | head -n 1 || true)"

if [[ -z "${ZIP:-}" || ! -f "$ZIP" ]]; then
  echo "No app-image zip found under $DIST_DIR"
  echo "Run: tools/snakeburger-app-image/build-snakeburger-cli-appimage.sh"
  exit 1
fi

KEEP_TMP="${SNAKEBURGER_KEEP_TMP:-0}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/snakeburger-appimage.XXXXXX")"

cleanup() {
  if [[ "$KEEP_TMP" != "1" ]]; then
    rm -rf "$TMP_DIR"
  else
    echo "Keeping temp dir: $TMP_DIR"
  fi
}
trap cleanup EXIT

unzip -q "$ZIP" -d "$TMP_DIR"

if [[ ! -x "$TMP_DIR/bin/snakeburger" ]]; then
  echo "Expected launcher not found: $TMP_DIR/bin/snakeburger"
  exit 1
fi

exec "$TMP_DIR/bin/snakeburger" "$@"
