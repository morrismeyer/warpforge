#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

APP_NAME="${SNAKEBURGER_APP_NAME:-Snake Burger}"
APP_DIR="$ROOT/snakeburger-cli/build/snakeburger-cli-jpackage/${APP_NAME}.app"

if [[ ! -d "$APP_DIR" ]]; then
  # fallback to first app
  APP_DIR="$(find "$ROOT/snakeburger-cli/build/snakeburger-cli-jpackage" -maxdepth 1 -name '*.app' -print -quit 2>/dev/null || true)"
fi

if [[ -z "${APP_DIR:-}" || ! -d "$APP_DIR" ]]; then
  echo "No .app found under: $ROOT/snakeburger-cli/build/snakeburger-cli-jpackage" >&2
  echo "Run: tools/snakeburger-jpackage/build-snakeburger-cli-jpackage.sh" >&2
  exit 1
fi

# The executable under Contents/MacOS usually matches --name (spaces included).
BIN="$APP_DIR/Contents/MacOS/$APP_NAME"
if [[ ! -x "$BIN" ]]; then
  # fallback: first executable in Contents/MacOS
  BIN="$(find "$APP_DIR/Contents/MacOS" -maxdepth 1 -type f -perm -111 -print -quit 2>/dev/null || true)"
fi

if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
  echo "ERROR: Could not find runnable binary under: $APP_DIR/Contents/MacOS" >&2
  exit 1
fi

exec "$BIN" "$@"
