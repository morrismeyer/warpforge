#!/usr/bin/env bash
set -euo pipefail

# Phase B: build a PKG (macOS installer) from the Phase A jpackage app-image.
#
# This script ONLY BUILDS the PKG. It does not install it.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

APP_NAME="${SNAKEBURGER_APP_NAME:-Snake Burger}"

# Prefer a stable, system jpackage for packaging.
if [[ -z "${SNAKEBURGER_JPACKAGE_BIN:-}" ]]; then
  if command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JH="$(/usr/libexec/java_home 2>/dev/null || true)"
    if [[ -n "$JH" && -x "$JH/bin/jpackage" ]]; then
      export SNAKEBURGER_JPACKAGE_BIN="$JH/bin/jpackage"
    fi
  fi
fi

"$ROOT/tools/snakeburger-jpackage/build-snakeburger-cli-jpackage.sh"

APP_IMAGE_DIR="$ROOT/snakeburger-cli/build/snakeburger-cli-jpackage/${APP_NAME}.app"
if [[ ! -d "$APP_IMAGE_DIR" ]]; then
  APP_IMAGE_DIR="$(find "$ROOT/snakeburger-cli/build/snakeburger-cli-jpackage" -maxdepth 1 -name '*.app' -print -quit || true)"
fi
if [[ -z "${APP_IMAGE_DIR:-}" || ! -d "$APP_IMAGE_DIR" ]]; then
  echo "ERROR: app-image not found under: $ROOT/snakeburger-cli/build/snakeburger-cli-jpackage" >&2
  exit 1
fi

RAW_VERSION="$(ls -1 "$ROOT/snakeburger-cli/build/distributions"/snakeburger-cli-jpackage-*.zip 2>/dev/null | tail -n 1 | sed -E 's/.*snakeburger-cli-jpackage-//; s/\.zip$//' || true)"
RAW_VERSION="${RAW_VERSION:-0.0.0}"
SANITIZED="$(echo "$RAW_VERSION" | tr -cd '0-9.' )"
IFS='.' read -r V1 V2 V3 _EXTRA <<< "${SANITIZED}...."
V1="${V1:-0}"; V2="${V2:-0}"; V3="${V3:-0}"
if [[ "$V1" == "0" || -z "$V1" ]]; then V1="1"; fi
APP_VERSION_JPACKAGE="${V1}.${V2}.${V3}"

DEST_DIR="$ROOT/snakeburger-cli/build/snakeburger-cli-jpackage-pkg"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

JPACKAGE_BIN="${SNAKEBURGER_JPACKAGE_BIN:-}"
if [[ -z "$JPACKAGE_BIN" ]]; then
  if command -v jpackage >/dev/null 2>&1; then
    JPACKAGE_BIN="$(command -v jpackage)"
  fi
fi
if [[ -z "$JPACKAGE_BIN" || ! -x "$JPACKAGE_BIN" ]]; then
  echo "ERROR: Could not find an executable jpackage. Set SNAKEBURGER_JPACKAGE_BIN to a JDK's jpackage binary." >&2
  exit 1
fi

JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djdk.lang.Process.launchMechanism=FORK"
export JAVA_TOOL_OPTIONS

echo "Using jpackage: $JPACKAGE_BIN"
echo "Using app-image: $APP_IMAGE_DIR"
echo "App name: $APP_NAME"
echo "App version (raw): $RAW_VERSION"
echo "App version (jpackage): $APP_VERSION_JPACKAGE"
echo "Dest: $DEST_DIR"

"$JPACKAGE_BIN" \
  --type pkg \
  --dest "$DEST_DIR" \
  --name "$APP_NAME" \
  --app-version "$APP_VERSION_JPACKAGE" \
  --app-image "$APP_IMAGE_DIR"

PKG_PATH="$(find "$DEST_DIR" -maxdepth 1 -name '*.pkg' -print -quit || true)"
if [[ -z "${PKG_PATH:-}" ]]; then
  echo "ERROR: PKG was not produced under: $DEST_DIR" >&2
  exit 1
fi

DIST_DIR="$ROOT/snakeburger-cli/build/distributions"
mkdir -p "$DIST_DIR"
PKG_OUT="$DIST_DIR/snakeburger-cli-${RAW_VERSION}.pkg"
rm -f "$PKG_OUT"
cp -f "$PKG_PATH" "$PKG_OUT"

echo "Wrote: $PKG_OUT"
