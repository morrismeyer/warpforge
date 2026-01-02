#!/usr/bin/env bash
set -euo pipefail

# Phase A (shared): build a macOS jpackage app-image (.app) for snakeburger-cli.
#
# This reuses the staged "appimage" layout produced by:
#   tools/snakeburger-app-image/build-snakeburger-cli-appimage.sh
#
# Env:
#   SNAKEBURGER_MAIN_CLASS (optional): override main class (e.g. io.surfworks.snakeburger.cli.SnakeBurgerMain)
#   SNAKEBURGER_APP_NAME   (optional): override app name (default: "Snake Burger")
#   SNAKEBURGER_JPACKAGE_BIN (optional): path to a specific jpackage executable to use.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

APP_NAME="${SNAKEBURGER_APP_NAME:-Snake Burger}"

# Reuse the appimage builder to produce stage dir + runtime
"$ROOT/tools/snakeburger-app-image/build-snakeburger-cli-appimage.sh"

STAGE="$ROOT/snakeburger-cli/build/snakeburger-cli-appimage"
RUNTIME="$STAGE/runtime"
APPDIR="$STAGE/app"

if [[ ! -d "$STAGE" || ! -d "$APPDIR" || ! -d "$RUNTIME" ]]; then
  echo "ERROR: stage directories not found. Expected:" >&2
  echo "  $STAGE" >&2
  echo "  $APPDIR" >&2
  echo "  $RUNTIME" >&2
  exit 1
fi

# Choose jpackage:
#  1) explicit SNAKEBURGER_JPACKAGE_BIN
#  2) macOS java_home default JDK's jpackage
#  3) PATH jpackage
#  4) fallback: run jpackage as a module using the staged runtime (java -m jdk.jpackage/...)
choose_jpackage() {
  if [[ -n "${SNAKEBURGER_JPACKAGE_BIN:-}" && -x "${SNAKEBURGER_JPACKAGE_BIN}" ]]; then
    echo "${SNAKEBURGER_JPACKAGE_BIN}"
    return 0
  fi

  if command -v /usr/libexec/java_home >/dev/null 2>&1; then
    local jh
    jh="$(/usr/libexec/java_home 2>/dev/null || true)"
    if [[ -n "$jh" && -x "$jh/bin/jpackage" ]]; then
      echo "$jh/bin/jpackage"
      return 0
    fi
  fi

  if command -v jpackage >/dev/null 2>&1; then
    echo "$(command -v jpackage)"
    return 0
  fi

  echo ""  # signal module fallback
}

JPACKAGE_BIN="$(choose_jpackage)"

# Work around macOS posix_spawn issues when jpackage runs /usr/bin/codesign
# by forcing Java to use the legacy fork/exec launch mechanism.
# We do this via JAVA_TOOL_OPTIONS so it applies whether jpackage is a script/exe or a module.
JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djdk.lang.Process.launchMechanism=FORK"
export JAVA_TOOL_OPTIONS

# Determine main jar (first jar in app/lib matching snakeburger-cli-*.jar)
MAIN_JAR="$(ls -1 "$APPDIR/lib"/snakeburger-cli-*.jar 2>/dev/null | head -n 1 || true)"
if [[ -z "${MAIN_JAR:-}" ]]; then
  echo "ERROR: could not find snakeburger-cli jar under: $APPDIR/lib" >&2
  exit 1
fi
MAIN_JAR_BASENAME="$(basename "$MAIN_JAR")"

# Determine main class:
#  1) explicit env var SNAKEBURGER_MAIN_CLASS
#  2) parse from Gradle launcher script (APP_MAIN_CLASS=...)
MAIN_CLASS="${SNAKEBURGER_MAIN_CLASS:-}"
if [[ -z "${MAIN_CLASS}" ]]; then
  LAUNCHER="$APPDIR/bin/snakeburger-cli"
  if [[ -f "$LAUNCHER" ]]; then
    MAIN_CLASS="$(grep -E '^APP_MAIN_CLASS=' "$LAUNCHER" | head -n 1 | cut -d= -f2- | tr -d '\r' || true)"
  fi
fi
if [[ -z "${MAIN_CLASS}" ]]; then
  echo "ERROR: Could not determine the main class for jpackage." >&2
  echo "Fix: export SNAKEBURGER_MAIN_CLASS='<your main class>' and rerun." >&2
  echo "Hint: the Gradle launcher usually contains APP_MAIN_CLASS=... in: $APPDIR/bin/snakeburger-cli" >&2
  exit 1
fi

# Determine app version from jar name (best-effort).
# jpackage requires 1..3 integers separated by dots, and major must be >= 1.
RAW_VERSION="$(echo "$MAIN_JAR_BASENAME" | sed -E 's/^snakeburger-cli-//; s/\.jar$//')"

# Convert to digits/dots only
SANITIZED="$(echo "$RAW_VERSION" | tr -cd '0-9.' )"
# Ensure it has at least 3 parts
IFS='.' read -r V1 V2 V3 _EXTRA <<< "${SANITIZED}...."
V1="${V1:-0}"; V2="${V2:-0}"; V3="${V3:-0}"

# If V1 is 0 or empty, force major to 1 and keep the rest as best-effort
if [[ "$V1" == "0" || -z "$V1" ]]; then
  V1="1"
fi

APP_VERSION_JPACKAGE="${V1}.${V2}.${V3}"

# ---- tzdb.dat fix (critical on macOS) ---------------------------------------
# jpackage internally uses TimeZone/ZoneInfo and expects a readable tzdb.dat at:
#   <runtime>/lib/tzdb.dat
# Our appimage build can create an empty placeholder or lose symlink targets.
ensure_tzdb() {
  local runtime="$1"
  local runtime_lib="$runtime/lib"
  local target="$runtime_lib/tzdb.dat"
  mkdir -p "$runtime_lib"

  if [[ -f "$target" && -s "$target" ]]; then
    echo "tzdb.dat present: $target ($(stat -f%z "$target" 2>/dev/null || echo "?") bytes)"
    return 0
  fi

  # If file exists but empty, remove it to avoid EOFException
  if [[ -f "$target" && ! -s "$target" ]]; then
    rm -f "$target"
  fi

  local candidates=()

  if [[ -n "${BABYLON_JDK_HOME:-}" ]]; then
    candidates+=( "$BABYLON_JDK_HOME/lib/tzdb.dat" )
    candidates+=( "$(cd "$BABYLON_JDK_HOME/.." && pwd)/images/jdk/lib/tzdb.dat" )
  fi

  if command -v /usr/libexec/java_home >/dev/null 2>&1; then
    local default_home
    default_home="$(/usr/libexec/java_home 2>/dev/null || true)"
    if [[ -n "$default_home" ]]; then
      candidates+=( "$default_home/lib/tzdb.dat" )
    fi
  fi

  if [[ -d /Library/Java/JavaVirtualMachines ]]; then
    local jhome
    for jhome in /Library/Java/JavaVirtualMachines/*/Contents/Home; do
      [[ -d "$jhome" ]] || continue
      candidates+=( "$jhome/lib/tzdb.dat" )
    done
  fi

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" && -s "$c" ]]; then
      cp -Lf "$c" "$target"
      echo "Copied tzdb.dat from: $c"
      echo "Created tzdb.dat at: $target ($(stat -f%z "$target" 2>/dev/null || echo "?") bytes)"
      return 0
    fi
  done

  echo "ERROR: tzdb.dat is missing or empty at: $target" >&2
  echo "Searched candidates:" >&2
  for c in "${candidates[@]}"; do echo "  - $c" >&2; done
  echo "Fix: Ensure your Babylon JDK image contains lib/tzdb.dat, or install a standard JDK so /usr/libexec/java_home can find one." >&2
  return 1
}

ensure_tzdb "$RUNTIME"
# -----------------------------------------------------------------------------

echo "Using stage: $STAGE"
if [[ -n "$JPACKAGE_BIN" ]]; then
  echo "Using jpackage (bin): $JPACKAGE_BIN"
else
  echo "Using jpackage (module): $RUNTIME/bin/java -m jdk.jpackage/jdk.jpackage.main.Main"
fi

echo "Main jar: $MAIN_JAR_BASENAME"
echo "Main class: $MAIN_CLASS"
echo "App name: $APP_NAME"
echo "App version (raw): $RAW_VERSION"
echo "App version (jpackage): $APP_VERSION_JPACKAGE"

DEST="$ROOT/snakeburger-cli/build/snakeburger-cli-jpackage"
rm -rf "$DEST"
mkdir -p "$DEST"

# Create an app-image (.app) using jpackage
if [[ -n "$JPACKAGE_BIN" ]]; then
  "$JPACKAGE_BIN" \
    --type app-image \
    --dest "$DEST" \
    --name "$APP_NAME" \
    --app-version "$APP_VERSION_JPACKAGE" \
    --input "$APPDIR/lib" \
    --main-jar "$MAIN_JAR_BASENAME" \
    --main-class "$MAIN_CLASS" \
    --runtime-image "$RUNTIME" \
    --java-options "-Dfile.encoding=UTF-8" \
    --java-options "-Duser.language=en" \
    --java-options "-Duser.country=US"
else
  "$RUNTIME/bin/java" -m jdk.jpackage/jdk.jpackage.main.Main \
    --type app-image \
    --dest "$DEST" \
    --name "$APP_NAME" \
    --app-version "$APP_VERSION_JPACKAGE" \
    --input "$APPDIR/lib" \
    --main-jar "$MAIN_JAR_BASENAME" \
    --main-class "$MAIN_CLASS" \
    --runtime-image "$RUNTIME" \
    --java-options "-Dfile.encoding=UTF-8" \
    --java-options "-Duser.language=en" \
    --java-options "-Duser.country=US"
fi

# Zip the resulting .app for easy copying
APP_PATH="$(find "$DEST" -maxdepth 1 -name '*.app' -print -quit || true)"
if [[ -z "${APP_PATH:-}" ]]; then
  echo "ERROR: No .app found under: $DEST" >&2
  exit 1
fi

ZIP_OUT="$ROOT/snakeburger-cli/build/distributions/snakeburger-cli-jpackage-${RAW_VERSION}.zip"
mkdir -p "$(dirname "$ZIP_OUT")"
rm -f "$ZIP_OUT"
(
  cd "$(dirname "$APP_PATH")"
  /usr/bin/zip -qry "$ZIP_OUT" "$(basename "$APP_PATH")"
)

echo "Wrote: $ZIP_OUT"
