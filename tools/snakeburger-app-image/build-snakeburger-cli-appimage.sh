#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Load BABYLON_JDK_HOME from the env file if present (bash only).
ENV_FILE="babylon-runtime/build/babylon.toolchain.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

if [[ -z "${BABYLON_JDK_HOME:-}" ]]; then
  echo "BABYLON_JDK_HOME is not set."
  echo "Run: babylon-runtime/run-snakeburger.sh  (it generates babylon-runtime/build/babylon.toolchain.env)"
  exit 1
fi

if [[ ! -d "$BABYLON_JDK_HOME" ]]; then
  echo "BABYLON_JDK_HOME does not exist: $BABYLON_JDK_HOME"
  exit 1
fi

copy_tree() {
  local src="$1"
  local dst="$2"

  if command -v ditto >/dev/null 2>&1; then
    # macOS: preserve permissions, xattrs, and resource forks.
    ditto "$src" "$dst"
    return
  fi

  # Linux: prefer rsync if available, otherwise fall back to cp -a.
  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "$dst"
    rsync -aH --links "$src"/ "$dst"/
  else
    mkdir -p "$dst"
    cp -a "$src"/. "$dst"/
  fi
}


# Find JMODs.
SNAKEBURGER_JMODS_DIR=""
if [[ -d "$BABYLON_JDK_HOME/jmods" ]]; then
  SNAKEBURGER_JMODS_DIR="$BABYLON_JDK_HOME/jmods"
else
  # Babylon builds often place JMODs under build/.../images/jmods
  CANDIDATE="$(cd "$(dirname "$BABYLON_JDK_HOME")" && pwd)/images/jmods"
  if [[ -d "$CANDIDATE" ]]; then
    SNAKEBURGER_JMODS_DIR="$CANDIDATE"
  fi
fi

echo "Using BABYLON_JDK_HOME: $BABYLON_JDK_HOME"
if [[ -n "$SNAKEBURGER_JMODS_DIR" ]]; then
  echo "Using SNAKEBURGER_JMODS_DIR: $SNAKEBURGER_JMODS_DIR"
else
  echo "No jmods/ directory found. jlink will be skipped and we will bundle the full Babylon JDK runtime image."
fi

# Force Gradle to use the Babylon JDK toolchain (and do not auto-download toolchains).
JAVA_TOOLCHAIN_OPTS=(
  "-Dorg.gradle.java.installations.paths=${BABYLON_JDK_HOME}"
  "-Dorg.gradle.java.installations.auto-detect=false"
  "-Dorg.gradle.java.installations.auto-download=false"
)

# Build the application distribution dir (Gradle application plugin).
# This produces: snakeburger-cli/build/install/snakeburger-cli/{bin,lib}
./gradlew --no-daemon --no-configuration-cache "${JAVA_TOOLCHAIN_OPTS[@]}" :snakeburger-cli:installDist

# Discover version for naming the appimage zip (fallback to SNAPSHOT).
VERSION="$(./gradlew -q :snakeburger-cli:properties --no-configuration-cache "${JAVA_TOOLCHAIN_OPTS[@]}" \
  | awk -F': ' '$1=="version"{print $2; exit}')"
if [[ -z "$VERSION" ]]; then
  VERSION="0.0.1-SNAPSHOT"
fi

STAGING_DIR="snakeburger-cli/build/snakeburger-cli-appimage"
DIST_DIR="snakeburger-cli/build/distributions"
OUT_ZIP="${DIST_DIR}/snakeburger-cli-appimage-${VERSION}.zip"

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
mkdir -p "$DIST_DIR"

# Stage app/ from installDist
mkdir -p "$STAGING_DIR/app"
copy_tree "snakeburger-cli/build/install/snakeburger-cli" "$STAGING_DIR/app"

# Write a small launcher that sets JAVA_HOME to the bundled runtime.
mkdir -p "$STAGING_DIR/bin"
cat > "$STAGING_DIR/bin/snakeburger" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$HERE/../runtime"

JAVA_HOME_CANDIDATE="$RUNTIME_DIR"
if [[ -x "$RUNTIME_DIR/Contents/Home/bin/java" ]]; then
  JAVA_HOME_CANDIDATE="$RUNTIME_DIR/Contents/Home"
fi

export JAVA_HOME="$JAVA_HOME_CANDIDATE"
export PATH="$JAVA_HOME/bin:$PATH"

exec "$HERE/../app/bin/snakeburger-cli" "$@"
EOF
chmod +x "$STAGING_DIR/bin/snakeburger"

# Helper: prune broken symlinks, then ensure a minimal lib/jvm.cfg exists.
prune_broken_symlinks() {
  local root="$1"
  local removed=0
  while IFS= read -r -d '' lnk; do
    if [[ -L "$lnk" && ! -e "$lnk" ]]; then
      rm -f "$lnk"
      removed=$((removed+1))
    fi
  done < <(find "$root" -type l -print0 2>/dev/null || true)
  if [[ "$removed" -gt 0 ]]; then
    echo "Removed ${removed} broken symlink(s) under runtime/."
  fi
}

ensure_jvm_cfg() {
  local runtime="$1"
  local java_home="$runtime"
  if [[ -x "$runtime/Contents/Home/bin/java" ]]; then
    java_home="$runtime/Contents/Home"
  fi
  local cfg="$java_home/lib/jvm.cfg"
  if [[ -L "$cfg" && ! -e "$cfg" ]]; then
    rm -f "$cfg"
  fi
  if [[ ! -f "$cfg" ]]; then
    mkdir -p "$(dirname "$cfg")"
    printf "%s\n" "-server KNOWN" > "$cfg"
    echo "Created missing jvm.cfg at: $cfg"
  fi
}

# Stage runtime/ via jlink if possible, otherwise copy the full Babylon JDK image.
RUNTIME_OUT="$STAGING_DIR/runtime"

try_jlink() {
  if [[ -z "$SNAKEBURGER_JMODS_DIR" ]]; then
    return 1
  fi
  if [[ ! -x "$BABYLON_JDK_HOME/bin/jlink" ]]; then
    return 1
  fi

  # Minimal set for now. If this is insufficient, jlink will fail and we will fall back.
  local modules="java.base,jdk.incubator.code"

  rm -rf "$RUNTIME_OUT"
  "$BABYLON_JDK_HOME/bin/jlink" \
    --module-path "$SNAKEBURGER_JMODS_DIR" \
    --add-modules "$modules" \
    --no-header-files \
    --no-man-pages \
    --output "$RUNTIME_OUT" \
    > "snakeburger-cli/build/snakeburgerCliJlinkRuntime.log" 2>&1
}

if try_jlink; then
  echo "jlink succeeded."
  prune_broken_symlinks "$RUNTIME_OUT"
  ensure_jvm_cfg "$RUNTIME_OUT"
else
  echo "jlink failed or was skipped. Bundling the full Babylon JDK runtime image (larger, but should run)."
  rm -rf "$RUNTIME_OUT"
  mkdir -p "$RUNTIME_OUT"
  copy_tree "$BABYLON_JDK_HOME" "$RUNTIME_OUT"
  prune_broken_symlinks "$RUNTIME_OUT"
  ensure_jvm_cfg "$RUNTIME_OUT"
fi

# Create the zip (store symlinks instead of following them).
rm -f "$OUT_ZIP"
(
  cd "$STAGING_DIR"
  # -q quiet, -r recurse, -y store symlinks
  zip -qry "$ROOT/$OUT_ZIP" .
)

echo "Wrote: $OUT_ZIP"
