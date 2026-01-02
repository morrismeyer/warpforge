#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Stop any lingering daemons so toolchain changes take effect immediately.
./gradlew --stop || true

BOOT_JDK="$(/usr/libexec/java_home)"

# Build Babylon and write env file with BABYLON_JDK_HOME
./gradlew --no-daemon --no-configuration-cache \
  :babylon-runtime:writeBabylonToolchainEnv \
  -Pbabylon.bootJdk="$BOOT_JDK"

# Load BABYLON_JDK_HOME
# shellcheck disable=SC1091
source babylon-runtime/build/babylon.toolchain.env

if [[ -z "${BABYLON_JDK_HOME:-}" ]]; then
  echo "ERROR: BABYLON_JDK_HOME was not set by babylon-runtime/build/babylon.toolchain.env" >&2
  exit 2
fi

TOOLCHAIN_ARGS=(
  "-Dorg.gradle.java.installations.paths=${BABYLON_JDK_HOME}"
  "-Dorg.gradle.java.installations.auto-detect=false"
  "-Dorg.gradle.java.installations.auto-download=false"
)

# Show what Gradle sees (useful sanity check)
./gradlew --no-daemon --no-configuration-cache \
  "${TOOLCHAIN_ARGS[@]}" \
  -q javaToolchains

# Run SnakeBurger using the Babylon toolchain
./gradlew --no-daemon --no-configuration-cache \
  "${TOOLCHAIN_ARGS[@]}" \
  :snakeburger-cli:run
