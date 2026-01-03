#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INIT="${ROOT}/babylon-runtime/gradle/init-babylon-bootjdk.gradle"

# Boot JDK detection (can override with BABYLON_BOOT_JDK)
if [[ -z "${BABYLON_BOOT_JDK:-}" ]]; then
  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/javac" ]]; then
    BABYLON_BOOT_JDK="${JAVA_HOME}"
  elif command -v javac >/dev/null 2>&1; then
    JAVAC_PATH="$(readlink -f "$(command -v javac)")"
    BABYLON_BOOT_JDK="$(cd "$(dirname "${JAVAC_PATH}")/.." && pwd)"
  fi
fi

if [[ -z "${BABYLON_BOOT_JDK:-}" || ! -x "${BABYLON_BOOT_JDK}/bin/javac" ]]; then
  echo "ERROR: Unable to determine a Boot JDK for Babylon. Set BABYLON_BOOT_JDK to a JDK home (with bin/javac)." >&2
  exit 1
fi

# Host-specific conf name (can override with BABYLON_CONF_NAME)
if [[ -z "${BABYLON_CONF_NAME:-}" ]]; then
  OS_UNAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
  ARCH_UNAME="$(uname -m | tr '[:upper:]' '[:lower:]')"

  case "${OS_UNAME}" in
    linux*)  OS_TOKEN="linux" ;;
    darwin*) OS_TOKEN="macosx" ;;
    msys*|mingw*|cygwin*) OS_TOKEN="windows" ;;
    *)       OS_TOKEN="${OS_UNAME}" ;;
  esac

  case "${ARCH_UNAME}" in
    x86_64|amd64) ARCH_TOKEN="x86_64" ;;
    aarch64|arm64) ARCH_TOKEN="aarch64" ;;
    *) ARCH_TOKEN="${ARCH_UNAME}" ;;
  esac

  BABYLON_CONF_NAME="${OS_TOKEN}-${ARCH_TOKEN}-server-release"
fi

echo "Using BABYLON_BOOT_JDK: ${BABYLON_BOOT_JDK}"
echo "Using BABYLON_CONF_NAME: ${BABYLON_CONF_NAME}"

cd "${ROOT}"

GRADLE_ARGS=(
  "--no-configuration-cache"
  "-I" "${INIT}"
  "-Pbabylon.bootJdk=${BABYLON_BOOT_JDK}"
  "-Pbabylon.confName=${BABYLON_CONF_NAME}"
)

./gradlew "${GRADLE_ARGS[@]}" :snakeburger-cli:jar
./gradlew "${GRADLE_ARGS[@]}" :babylon-runtime:zipSnakeBurgerRuntimeImage
