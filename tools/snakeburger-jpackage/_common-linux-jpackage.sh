#!/usr/bin/env bash
# Common helpers for Linux jpackage wrappers.
# This file is sourced by build-snakeburger-cli-deb.sh and build-snakeburger-cli-rpm.sh.

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

repo_root() {
  # Resolve repo root based on this file location
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "$here/../.." && pwd)
}

default_if_unset() {
  local var="$1"
  local val="$2"
  if [[ -z "${!var:-}" ]]; then
    export "$var"="$val"
  fi
}

ensure_main_class() {
  default_if_unset SNAKEBURGER_MAIN_CLASS "io.surfworks.snakeburger.cli.SnakeBurgerMain"
}

ensure_app_name() {
  default_if_unset SNAKEBURGER_APP_NAME "Snake Burger"
}

ensure_pkg_name() {
  # Package name cannot contain spaces for deb/rpm metadata.
  default_if_unset SNAKEBURGER_PKG_NAME "snakeburger"
}

jpackage_bin() {
  if [[ -n "${SNAKEBURGER_JPACKAGE_BIN:-}" ]]; then
    echo "$SNAKEBURGER_JPACKAGE_BIN"
    return 0
  fi

  # Prefer the Babylon JDK's jpackage if available (keeps toolchain consistent).
  if [[ -n "${BABYLON_JDK_HOME:-}" ]] && [[ -x "${BABYLON_JDK_HOME}/bin/jpackage" ]]; then
    echo "${BABYLON_JDK_HOME}/bin/jpackage"
    return 0
  fi

  if require_cmd jpackage; then
    command -v jpackage
    return 0
  fi

  return 1
}

stage_dir() {
  local root
  root="$(repo_root)"
  echo "$root/snakeburger-cli/build/snakeburger-cli-appimage"
}

build_stage() {
  local root
  root="$(repo_root)"
  ensure_main_class
  "$root/tools/snakeburger-jpackage/build-snakeburger-cli-linux-appimage.sh"
}

main_jar_path() {
  local stage
  stage="$(stage_dir)"

  # Prefer snakeburger-cli-*.jar; fall back to any jar if needed.
  local jar
  jar="$(ls -1 "$stage/app/lib"/snakeburger-cli-*.jar 2>/dev/null | head -n 1 || true)"
  if [[ -z "$jar" ]]; then
    jar="$(ls -1 "$stage/app/lib"/*.jar 2>/dev/null | head -n 1 || true)"
  fi
  [[ -n "$jar" ]] || return 1
  echo "$jar"
}

derive_raw_version_from_jar() {
  local jar
  jar="$(basename "$1")"
  # Expect: snakeburger-cli-<version>.jar
  local v="${jar#snakeburger-cli-}"
  v="${v%.jar}"
  echo "$v"
}

sanitize_jpackage_version() {
  # jpackage wants 1-3 dot-separated integers, first must be >= 1.
  local raw="$1"

  # Strip any suffix like -SNAPSHOT
  raw="${raw%%-*}"

  # Keep only digits and dots
  raw="$(echo "$raw" | tr -cd '0-9.' )"
  [[ -n "$raw" ]] || raw="1.0.0"

  IFS='.' read -r a b c _rest <<<"$raw"
  a="${a:-1}"
  b="${b:-0}"
  c="${c:-0}"

  # First number cannot be 0.
  if [[ "$a" == "0" ]]; then
    a="1"
  fi

  echo "${a}.${b}.${c}"
}

app_version_raw() {
  if [[ -n "${SNAKEBURGER_APP_VERSION_RAW:-}" ]]; then
    echo "$SNAKEBURGER_APP_VERSION_RAW"
    return 0
  fi
  local jar
  jar="$(main_jar_path)" || return 1
  derive_raw_version_from_jar "$jar"
}

app_version_jpackage() {
  local raw
  raw="$(app_version_raw)" || return 1
  sanitize_jpackage_version "$raw"
}

build_installer_from_stage() {
  local type="$1" # deb|rpm
  local root stage jar jar_base runtime jp dest app_name pkg_name v_raw v_pkg

  root="$(repo_root)"
  stage="$(stage_dir)"
  [[ -d "$stage" ]] || die "Stage not found at: $stage"

  jar="$(main_jar_path)" || die "Could not locate main jar under: $stage/app/lib"
  jar_base="$(basename "$jar")"
  runtime="$stage/runtime"

  jp="$(jpackage_bin)" || die "Could not find 'jpackage'. Install a JDK that includes it, or set SNAKEBURGER_JPACKAGE_BIN."
  ensure_main_class
  ensure_app_name
  ensure_pkg_name

  app_name="$SNAKEBURGER_APP_NAME"
  pkg_name="$SNAKEBURGER_PKG_NAME"
  v_raw="$(app_version_raw)"
  v_pkg="$(app_version_jpackage)"

  dest="$root/snakeburger-cli/build/snakeburger-cli-jpackage-${type}"
  rm -rf "$dest"
  mkdir -p "$dest"

  echo "Using jpackage: $jp"
  echo "App name: $app_name"
  echo "Package name: $pkg_name"
  echo "Main jar: $jar_base"
  echo "Main class: $SNAKEBURGER_MAIN_CLASS"
  echo "App version (raw): $v_raw"
  echo "App version (jpackage): $v_pkg"
  echo "Dest: $dest"

  # jpackage input dir contains the main jar and dependency jars.
  if [[ "$type" == "rpm" ]]; then
  # jpackage can warn if it has to derive an installer from a non-jpackage app-image.
  # Build a minimal app-image via jpackage first (reusing the staged Babylon runtime),
  # then build the RPM from that app-image so output stays clean.
  local jp_appimage_root="$root/snakeburger-cli/build/snakeburger-cli-jpackage-appimage"
  rm -rf "$jp_appimage_root"
  mkdir -p "$jp_appimage_root"

  echo "Preparing jpackage app-image for RPM..."
  "$jp" \
    --type app-image \
    --dest "$jp_appimage_root" \
    --name "$app_name" \
    --app-version "$v_pkg" \
    --input "$stage/app/lib" \
    --main-jar "$jar_base" \
    --main-class "$SNAKEBURGER_MAIN_CLASS" \
    --runtime-image "$stage/runtime" \
    --verbose

  local jp_app_image
  jp_app_image="$(find "$jp_appimage_root" -maxdepth 1 -mindepth 1 -type d | head -n 1 || true)"
  if [[ -z "$jp_app_image" ]]; then
    die "jpackage app-image did not produce output under: $jp_appimage_root"
  fi

  "$jp" \
    --type "$type" \
    --dest "$dest" \
    --name "$app_name" \
    --app-version "$v_pkg" \
    --app-image "$jp_app_image" \
    --linux-package-name "$pkg_name" \
    --verbose
else
  "$jp" \
    --type "$type" \
    --dest "$dest" \
    --name "$app_name" \
    --app-version "$v_pkg" \
    --input "$stage/app/lib" \
    --main-jar "$jar_base" \
    --main-class "$SNAKEBURGER_MAIN_CLASS" \
    --runtime-image "$stage/runtime" \
    --linux-package-name "$pkg_name" \
    --verbose
fi

  local out
  out="$(ls -1 "$dest"/*.${type} 2>/dev/null | head -n 1 || true)"
  if [[ -z "$out" ]]; then
    die "Expected .${type} output under: $dest"
  fi

  mkdir -p "$root/snakeburger-cli/build/distributions"
  cp -f "$out" "$root/snakeburger-cli/build/distributions/"

  echo "Wrote: $out"
  echo "Copied to: $root/snakeburger-cli/build/distributions/$(basename "$out")"
}
