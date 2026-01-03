#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$here/_common-linux-jpackage.sh"

# RPM package prerequisites
if ! command -v rpmbuild >/dev/null 2>&1; then
  echo "ERROR: missing required command: rpmbuild" >&2
  echo "Install on Ubuntu/Debian with: sudo apt-get install -y rpm" >&2
  exit 1
fi

# Build the stage image (reusable for installers)
build_stage

# Build .rpm installer from stage/runtime
build_installer_from_stage "rpm"
