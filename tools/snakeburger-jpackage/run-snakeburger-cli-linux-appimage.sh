#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the existing "app-image" runner so Linux can use the same stage zip.
# Usage:
#   tools/snakeburger-jpackage/run-snakeburger-cli-linux-appimage.sh --help

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "${ROOT}/tools/snakeburger-app-image/run-snakeburger-cli-appimage.sh" "$@"
