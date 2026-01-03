#!/usr/bin/env bash
set -euo pipefail

# Linux stage builder (Goal 1/2):
# - Reuses the snakeburger-app-image staging logic (runtime + app bits)
# - Produces a ZIP under snakeburger-cli/build/distributions/
#
# This does NOT attempt multi-distro packaging (.deb/.rpm). We'll add that later if needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Default main class for SnakeBurger CLI if not provided.
# Override with: export SNAKEBURGER_MAIN_CLASS=...
: "${SNAKEBURGER_MAIN_CLASS:=io.surfworks.snakeburger.cli.SnakeBurgerMain}"
export SNAKEBURGER_MAIN_CLASS

echo "Building SnakeBurger stage via snakeburger-app-image tooling..."
echo "Using SNAKEBURGER_MAIN_CLASS: ${SNAKEBURGER_MAIN_CLASS}"

exec "${ROOT_DIR}/tools/snakeburger-app-image/build-snakeburger-cli-appimage.sh"
