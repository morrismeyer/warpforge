#!/bin/bash
#
# Build a macOS DMG installer for SnakeGrinder
#
# Creates a drag-and-drop DMG with:
# - SnakeGrinder.app
# - Alias to /Applications folder
#
# This script ONLY BUILDS the DMG. It does not mount it or copy anything
# into /Applications. No user interaction required.
#
# Usage:
#   ./build-dmg.sh
#
# Prerequisites:
#   - Run ./gradlew :snakegrinder-dist:assembleDist first
#   - Or: ./build-app.sh (this script will call it if needed)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "${SCRIPT_DIR}")"
ROOT="$(dirname "${DIST_DIR}")"

# Source version configuration
if [ -f "${DIST_DIR}/versions.env" ]; then
    source "${DIST_DIR}/versions.env"
fi

# Configuration
APP_NAME="${SNAKEGRINDER_APP_NAME:-SnakeGrinder}"
APP_VERSION="${SNAKEGRINDER_APP_VERSION:-0.1.0}"
VOLUME_NAME="${APP_NAME} ${APP_VERSION}"

# Paths
BUILD_DIR="${DIST_DIR}/build"
APP_BUNDLE="${BUILD_DIR}/${APP_NAME}.app"
DMG_STAGING="${BUILD_DIR}/dmg-staging"
DMG_FINAL="${BUILD_DIR}/distributions/${APP_NAME}-${APP_VERSION}.dmg"

# ============================================================================
# BUILD APP BUNDLE IF NEEDED
# ============================================================================

echo "=== Building ${APP_NAME} DMG ==="
echo ""

if [[ ! -d "${APP_BUNDLE}" ]]; then
    echo "App bundle not found, building..."
    "${SCRIPT_DIR}/build-app.sh"
fi

if [[ ! -d "${APP_BUNDLE}" ]]; then
    echo "ERROR: Failed to create app bundle at: ${APP_BUNDLE}"
    exit 1
fi

# ============================================================================
# PREPARE DMG STAGING AREA
# ============================================================================

echo "Preparing DMG staging area..."
rm -rf "${DMG_STAGING}"
mkdir -p "${DMG_STAGING}"

# Copy app bundle
cp -R "${APP_BUNDLE}" "${DMG_STAGING}/"

# Create Applications alias (symlink)
ln -s /Applications "${DMG_STAGING}/Applications"

# Optional: Add README
cat > "${DMG_STAGING}/README.txt" << 'EOF'
SnakeGrinder
============

Drag SnakeGrinder.app to the Applications folder to install.

To use from Terminal:
  /Applications/SnakeGrinder.app/Contents/MacOS/snakegrinder --help

Or add an alias to your shell profile:
  alias snakegrinder='/Applications/SnakeGrinder.app/Contents/MacOS/snakegrinder'

For more information, visit:
  https://github.com/surfworks/warpforge
EOF

# ============================================================================
# CREATE DMG (non-interactive)
# ============================================================================

echo "Creating DMG..."
mkdir -p "$(dirname "${DMG_FINAL}")"
rm -f "${DMG_FINAL}"

# Create compressed, read-only DMG directly from staging folder
# This avoids any mounting/unmounting or user interaction
hdiutil create \
    -srcfolder "${DMG_STAGING}" \
    -volname "${VOLUME_NAME}" \
    -fs HFS+ \
    -format UDZO \
    -imagekey zlib-level=9 \
    "${DMG_FINAL}"

# Clean up staging
rm -rf "${DMG_STAGING}"

# ============================================================================
# SUMMARY
# ============================================================================

DMG_SIZE_MB=$(du -m "${DMG_FINAL}" | cut -f1)

echo ""
echo "=== DMG Build Complete ==="
echo ""
echo "DMG: ${DMG_FINAL}"
echo "Size: ${DMG_SIZE_MB} MB"
echo ""
echo "To test:"
echo "  open ${DMG_FINAL}"
echo ""
