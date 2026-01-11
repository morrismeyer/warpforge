#!/bin/bash
#
# Build a macOS .app bundle for SnakeGrinder
#
# This script creates a standard macOS application bundle containing:
# - Native image binary
# - PyTorch venv with native libraries
# - Wrapper script that sets up library paths
#
# Unlike snakeburger (which uses jpackage), snakegrinder is a native-image
# binary so we construct the .app bundle manually.
#
# Usage:
#   ./build-app.sh              # Build .app from assembleDist output
#   ./build-app.sh --from-dist  # Same as default
#
# Prerequisites:
#   - Run ./gradlew :snakegrinder-dist:assembleDist first
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "${SCRIPT_DIR}")"
ROOT="$(dirname "${DIST_DIR}")"

# Source version configuration
if [ -f "${DIST_DIR}/versions.env" ]; then
    source "${DIST_DIR}/versions.env"
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Configuration
APP_NAME="${SNAKEGRINDER_APP_NAME:-SnakeGrinder}"
APP_VERSION="${SNAKEGRINDER_APP_VERSION:-0.1.0}"
BUNDLE_ID="${SNAKEGRINDER_BUNDLE_ID:-io.surfworks.snakegrinder}"

# Input: assembled distribution
ASSEMBLED_DIST="${DIST_DIR}/build/dist"

# Output: .app bundle
BUILD_DIR="${DIST_DIR}/build"
APP_BUNDLE="${BUILD_DIR}/${APP_NAME}.app"

# ============================================================================
# VALIDATION
# ============================================================================

echo "=== Building ${APP_NAME}.app ==="
echo ""

if [[ ! -d "${ASSEMBLED_DIST}" ]]; then
    echo "ERROR: Assembled distribution not found at: ${ASSEMBLED_DIST}"
    echo ""
    echo "Run first:"
    echo "  ./gradlew :snakegrinder-dist:assembleDist"
    exit 1
fi

if [[ ! -f "${ASSEMBLED_DIST}/bin/snakegrinder-bin" ]]; then
    echo "ERROR: Native binary not found at: ${ASSEMBLED_DIST}/bin/snakegrinder-bin"
    exit 1
fi

# ============================================================================
# CREATE APP BUNDLE STRUCTURE
# ============================================================================

echo "Creating app bundle structure..."
rm -rf "${APP_BUNDLE}"
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources"

# ============================================================================
# INFO.PLIST
# ============================================================================

echo "Writing Info.plist..."
cat > "${APP_BUNDLE}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>snakegrinder</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${APP_VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2025 SurfWorks. All rights reserved.</string>
</dict>
</plist>
EOF

# ============================================================================
# COPY RESOURCES
# ============================================================================

echo "Copying resources..."

# Copy the venv to Resources
if [[ -d "${ASSEMBLED_DIST}/venv" ]]; then
    cp -R "${ASSEMBLED_DIST}/venv" "${APP_BUNDLE}/Contents/Resources/"
else
    echo "WARNING: venv not found in assembled distribution"
fi

# Copy the native binary to Resources
cp "${ASSEMBLED_DIST}/bin/snakegrinder-bin" "${APP_BUNDLE}/Contents/Resources/"
chmod +x "${APP_BUNDLE}/Contents/Resources/snakegrinder-bin"

# ============================================================================
# CREATE LAUNCHER SCRIPT
# ============================================================================

echo "Creating launcher script..."

# The launcher script in Contents/MacOS sets up paths and runs the binary
cat > "${APP_BUNDLE}/Contents/MacOS/snakegrinder" << 'LAUNCHER'
#!/bin/bash
#
# SnakeGrinder launcher - sets up library paths for PyTorch native libs
#

# Resolve the Contents directory
CONTENTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

# Path to PyTorch native libraries
TORCH_LIB="${RESOURCES_DIR}/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib"

# Set library path for native PyTorch libs
export DYLD_LIBRARY_PATH="${TORCH_LIB}:${DYLD_LIBRARY_PATH:-}"

# Set PYTORCH_VENV for Java code
export PYTORCH_VENV="${RESOURCES_DIR}/venv"

# Run the native image
exec "${RESOURCES_DIR}/snakegrinder-bin" "$@"
LAUNCHER

# Substitute Python version in the launcher
sed -i '' "s/\${PYTHON_VERSION}/${PYTHON_VERSION}/g" "${APP_BUNDLE}/Contents/MacOS/snakegrinder"
chmod +x "${APP_BUNDLE}/Contents/MacOS/snakegrinder"

# ============================================================================
# OPTIONAL: ICON
# ============================================================================

# Check for icon in known locations
ICON_LOCATIONS=(
    "${DIST_DIR}/resources/AppIcon.icns"
    "${ROOT}/resources/snakegrinder.icns"
)

for icon in "${ICON_LOCATIONS[@]}"; do
    if [[ -f "${icon}" ]]; then
        echo "Copying icon from: ${icon}"
        cp "${icon}" "${APP_BUNDLE}/Contents/Resources/AppIcon.icns"
        break
    fi
done

# ============================================================================
# CREATE ZIP ARCHIVE
# ============================================================================

echo "Creating zip archive..."
DIST_OUTPUT="${BUILD_DIR}/distributions"
mkdir -p "${DIST_OUTPUT}"

ZIP_NAME="snakegrinder-${APP_VERSION}-macos.zip"
ZIP_PATH="${DIST_OUTPUT}/${ZIP_NAME}"
rm -f "${ZIP_PATH}"

(
    cd "${BUILD_DIR}"
    /usr/bin/zip -qry "${ZIP_PATH}" "${APP_NAME}.app"
)

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=== Build Complete ==="
echo ""
echo "App bundle: ${APP_BUNDLE}"
echo "Zip archive: ${ZIP_PATH}"
echo ""
echo "To test:"
echo "  open ${APP_BUNDLE}"
echo ""
echo "Or from Terminal:"
echo "  ${APP_BUNDLE}/Contents/MacOS/snakegrinder --help"
echo ""
