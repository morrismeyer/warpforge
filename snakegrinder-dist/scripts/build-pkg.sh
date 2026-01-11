#!/bin/bash
#
# Build a macOS PKG installer for SnakeGrinder
#
# Creates a signed installer package that:
# - Installs SnakeGrinder.app to /Applications
# - Optionally creates CLI symlink in /usr/local/bin
#
# Usage:
#   ./build-pkg.sh
#   ./build-pkg.sh --with-cli-link    # Also install /usr/local/bin/snakegrinder
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
BUNDLE_ID="${SNAKEGRINDER_BUNDLE_ID:-io.surfworks.snakegrinder}"
PKG_ID="${BUNDLE_ID}.pkg"

# Parse arguments
WITH_CLI_LINK=0
for arg in "$@"; do
    case "$arg" in
        --with-cli-link) WITH_CLI_LINK=1 ;;
        --help|-h)
            echo "Usage: $0 [--with-cli-link]"
            echo ""
            echo "Options:"
            echo "  --with-cli-link  Also install /usr/local/bin/snakegrinder symlink"
            exit 0
            ;;
    esac
done

# Paths
BUILD_DIR="${DIST_DIR}/build"
APP_BUNDLE="${BUILD_DIR}/${APP_NAME}.app"
PKG_ROOT="${BUILD_DIR}/pkg-root"
PKG_SCRIPTS="${BUILD_DIR}/pkg-scripts"
PKG_COMPONENT="${BUILD_DIR}/${APP_NAME}-component.pkg"
PKG_FINAL="${BUILD_DIR}/distributions/${APP_NAME}-${APP_VERSION}.pkg"

# ============================================================================
# BUILD APP BUNDLE IF NEEDED
# ============================================================================

echo "=== Building ${APP_NAME} PKG ==="
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
# PREPARE PKG ROOT
# ============================================================================

echo "Preparing package root..."
rm -rf "${PKG_ROOT}" "${PKG_SCRIPTS}"
mkdir -p "${PKG_ROOT}/Applications"
mkdir -p "${PKG_SCRIPTS}"

# Copy app to pkg root (will be installed to /Applications)
cp -R "${APP_BUNDLE}" "${PKG_ROOT}/Applications/"

# ============================================================================
# CREATE POSTINSTALL SCRIPT
# ============================================================================

if [[ "${WITH_CLI_LINK}" == "1" ]]; then
    echo "Including CLI symlink installation..."

    cat > "${PKG_SCRIPTS}/postinstall" << 'EOF'
#!/bin/bash
# Post-installation script for SnakeGrinder
# Creates a symlink in /usr/local/bin for CLI access

CLI_LINK="/usr/local/bin/snakegrinder"
APP_BINARY="/Applications/SnakeGrinder.app/Contents/MacOS/snakegrinder"

# Create /usr/local/bin if it doesn't exist
mkdir -p /usr/local/bin

# Remove existing link if present
rm -f "${CLI_LINK}"

# Create symlink
ln -s "${APP_BINARY}" "${CLI_LINK}"

echo "Installed CLI symlink: ${CLI_LINK}"
exit 0
EOF
    chmod +x "${PKG_SCRIPTS}/postinstall"
fi

# ============================================================================
# BUILD COMPONENT PKG
# ============================================================================

echo "Building component package..."
mkdir -p "$(dirname "${PKG_FINAL}")"
rm -f "${PKG_COMPONENT}" "${PKG_FINAL}"

PKGBUILD_ARGS=(
    --root "${PKG_ROOT}"
    --identifier "${PKG_ID}"
    --version "${APP_VERSION}"
    --install-location "/"
)

if [[ "${WITH_CLI_LINK}" == "1" ]]; then
    PKGBUILD_ARGS+=(--scripts "${PKG_SCRIPTS}")
fi

pkgbuild "${PKGBUILD_ARGS[@]}" "${PKG_COMPONENT}"

# ============================================================================
# BUILD PRODUCT PKG (distribution)
# ============================================================================

echo "Building distribution package..."

# Create distribution.xml
DIST_XML="${BUILD_DIR}/distribution.xml"
cat > "${DIST_XML}" << EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>${APP_NAME}</title>
    <organization>${BUNDLE_ID}</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="false" hostArchitectures="x86_64,arm64"/>

    <welcome file="welcome.html"/>
    <license file="license.html"/>
    <conclusion file="conclusion.html"/>

    <choices-outline>
        <line choice="default">
            <line choice="${PKG_ID}"/>
        </line>
    </choices-outline>

    <choice id="default"/>
    <choice id="${PKG_ID}" visible="false">
        <pkg-ref id="${PKG_ID}"/>
    </choice>

    <pkg-ref id="${PKG_ID}" version="${APP_VERSION}" onConclusion="none">$(basename "${PKG_COMPONENT}")</pkg-ref>
</installer-gui-script>
EOF

# Create resources directory with HTML files
PKG_RESOURCES="${BUILD_DIR}/pkg-resources"
rm -rf "${PKG_RESOURCES}"
mkdir -p "${PKG_RESOURCES}"

# Welcome page
cat > "${PKG_RESOURCES}/welcome.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }
        h1 { color: #333; }
        p { color: #666; line-height: 1.6; }
    </style>
</head>
<body>
    <h1>Welcome to ${APP_NAME}</h1>
    <p>This installer will install ${APP_NAME} version ${APP_VERSION} on your computer.</p>
    <p>${APP_NAME} is a PyTorch model tracing tool that exports to StableHLO format.</p>
    <p>Click Continue to proceed with the installation.</p>
</body>
</html>
EOF

# License page
cat > "${PKG_RESOURCES}/license.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }
        h1 { color: #333; }
        pre { background: #f5f5f5; padding: 15px; overflow: auto; font-size: 12px; }
    </style>
</head>
<body>
    <h1>License Agreement</h1>
    <p>By installing this software, you agree to the following terms:</p>
    <pre>
MIT License

Copyright (c) 2025 SurfWorks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
    </pre>
</body>
</html>
EOF

# Conclusion page
if [[ "${WITH_CLI_LINK}" == "1" ]]; then
    CLI_NOTE="<p>A command-line shortcut has been installed at <code>/usr/local/bin/snakegrinder</code>.</p>"
else
    CLI_NOTE="<p>To use from Terminal, run:<br><code>/Applications/${APP_NAME}.app/Contents/MacOS/snakegrinder</code></p>"
fi

cat > "${PKG_RESOURCES}/conclusion.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }
        h1 { color: #333; }
        p { color: #666; line-height: 1.6; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Installation Complete</h1>
    <p>${APP_NAME} has been installed to <code>/Applications/${APP_NAME}.app</code>.</p>
    ${CLI_NOTE}
    <p>For more information, visit the project documentation.</p>
</body>
</html>
EOF

# Build the distribution package
productbuild \
    --distribution "${DIST_XML}" \
    --resources "${PKG_RESOURCES}" \
    --package-path "${BUILD_DIR}" \
    "${PKG_FINAL}"

# ============================================================================
# CLEANUP
# ============================================================================

rm -rf "${PKG_ROOT}" "${PKG_SCRIPTS}" "${PKG_RESOURCES}" "${PKG_COMPONENT}" "${DIST_XML}"

# ============================================================================
# SUMMARY
# ============================================================================

PKG_SIZE_MB=$(du -m "${PKG_FINAL}" | cut -f1)

echo ""
echo "=== PKG Build Complete ==="
echo ""
echo "PKG: ${PKG_FINAL}"
echo "Size: ${PKG_SIZE_MB} MB"
if [[ "${WITH_CLI_LINK}" == "1" ]]; then
    echo "CLI link: Will install /usr/local/bin/snakegrinder"
fi
echo ""
echo "To test:"
echo "  open ${PKG_FINAL}"
echo ""
