#!/bin/bash
#
# Check GraalPy repository for new PyTorch patch versions
#
# This script compares the PyTorch patches available in the GraalPy repo
# against what we have bundled, and alerts if newer versions are available.
#
# Usage:
#   ./check-graalpy-patches.sh           # Check and report
#   ./check-graalpy-patches.sh --ci      # Exit non-zero if updates available (for CI)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "${SCRIPT_DIR}")"
PATCHES_DIR="${SCRIPT_DIR}/patches"

# GraalPy patches URL (raw GitHub API)
GRAALPY_PATCHES_API="https://api.github.com/repos/oracle/graalpython/contents/graalpython/lib-graalpython/patches"

# Source our versions
if [ -f "${DIST_DIR}/versions.env" ]; then
    source "${DIST_DIR}/versions.env"
fi

CURRENT_PYTORCH="${PYTORCH_VERSION:-2.7.0}"

CI_MODE=0
if [ "$1" = "--ci" ]; then
    CI_MODE=1
fi

echo "=== GraalPy PyTorch Patch Checker ==="
echo ""
echo "Current bundled version: PyTorch ${CURRENT_PYTORCH}"
echo ""

# Fetch available patches from GraalPy repo
echo "Checking GraalPy repository..."
RESPONSE=$(curl -s "${GRAALPY_PATCHES_API}")

if [ -z "${RESPONSE}" ] || echo "${RESPONSE}" | grep -q "API rate limit"; then
    echo "ERROR: Failed to fetch from GitHub API (rate limited or network error)"
    exit 1
fi

# Extract torch patch versions (handle JSON with/without spaces)
AVAILABLE_PATCHES=$(echo "${RESPONSE}" | grep -oE '"name": *"torch-[^"]*\.patch"' | sed 's/"name": *"torch-//g; s/\.patch"//g' | sort -V)

if [ -z "${AVAILABLE_PATCHES}" ]; then
    echo "WARNING: No torch patches found in GraalPy repo"
    exit 1
fi

echo "Available PyTorch patches in GraalPy:"
for version in ${AVAILABLE_PATCHES}; do
    if [ "${version}" = "${CURRENT_PYTORCH}" ]; then
        echo "  ${version} (current)"
    else
        echo "  ${version}"
    fi
done
echo ""

# Find the newest version
NEWEST=$(echo "${AVAILABLE_PATCHES}" | tail -1)

# Compare versions (simple string compare works for semver)
version_gt() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ] && [ "$1" != "$2" ]
}

if version_gt "${NEWEST}" "${CURRENT_PYTORCH}"; then
    echo "UPDATE AVAILABLE: PyTorch ${NEWEST} patch is available (we have ${CURRENT_PYTORCH})"
    echo ""
    echo "To upgrade:"
    echo "  1. Download the patch from GraalPy repo:"
    echo "     curl -o ${PATCHES_DIR}/torch-${NEWEST}.patch \\"
    echo "       https://raw.githubusercontent.com/oracle/graalpython/master/graalpython/lib-graalpython/patches/torch-${NEWEST}.patch"
    echo ""
    echo "  2. Update versions.env:"
    echo "     PYTORCH_VERSION=\"${NEWEST}\""
    echo ""
    echo "  3. Rebuild:"
    echo "     ./gradlew :snakegrinder-dist:rebuildPytorchVenv"
    echo ""

    if [ "${CI_MODE}" = "1" ]; then
        exit 1
    fi
else
    echo "OK: We have the latest available PyTorch patch (${CURRENT_PYTORCH})"

    # Also check if there are newer PyTorch releases without patches
    echo ""
    echo "Note: PyTorch mainline is at 2.11+, but GraalPy only has patches up to ${NEWEST}"
    echo "Track: https://github.com/oracle/graalpython/issues for updates"
fi

# Show what patches we have bundled vs available
echo ""
echo "Bundled patches:"
ls -1 "${PATCHES_DIR}"/torch-*.patch 2>/dev/null | while read f; do
    basename "$f"
done || echo "  (none)"
