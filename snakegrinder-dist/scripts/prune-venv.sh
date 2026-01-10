#!/bin/bash
#
# Safe pruning of PyTorch venv for native-image distribution
#
# This script removes unnecessary files to reduce distribution size while
# preserving all functionality needed for torch.fx tracing and StableHLO export.
#
# Safe to remove:
# - networkx, pip, setuptools (not needed at runtime)
# - torch/include (C++ headers, only for compilation)
# - torch/_inductor, torch/_dynamo, torch/onnx (lazy-loaded, not used)
# - __pycache__, *.pyc, *.pyi (cache and type stubs)
#
# Must keep:
# - sympy, mpmath (required for torch.fx shape propagation)
# - torch/lib/*.so or *.dylib (native libraries)
#
# Environment variables:
# - VENV_DIR: Path to the venv to prune (required)
# - GRAALPY_HOME: Path to GraalPy installation (required for verification)
#

set -e

# Script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "${SCRIPT_DIR}")"

# Source version configuration
if [ -f "${DIST_DIR}/versions.env" ]; then
    source "${DIST_DIR}/versions.env"
fi

GRAALPY_VERSION="${GRAALPY_VERSION:-25.0.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Detect platform
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

case "${UNAME_S}" in
    Darwin)
        case "${UNAME_M}" in
            arm64) GRAALPY_PLATFORM="macos-aarch64" ;;
            x86_64) GRAALPY_PLATFORM="macos-amd64" ;;
        esac
        ;;
    Linux)
        case "${UNAME_M}" in
            x86_64) GRAALPY_PLATFORM="linux-amd64" ;;
            aarch64) GRAALPY_PLATFORM="linux-aarch64" ;;
        esac
        ;;
esac

VENV_DIR="${VENV_DIR:?VENV_DIR must be set}"
GRAALPY_HOME="${GRAALPY_HOME:-${DIST_DIR}/tools/graalpy-${GRAALPY_VERSION}-${GRAALPY_PLATFORM}}"

SITE_PACKAGES="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages"
TORCH_DIR="${SITE_PACKAGES}/torch"

if [ ! -d "${SITE_PACKAGES}" ]; then
    echo "ERROR: Site-packages not found at ${SITE_PACKAGES}"
    exit 1
fi

echo "=== Pruning PyTorch venv for distribution ==="
echo ""
echo "Venv: ${VENV_DIR}"
echo ""

INITIAL_SIZE=$(du -sm "${SITE_PACKAGES}" | cut -f1)
echo "Initial size: ${INITIAL_SIZE} MB"
echo ""

# ============================================================================
# PHASE 1: Remove unnecessary top-level packages (100% safe)
# ============================================================================
echo "Phase 1: Removing unnecessary packages..."

# Note: sympy and mpmath are REQUIRED for torch.fx shape propagation - DO NOT REMOVE
for pkg in networkx networkx-*.dist-info \
           pip pip-*.dist-info \
           setuptools setuptools-*.dist-info \
           pkg_resources _distutils_hack distutils-precedence.pth; do
    for item in "${SITE_PACKAGES}"/$pkg; do
        if [ -e "$item" ]; then
            echo "  Removing: $(basename "$item")"
            rm -rf "$item"
        fi
    done
done

# ============================================================================
# PHASE 2: Remove C++ headers and build artifacts (100% safe)
# ============================================================================
echo ""
echo "Phase 2: Removing C++ headers and build artifacts..."

for dir in include share; do
    if [ -d "${TORCH_DIR}/$dir" ]; then
        SIZE=$(du -sm "${TORCH_DIR}/$dir" | cut -f1)
        echo "  Removing: torch/$dir (${SIZE} MB)"
        rm -rf "${TORCH_DIR}/$dir"
    fi
done

# ============================================================================
# PHASE 3: Remove lazy-loaded torch modules (safe for FX tracing)
# These are only imported via __getattr__ when explicitly accessed
# ============================================================================
echo ""
echo "Phase 3: Removing lazy-loaded modules (not needed for FX tracing)..."

# Note: _export is needed by torch.quantization, keeping it
for mod in _inductor _dynamo onnx; do
    if [ -d "${TORCH_DIR}/$mod" ]; then
        SIZE=$(du -sm "${TORCH_DIR}/$mod" | cut -f1)
        echo "  Removing: torch/$mod (${SIZE} MB)"
        rm -rf "${TORCH_DIR}/$mod"
    fi
done

# ============================================================================
# PHASE 4: Remove cache and type stubs
# ============================================================================
echo ""
echo "Phase 4: Removing cache files and type stubs..."

find "${SITE_PACKAGES}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${SITE_PACKAGES}" -name "*.pyc" -delete 2>/dev/null || true
find "${SITE_PACKAGES}" -name "*.pyi" -delete 2>/dev/null || true
echo "  Done"

# ============================================================================
# RESULTS
# ============================================================================
echo ""
echo "=== Pruning Complete ==="

FINAL_SIZE=$(du -sm "${SITE_PACKAGES}" | cut -f1)
SAVED=$((INITIAL_SIZE - FINAL_SIZE))

echo ""
echo "Initial size: ${INITIAL_SIZE} MB"
echo "Final size:   ${FINAL_SIZE} MB"
echo "Saved:        ${SAVED} MB"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================
echo "Verifying torch + FX import..."

# Set library path based on platform
if [ "${UNAME_S}" = "Darwin" ]; then
    export DYLD_LIBRARY_PATH="${TORCH_DIR}/lib:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${TORCH_DIR}/lib:${LD_LIBRARY_PATH}"
fi

"${GRAALPY_HOME}/bin/graalpy" -c "
import sys
sys.path.insert(0, '${SITE_PACKAGES}')
import torch
print('torch version:', torch.__version__)
from torch.fx import symbolic_trace
print('torch.fx.symbolic_trace: OK')
import torch.nn as nn
print('torch.nn: OK')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS: Venv pruned and verified."
else
    echo ""
    echo "ERROR: Verification failed. The venv may be corrupted."
    exit 1
fi
