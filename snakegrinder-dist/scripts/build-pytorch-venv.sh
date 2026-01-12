#!/bin/bash
#
# Build PyTorch from source for GraalPy
#
# This script creates a virtualenv with PyTorch built specifically for GraalPy.
# PyTorch wheels are not available for GraalPy, so we must build from source.
#
# Usage:
#   ./build-pytorch-venv.sh           # Build with defaults from versions.env
#   ./build-pytorch-venv.sh --clean   # Remove existing venv first
#
# The script will:
# 1. Check/install dependencies (with clear apt/brew instructions if missing)
# 2. Download GraalPy if not present
# 3. Create virtualenv and build PyTorch from source using GraalPy's built-in
#    auto-patching and patch system
#
# Environment variables (override versions.env):
#   GRAALPY_HOME     - Path to existing GraalPy installation
#   VENV_DIR         - Path to create the venv
#   GRAALPY_VERSION  - GraalPy version to download
#   PYTORCH_VERSION  - PyTorch version to build
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "${SCRIPT_DIR}")"

# Source version configuration
if [ -f "${DIST_DIR}/versions.env" ]; then
    source "${DIST_DIR}/versions.env"
fi

# Allow environment overrides
GRAALPY_VERSION="${GRAALPY_VERSION:-25.0.1}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# ============================================================================
# MEMORY MANAGEMENT - Prevent OOM on memory-constrained systems
# ============================================================================
# These must be set EARLY, before any pip/cmake operations
# All three flags are needed because different build phases check different vars

# Calculate job limit based on available RAM (2GB per job minimum)
TOTAL_RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "16")
CALCULATED_JOBS=$((TOTAL_RAM_GB / 2))
# Clamp between 2 and 8 jobs
if [ "${CALCULATED_JOBS}" -lt 2 ]; then
    CALCULATED_JOBS=2
elif [ "${CALCULATED_JOBS}" -gt 8 ]; then
    CALCULATED_JOBS=8
fi
PARALLEL_JOBS="${PARALLEL_JOBS:-${CALCULATED_JOBS}}"

# Flag 1: PyTorch's own parallel job limit
export MAX_JOBS="${PARALLEL_JOBS}"
# Flag 2: CMake's parallel build level
export CMAKE_BUILD_PARALLEL_LEVEL="${PARALLEL_JOBS}"
# Flag 3: Make/ninja parallel jobs via MAKEFLAGS
export MAKEFLAGS="-j${PARALLEL_JOBS}"

echo "Memory management: limiting parallel jobs to ${PARALLEL_JOBS} (based on ${TOTAL_RAM_GB}GB RAM)"

# ============================================================================
# PLATFORM DETECTION
# ============================================================================

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

case "${UNAME_S}" in
    Darwin)
        PLATFORM_OS="macos"
        PKG_MANAGER="brew"
        case "${UNAME_M}" in
            arm64)  PLATFORM_ARCH="aarch64"; GRAALPY_PLATFORM="macos-aarch64" ;;
            x86_64) PLATFORM_ARCH="amd64";   GRAALPY_PLATFORM="macos-amd64" ;;
            *) echo "ERROR: Unsupported macOS architecture: ${UNAME_M}"; exit 1 ;;
        esac
        ;;
    Linux)
        PLATFORM_OS="linux"
        PKG_MANAGER="apt"
        case "${UNAME_M}" in
            x86_64)  PLATFORM_ARCH="amd64";   GRAALPY_PLATFORM="linux-amd64" ;;
            aarch64) PLATFORM_ARCH="aarch64"; GRAALPY_PLATFORM="linux-aarch64" ;;
            *) echo "ERROR: Unsupported Linux architecture: ${UNAME_M}"; exit 1 ;;
        esac
        ;;
    *)
        echo "ERROR: Unsupported OS: ${UNAME_S}"
        exit 1
        ;;
esac

# Set paths
if [ -z "${GRAALPY_HOME}" ]; then
    GRAALPY_HOME="${DIST_DIR}/tools/graalpy-${GRAALPY_VERSION}-${GRAALPY_PLATFORM}"
fi
VENV_DIR="${VENV_DIR:-${DIST_DIR}/.pytorch-venv}"
GRAALPY_DOWNLOAD_URL="https://github.com/oracle/graalpython/releases/download/graal-${GRAALPY_VERSION}/graalpy-${GRAALPY_VERSION}-${GRAALPY_PLATFORM}.tar.gz"

# Resolve absolute path for VENV_DIR
mkdir -p "$(dirname "${VENV_DIR}")"
VENV_DIR="$(cd "$(dirname "${VENV_DIR}")" && pwd)/$(basename "${VENV_DIR}")"

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

CLEAN_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN_BUILD=1 ;;
        --help|-h)
            echo "Usage: $0 [--clean]"
            echo ""
            echo "Options:"
            echo "  --clean    Remove existing venv before building"
            echo ""
            echo "Configuration (from versions.env or environment):"
            echo "  GRAALPY_VERSION=${GRAALPY_VERSION}"
            echo "  PYTORCH_VERSION=${PYTORCH_VERSION}"
            echo "  PYTHON_VERSION=${PYTHON_VERSION}"
            exit 0
            ;;
    esac
done

if [ "${CLEAN_BUILD}" = "1" ] && [ -d "${VENV_DIR}" ]; then
    echo "Removing existing venv: ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

# ============================================================================
# DEPENDENCY CHECKING
# ============================================================================

echo "=== SnakeGrinder PyTorch Build ==="
echo ""
echo "Platform:       ${PLATFORM_OS}-${PLATFORM_ARCH}"
echo "GraalPy:        ${GRAALPY_VERSION}"
echo "PyTorch:        ${PYTORCH_VERSION}"
echo "Python:         ${PYTHON_VERSION}"
echo ""

check_dependency() {
    local cmd="$1"
    local brew_pkg="$2"
    local apt_pkg="$3"
    local optional="$4"

    if command -v "$cmd" &> /dev/null; then
        echo "  [OK] $cmd"
        return 0
    else
        if [ "$optional" = "optional" ]; then
            echo "  [--] $cmd (optional, not found)"
            return 1
        else
            echo "  [MISSING] $cmd"
            if [ "${PKG_MANAGER}" = "brew" ]; then
                MISSING_BREW="${MISSING_BREW} ${brew_pkg}"
            else
                MISSING_APT="${MISSING_APT} ${apt_pkg}"
            fi
            return 1
        fi
    fi
}

echo "Checking dependencies..."
MISSING_BREW=""
MISSING_APT=""

check_dependency "cmake" "cmake" "cmake"
check_dependency "make" "make" "build-essential"
check_dependency "g++" "gcc" "build-essential"
check_dependency "ninja" "ninja" "ninja-build" "optional" && USE_NINJA=1 || USE_NINJA=0
check_dependency "curl" "curl" "curl"
check_dependency "tar" "gnutar" "tar"

# Platform-specific checks
if [ "${PLATFORM_OS}" = "macos" ]; then
    if ! xcode-select -p &> /dev/null; then
        echo "  [MISSING] Xcode Command Line Tools"
        echo ""
        echo "ERROR: Xcode Command Line Tools required. Install with:"
        echo "  xcode-select --install"
        exit 1
    else
        echo "  [OK] Xcode Command Line Tools"
    fi
fi

# Report missing dependencies
if [ -n "${MISSING_BREW}" ] || [ -n "${MISSING_APT}" ]; then
    echo ""
    echo "ERROR: Missing required dependencies."
    echo ""
    if [ "${PKG_MANAGER}" = "brew" ]; then
        echo "Install with Homebrew:"
        echo "  brew install${MISSING_BREW}"
    else
        echo "Install with apt:"
        echo "  sudo apt update && sudo apt install -y${MISSING_APT}"
    fi
    echo ""
    exit 1
fi

echo ""

# ============================================================================
# GRAALPY INSTALLATION
# ============================================================================

if [ ! -x "${GRAALPY_HOME}/bin/graalpy" ]; then
    echo "GraalPy not found at ${GRAALPY_HOME}"
    echo "Downloading GraalPy ${GRAALPY_VERSION} for ${GRAALPY_PLATFORM}..."

    mkdir -p "$(dirname "${GRAALPY_HOME}")"

    TARBALL="/tmp/graalpy-${GRAALPY_VERSION}-${GRAALPY_PLATFORM}.tar.gz"
    if [ ! -f "${TARBALL}" ]; then
        echo "Downloading from: ${GRAALPY_DOWNLOAD_URL}"
        curl -L --progress-bar -o "${TARBALL}" "${GRAALPY_DOWNLOAD_URL}"
    else
        echo "Using cached download: ${TARBALL}"
    fi

    echo "Extracting to ${GRAALPY_HOME}..."
    tar -xzf "${TARBALL}" -C "$(dirname "${GRAALPY_HOME}")"

    if [ ! -x "${GRAALPY_HOME}/bin/graalpy" ]; then
        echo "ERROR: GraalPy extraction failed."
        echo "Expected: ${GRAALPY_HOME}/bin/graalpy"
        ls -la "$(dirname "${GRAALPY_HOME}")" 2>/dev/null || true
        exit 1
    fi

    echo "GraalPy ${GRAALPY_VERSION} installed successfully"
    echo ""
fi

echo "Using GraalPy: ${GRAALPY_HOME}"
"${GRAALPY_HOME}/bin/graalpy" --version
echo ""

# ============================================================================
# VIRTUALENV CREATION
# ============================================================================

if [ -d "${VENV_DIR}" ]; then
    echo "Using existing venv: ${VENV_DIR}"
else
    echo "Creating GraalPy virtualenv..."
    "${GRAALPY_HOME}/bin/graalpy" -m venv "${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Longer timeout for large downloads
export PIP_DEFAULT_TIMEOUT=600

echo "Upgrading pip and installing build tools..."
pip install --upgrade pip wheel setuptools

# ============================================================================
# PYTORCH BUILD - Using GraalPy's built-in patching system
# ============================================================================
#
# GraalPy automatically handles PyTorch compatibility through:
# 1. autopatch_capi.py - Converts ->ob_type to Py_TYPE(), etc.
# 2. torch-X.Y.Z.patch - Official GraalPy patches from GitHub repository
#
# The patches are fetched from:
#   https://raw.githubusercontent.com/oracle/graalpython/refs/heads/github/patches/
#
# This is the officially supported way to build PyTorch for GraalPy.
# See: https://github.com/oracle/graalpython/issues/589
#

echo ""
echo "Installing build dependencies..."
pip install pyyaml typing_extensions

echo ""
echo "Configuring PyTorch build..."

# Use system cmake (pip cmake wrapper is broken on GraalPy)
export CMAKE_COMMAND=$(which cmake)
echo "Using system cmake: ${CMAKE_COMMAND}"

# CPU-only build configuration
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=1
export USE_OPENMP=1
export BUILD_TEST=0

# Note: MAX_JOBS, CMAKE_BUILD_PARALLEL_LEVEL, and MAKEFLAGS are set at script start

if [ "$USE_NINJA" = "1" ]; then
    export CMAKE_GENERATOR=Ninja
    echo "Using Ninja for faster builds"
fi

echo ""
echo "Building PyTorch ${PYTORCH_VERSION} from source..."
echo "GraalPy will automatically apply compatibility patches."
echo "Started at: $(date)"

# Use the officially supported pip install method
# --no-binary torch  : Force build from source (no pre-built wheel)
# --no-cache         : Ensure fresh download with latest patches
# -v                 : Verbose output to see patching progress
pip install "torch==${PYTORCH_VERSION}" --no-binary torch --no-cache -v

echo "Completed at: $(date)"

# Install sympy (required for torch.fx shape propagation)
echo ""
echo "Installing sympy..."
pip install sympy mpmath

# ============================================================================
# VERIFICATION
# ============================================================================

cd "${VENV_DIR}"
echo ""
echo "Verifying installation..."

# Set library path
if [ "${PLATFORM_OS}" = "macos" ]; then
    export DYLD_LIBRARY_PATH="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/torch/lib:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/torch/lib:${LD_LIBRARY_PATH}"
fi

python -c "
import torch
print('torch version:', torch.__version__)
from torch.fx import symbolic_trace
print('torch.fx.symbolic_trace: OK')
import torch.nn as nn
print('torch.nn: OK')
"

echo ""
echo "=== PyTorch build complete ==="
echo ""
echo "Venv: ${VENV_DIR}"
echo ""
echo "Next steps:"
echo "  1. Prune venv:  VENV_DIR=${VENV_DIR} ./scripts/prune-venv.sh"
echo "  2. Build dist:  ./gradlew :snakegrinder-dist:assembleDist"
echo ""
