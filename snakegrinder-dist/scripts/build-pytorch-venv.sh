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
# 3. Create virtualenv and build PyTorch from source
# 4. Apply GraalPy compatibility patches
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
# CHECK GRAALPY PYTORCH PATCH AVAILABILITY
# ============================================================================

PATCH_DIR="${GRAALPY_HOME}/lib/graalpy${GRAALPY_VERSION%%.*}.${GRAALPY_VERSION#*.}/patches"
PYTORCH_PATCH="${PATCH_DIR}/torch-${PYTORCH_VERSION}.patch"

if [ ! -f "${PYTORCH_PATCH}" ]; then
    echo "WARNING: GraalPy patch for PyTorch ${PYTORCH_VERSION} not found at:"
    echo "  ${PYTORCH_PATCH}"
    echo ""
    echo "Available patches:"
    ls -1 "${PATCH_DIR}"/torch-*.patch 2>/dev/null || echo "  (none found)"
    echo ""
    echo "The build may fail or produce incompatible binaries."
    echo "Consider using a PyTorch version that has a matching patch."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "GraalPy patch found: torch-${PYTORCH_VERSION}.patch"
    echo ""
fi

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
# PYTORCH SOURCE DOWNLOAD AND PATCHING
# ============================================================================

PYTORCH_SRC="${VENV_DIR}/pytorch-src"
if [ ! -d "${PYTORCH_SRC}" ]; then
    echo ""
    echo "Downloading PyTorch ${PYTORCH_VERSION} source from PyPI..."
    cd "${VENV_DIR}"

    pip download --no-binary :all: --no-deps "torch==${PYTORCH_VERSION}" -d .

    TARBALL=$(ls torch-${PYTORCH_VERSION}*.tar.gz 2>/dev/null | head -1)
    if [ -z "${TARBALL}" ]; then
        echo "ERROR: Could not find downloaded PyTorch tarball"
        exit 1
    fi

    # Get extracted directory name
    EXTRACTED_DIR=$(tar -tzf "${TARBALL}" | head -1 | cut -d'/' -f1)
    echo "Extracting ${TARBALL}..."

    [ -d "${EXTRACTED_DIR}" ] && rm -rf "${EXTRACTED_DIR}"
    tar -xzf "${TARBALL}"
    mv "${EXTRACTED_DIR}" "${PYTORCH_SRC}"

    cd "${PYTORCH_SRC}"
    echo "Source extracted to: ${PYTORCH_SRC}"

    # Apply GraalPy patch
    echo ""
    echo "Applying GraalPy compatibility patch..."
    if [ -f "${PYTORCH_PATCH}" ]; then
        if [ -f "torch/csrc/Generator.cpp" ]; then
            patch -p1 --force < "${PYTORCH_PATCH}" || true

            # Apply supplementary fixes for issues where main patch doesn't apply cleanly
            echo "Applying supplementary GraalPy fixes..."

            # Fix 1: pybind11 docstring handling
            PYBIND_FILE="third_party/pybind11/include/pybind11/pybind11.h"
            if grep -q "func->m_ml->ml_doc" "${PYBIND_FILE}" 2>/dev/null; then
                echo "  Fixing pybind11 docstring handling..."
                python3 << 'PYFIX'
with open("third_party/pybind11/include/pybind11/pybind11.h", "r") as f:
    content = f.read()
old_code = '''        std::free(const_cast<char *>(func->m_ml->ml_doc));
        // Install docstring if it's non-empty (when at least one option is enabled)
        func->m_ml->ml_doc
            = signatures.empty() ? nullptr : PYBIND11_COMPAT_STRDUP(signatures.c_str());'''
new_code = '''#if defined(GRAALVM_PYTHON)
        std::free(const_cast<char *>(GraalPyCFunction_GetDoc(m_ptr)));
        GraalPyCFunction_SetDoc(
            m_ptr, signatures.empty() ? nullptr : PYBIND11_COMPAT_STRDUP(signatures.c_str()));
#else
        std::free(const_cast<char *>(func->m_ml->ml_doc));
        // Install docstring if it's non-empty (when at least one option is enabled)
        func->m_ml->ml_doc
            = signatures.empty() ? nullptr : PYBIND11_COMPAT_STRDUP(signatures.c_str());
#endif'''
if old_code in content:
    content = content.replace(old_code, new_code)
    with open("third_party/pybind11/include/pybind11/pybind11.h", "w") as f:
        f.write(content)
    print("    pybind11 fix applied")
else:
    print("    pybind11 already patched or pattern not found")
PYFIX
            fi

            # Fix 2: Module.cpp docstring handling
            MODULE_FILE="torch/csrc/Module.cpp"
            if grep -q "f->m_ml->ml_doc" "${MODULE_FILE}" 2>/dev/null; then
                echo "  Fixing Module.cpp docstring handling..."
                python3 << 'PYFIX'
with open("torch/csrc/Module.cpp", "r") as f:
    content = f.read()
old_pattern = '''  if (Py_TYPE(obj) == &PyCFunction_Type) {
    PyCFunctionObject* f = (PyCFunctionObject*)obj;
    if (f->m_ml->ml_doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "function '%s' already has a docstring",
          f->m_ml->ml_name);
    }
    f->m_ml->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = (PyMethodDescrObject*)obj;
    if (m->d_method->ml_doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "method '%s' already has a docstring",
          m->d_method->ml_name);
    }
    m->d_method->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    PyGetSetDescrObject* m = (PyGetSetDescrObject*)obj;
    if (m->d_getset->doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "attribute '%s' already has a docstring",
          m->d_getset->name);
    }
    m->d_getset->doc = doc_str;
  } else if (Py_TYPE(obj) == &PyType_Type) {
    PyTypeObject* t = (PyTypeObject*)obj;
    if (t->tp_doc) {
      return PyErr_Format(
          PyExc_RuntimeError, "Type '%s' already has a docstring", t->tp_name);
    }
    t->tp_doc = doc_str;
  } else {
    return PyErr_Format(
        PyExc_TypeError,
        "don't know how to add docstring to type '%s'",
        Py_TYPE(obj)->tp_name);
  }'''
new_code = '''  // GraalPy change - use generic PyObject_GetDoc/SetDoc API
  if (PyObject_GetDoc(obj)) {
    return PyErr_Format(
        PyExc_RuntimeError,
        "object '%100R' already has a docstring",
        obj);
  }
  // GraalPy change
  if (PyObject_SetDoc(obj, doc_str) < 0) {
      return NULL;
  }'''
if old_pattern in content:
    content = content.replace(old_pattern, new_code)
    with open("torch/csrc/Module.cpp", "w") as f:
        f.write(content)
    print("    Module.cpp fix applied")
else:
    print("    Module.cpp already patched or pattern not found")
PYFIX
            fi

            # Clean up patch artifacts
            find . -name "*.rej" -delete 2>/dev/null || true
            find . -name "*.orig" -delete 2>/dev/null || true
            echo "Patch application complete"
        else
            echo "ERROR: PyTorch source structure unexpected"
            exit 1
        fi
    else
        echo "WARNING: No GraalPy patch found, building unpatched"
    fi
else
    echo "Using existing PyTorch source at ${PYTORCH_SRC}"
    cd "${PYTORCH_SRC}"
fi

# ============================================================================
# PYTORCH BUILD
# ============================================================================

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

if [ "$USE_NINJA" = "1" ]; then
    export CMAKE_GENERATOR=Ninja
    echo "Using Ninja for faster builds"
fi

echo ""
echo "Building PyTorch (this takes 30-60 minutes on first build)..."
echo "Started at: $(date)"
pip install .
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
