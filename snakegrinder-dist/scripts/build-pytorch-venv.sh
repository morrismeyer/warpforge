#!/bin/bash
#
# Build PyTorch from source for GraalPy
#
# This script creates a virtualenv with PyTorch built specifically for GraalPy.
# PyTorch wheels are not available for GraalPy, so we must build from source.
#
# Prerequisites:
# - GraalPy 25.0.1+ installed
# - cmake >= 3.18
# - C++ compiler (Xcode on macOS, g++ on Linux)
# - ninja (optional, speeds up build)
#
# Environment variables:
# - GRAALPY_HOME: Path to GraalPy installation (required)
# - VENV_DIR: Path to create the venv (required)
# - PYTORCH_VERSION: PyTorch version to build (default: 2.7.0)
#

set -e

# Configuration
GRAALPY_HOME="${GRAALPY_HOME:-/private/tmp/graalpy-25.0.1-macos-aarch64}"
VENV_DIR="${VENV_DIR:-$(dirname "$0")/../.pytorch-venv}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.0}"
PYTORCH_REPO="https://github.com/pytorch/pytorch.git"

# Resolve absolute path
VENV_DIR="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"

echo "=== Building PyTorch ${PYTORCH_VERSION} for GraalPy ==="
echo ""
echo "GraalPy:    ${GRAALPY_HOME}"
echo "Venv:       ${VENV_DIR}"
echo "PyTorch:    ${PYTORCH_VERSION}"
echo ""

# Check GraalPy exists
if [ ! -x "${GRAALPY_HOME}/bin/graalpy" ]; then
    echo "ERROR: GraalPy not found at ${GRAALPY_HOME}"
    echo "Please set GRAALPY_HOME to your GraalPy installation."
    exit 1
fi

# Check prerequisites
echo "Checking prerequisites..."
for cmd in cmake make; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "ERROR: $cmd not found. Please install it."
        exit 1
    fi
done

if command -v ninja &> /dev/null; then
    echo "  ninja: found (will use for faster builds)"
    USE_NINJA=1
else
    echo "  ninja: not found (using make - slower)"
    USE_NINJA=0
fi

# Create virtualenv
echo ""
echo "Creating GraalPy virtualenv..."
"${GRAALPY_HOME}/bin/graalpy" -m venv "${VENV_DIR}"

# Activate venv
source "${VENV_DIR}/bin/activate"

# Set longer timeout for large downloads (PyTorch is ~286MB)
export PIP_DEFAULT_TIMEOUT=600

# Upgrade pip and install setuptools
echo ""
echo "Upgrading pip and installing setuptools..."
pip install --upgrade pip wheel setuptools

# Download PyTorch source from PyPI (matches GraalPy patch expectations)
PYTORCH_SRC="${VENV_DIR}/pytorch-src"
if [ ! -d "${PYTORCH_SRC}" ]; then
    echo ""
    echo "Downloading PyTorch ${PYTORCH_VERSION} source from PyPI..."
    cd "${VENV_DIR}"

    # Download source distribution from PyPI
    pip download --no-binary :all: --no-deps torch==${PYTORCH_VERSION} -d .

    # Extract the tarball
    TARBALL=$(ls torch-${PYTORCH_VERSION}*.tar.gz 2>/dev/null | head -1)
    if [ -z "${TARBALL}" ]; then
        echo "ERROR: Could not find downloaded PyTorch tarball"
        exit 1
    fi

    # Get the top-level directory name from tarball before extracting
    echo "Inspecting tarball contents..."
    EXTRACTED_DIR=$(tar -tzf "${TARBALL}" | head -1 | cut -d'/' -f1)
    echo "Tarball extracts to: ${EXTRACTED_DIR}"

    # Remove any old extracted directory with the same name
    if [ -d "${EXTRACTED_DIR}" ]; then
        echo "Removing old ${EXTRACTED_DIR}..."
        rm -rf "${EXTRACTED_DIR}"
    fi

    echo "Extracting ${TARBALL}..."
    tar -xzf "${TARBALL}"

    if [ -d "${EXTRACTED_DIR}" ]; then
        echo "Found extracted directory: ${EXTRACTED_DIR}"
        mv "${EXTRACTED_DIR}" "${PYTORCH_SRC}"
    else
        echo "ERROR: Expected directory ${EXTRACTED_DIR} not found after extraction"
        echo "Directory contents:"
        ls -la
        exit 1
    fi

    cd "${PYTORCH_SRC}"
    echo "Now in directory: $(pwd)"
    echo "Contents: $(ls -la | head -10)"

    # Apply GraalPy compatibility patch
    echo ""
    echo "Applying GraalPy compatibility patch..."
    PATCH_FILE="${GRAALPY_HOME}/lib/graalpy25.0/patches/torch-${PYTORCH_VERSION}.patch"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SUPPLEMENTARY_PATCHES_DIR="${SCRIPT_DIR}/patches"

    if [ -f "${PATCH_FILE}" ]; then
        # Verify we can see expected files before patching
        if [ -f "torch/csrc/Generator.cpp" ]; then
            echo "Source structure verified, applying patch..."
            # Use --force to continue even if some hunks fail
            # The patch may not apply cleanly due to version differences
            patch -p1 --force < "${PATCH_FILE}" || true

            # Apply supplementary GraalPy fixes using inline modifications
            # These fix issues where the main GraalPy patch doesn't apply cleanly

            # Fix 1: pybind11 docstring handling - use GraalPy API instead of CPython struct access
            PYBIND_FILE="third_party/pybind11/include/pybind11/pybind11.h"
            if grep -q "func->m_ml->ml_doc" "${PYBIND_FILE}" 2>/dev/null; then
                echo "Applying GraalPy fix for pybind11 docstring handling..."
                python3 << 'PYFIX'
import re
with open("third_party/pybind11/include/pybind11/pybind11.h", "r") as f:
    content = f.read()
# Replace CPython struct access with GraalPy API
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
    print("  pybind11 fix applied successfully")
else:
    print("  pybind11 already patched or pattern not found")
PYFIX
            fi

            # Fix 2: Module.cpp docstring handling - use generic PyObject API
            MODULE_FILE="torch/csrc/Module.cpp"
            if grep -q "f->m_ml->ml_doc" "${MODULE_FILE}" 2>/dev/null; then
                echo "Applying GraalPy fix for Module.cpp docstring handling..."
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
    print("  Module.cpp fix applied successfully")
else:
    print("  Module.cpp already patched or pattern not found")
PYFIX
            fi

            # Clean up .rej and .orig files
            find . -name "*.rej" -delete 2>/dev/null || true
            find . -name "*.orig" -delete 2>/dev/null || true

            echo "Patch application complete"
        else
            echo "ERROR: Expected source file torch/csrc/Generator.cpp not found"
            echo "Directory contents:"
            find . -name "Generator.cpp" 2>/dev/null | head -5
            exit 1
        fi
    else
        echo "WARNING: Patch file not found at ${PATCH_FILE}"
        echo "Build may fail without GraalPy compatibility patches"
    fi
else
    echo ""
    echo "Using existing PyTorch source at ${PYTORCH_SRC}"
    cd "${PYTORCH_SRC}"
fi

# Install build dependencies
# Note: Do NOT install cmake via pip - the Python wrapper is broken on GraalPy
# We use system cmake instead (already verified in prerequisites check)
echo ""
echo "Installing build dependencies..."
pip install pyyaml typing_extensions

# Configure build
echo ""
echo "Configuring PyTorch build..."

# Use system cmake explicitly (pip cmake wrapper is broken on GraalPy)
export CMAKE_COMMAND=$(which cmake)
echo "Using system cmake: ${CMAKE_COMMAND}"

# Minimal build - CPU only, no CUDA, no distributed
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=1
export USE_OPENMP=1
export BUILD_TEST=0

# Use ninja if available
if [ "$USE_NINJA" = "1" ]; then
    export CMAKE_GENERATOR=Ninja
fi

# Build PyTorch
# Use pip install to trigger GraalPy's automatic patching system
# (patches are in $GRAALPY_HOME/lib/graalpy25.0/patches/torch-2.7.0.patch)
echo ""
echo "Building PyTorch (this takes ~30 minutes on first build)..."
pip install .

# Install sympy (required for torch.fx shape propagation)
echo ""
echo "Installing sympy..."
pip install sympy mpmath

# Verify installation (must run from outside pytorch-src to avoid import confusion)
cd "${VENV_DIR}"
echo ""
echo "Verifying installation..."
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
echo "Venv created at: ${VENV_DIR}"
echo ""
echo "To use this venv with snakegrinder:"
echo "  export PYTORCH_VENV=${VENV_DIR}"
echo ""
