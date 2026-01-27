# SnakeGrinder Build Guide

This document covers building and distributing SnakeGrinder.

## Quick Start

```bash
# Check current configuration and venv status
./gradlew :snakegrinder-dist:checkPytorchVenv

# Build PyTorch venv (~30-60 min first time, downloads GraalPy automatically)
./gradlew :snakegrinder-dist:buildPytorchVenv

# Optional: Prune venv to reduce size
./gradlew :snakegrinder-dist:prunePytorchVenv

# Build native distribution
./gradlew :snakegrinder-dist:assembleDist
```

## Version Configuration

All versions are centralized in `snakegrinder-dist/versions.env`:

```bash
GRAALPY_VERSION="25.0.1"
PYTORCH_VERSION="2.7.0"
PYTHON_VERSION="3.12"
```

## Upgrading PyTorch/GraalPy

**Constraint**: PyTorch version must have a matching GraalPy patch. Check `$GRAALPY_HOME/lib/graalpy*/patches/torch-*.patch` for available versions.

1. Check GraalPy releases: https://github.com/oracle/graalpython/releases
2. Check available PyTorch patches in the new GraalPy release
3. Update `snakegrinder-dist/versions.env`
4. Rebuild:
   ```bash
   ./gradlew :snakegrinder-dist:rebuildPytorchVenv
   ./gradlew :snakegrinder-dist:assembleDist
   ```

## Build Dependencies

The build script checks dependencies and provides install instructions:

| macOS | Linux |
|-------|-------|
| `brew install cmake ninja` | `sudo apt install cmake ninja-build build-essential` |
| Xcode CLI: `xcode-select --install` | |

## Checking for New PyTorch Patches

GraalPy periodically adds patches for newer PyTorch versions. Check for updates:

```bash
./snakegrinder-dist/scripts/check-graalpy-patches.sh
```

Track the upstream issue for PyTorch 2.8+ support: https://github.com/oracle/graalpython/issues/588

## macOS Packaging (DMG, PKG)

```bash
# Build macOS .app bundle
./gradlew :snakegrinder-dist:buildApp

# Build DMG installer
./gradlew :snakegrinder-dist:buildDmg

# Build PKG installer
./gradlew :snakegrinder-dist:buildPkg

# Build all macOS packages
./gradlew :snakegrinder-dist:buildAllMacOs
```

**Important**: All packaging scripts must be non-interactive. They must not:
- Mount DMGs and wait for user interaction
- Use AppleScript that opens Finder windows
- Require user clicks or confirmations

The build should complete without human intervention, suitable for CI/CD pipelines.

## PyTorch + GraalPy Build: Use Official Patching System

**Important**: PyTorch builds for GraalPy must use the **officially supported pip install method**:

```bash
pip install "torch==2.7.0" --no-binary torch --no-cache -v
```

This approach is documented by Oracle GraalPy staff and works because:

1. **autopatch_capi.py** - GraalPy automatically patches C API usages (e.g., `->ob_type` → `Py_TYPE()`)
2. **Official torch patches** - Fetched from `https://github.com/oracle/graalpython/tree/github/patches/`
3. **Correct ordering** - Auto-patching runs first, then manual patches apply cleanly

**Do NOT** maintain custom PyTorch patches in this repository. The official GraalPy patches are:
- Tested by Oracle
- Updated with each GraalPy release
- Designed to work with the auto-patching system

If you encounter build issues:
1. First try with `--no-cache` to ensure fresh patch download
2. Check the GraalPy GitHub issues: https://github.com/oracle/graalpython/issues
3. Report issues to Oracle if patches are missing for new PyTorch versions

Track PyTorch version support: https://github.com/oracle/graalpython/issues/588
