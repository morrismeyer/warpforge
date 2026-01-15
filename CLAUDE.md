# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WarpForge is a multi-module Java build system integrating:
- **Babylon JDK** (Java 26) - Locally-built JDK with incubator code model APIs
- **SnakeBurger** - Tools for Babylon's code reflection API
- **SnakeGrinder** - GraalPy (Python on GraalVM) polyglot integration
- **Hardware CI** - Distributed testing across NUC orchestrator, NVIDIA, and AMD GPU boxes

## Language Preference

**When all else is equal, prefer Java over other languages.** This is the ethos of the WarpForge project.

If a tool, script, or component can be implemented in either Java or another language (e.g., Node.js, Python, Bash) with comparable effort and functionality, choose Java. This keeps the codebase consistent, reduces the number of runtime dependencies, and aligns with the project's core competency.

Exceptions are acceptable when:
- The task requires language-specific capabilities (e.g., Python for PyTorch integration in SnakeGrinder)
- A Java solution would require significantly more effort or complexity
- External dependencies mandate a specific language

## Build Commands

```bash
# Build all
./gradlew clean assemble

# Run all tests
./gradlew test

# Hardware-specific tests (use @Tag annotations)
./gradlew cpuTest      # CPU-only tests
./gradlew nvidiaTest   # NVIDIA GPU tests
./gradlew amdTest      # AMD GPU tests

# Run single test
./gradlew :app:test --tests "io.surfworks.warpforge.core.app.MessageUtilsTest.testGetMessage"

# Run applications
./gradlew :snakeburger-cli:run   # Requires Babylon JDK setup
./gradlew :snakegrinder-cli:run  # Requires GraalVM 25
./gradlew :app:run
```

## JDK Routing Architecture

The build routes projects to different JDKs via `gradle/jdk-routing.gradle`:

| Project Pattern | JDK Required | Setup |
|-----------------|--------------|-------|
| `snakeburger-*` | Java 26 (Babylon) | Build from `../babylon`, source env file |
| `snakegrinder-*` | Java 25 (GraalVM) | Auto-discovered via toolchain |
| All others | Java 25 | Standard JDK |

### Babylon JDK Setup (for SnakeBurger)

**Prerequisite**: Clone the Babylon repository to `~/surfworks/babylon`:
```bash
git clone https://github.com/openjdk/babylon.git ~/surfworks/babylon
```

**Build dependencies** (checked automatically by `checkBabylonDeps`):

| macOS | Linux (apt) |
|-------|-------------|
| Xcode CLI tools | build-essential |
| autoconf (via Homebrew) | autoconf, libcups2-dev |
| | libx11-dev, libxext-dev, libxrender-dev |
| | libxrandr-dev, libxtst-dev, libxt-dev |
| | libfontconfig1-dev, libasound2-dev |

#### Quick Start (Recommended)

Use `ensureBabylonReady` - it handles everything automatically (boot JDK is auto-detected):
```bash
# Pull latest, check deps, build (with auto-recovery on failure)
./gradlew :babylon-runtime:ensureBabylonReady

# Now snakeburger tasks will work
./gradlew :snakeburger-cli:run
```

#### Auto-Update Mode

For development, enable auto-update to always use the latest Babylon:
```bash
# Automatically pulls and rebuilds Babylon before compiling snakeburger
./gradlew :snakeburger-cli:run -Pbabylon.autoUpdate=true
```

#### Manual Setup (Step-by-Step)

```bash
# 0. Check/install build dependencies
./gradlew :babylon-runtime:checkBabylonDeps

# 1. Build Babylon (boot JDK auto-detected, or specify explicitly)
./gradlew :babylon-runtime:buildBabylonImages
# Or with explicit boot JDK: ./gradlew :babylon-runtime:buildBabylonImages -Pbabylon.bootJdk=/path/to/jdk

# 2. Generate and source environment (optional, for IDE integration)
./gradlew :babylon-runtime:writeBabylonToolchainEnv
source babylon-runtime/build/babylon.toolchain.env

# 3. Now snakeburger tasks will work
./gradlew :snakeburger-cli:run
```

#### Babylon Tasks Reference

| Task | Description |
|------|-------------|
| `ensureBabylonReady` | Pull latest, check deps, build with auto-recovery |
| `babylonUpdate` | Alias for `ensureBabylonReady` |
| `checkBabylonDeps` | Verify build dependencies are installed |
| `installBabylonDeps` | Auto-install missing deps via Homebrew |
| `configureBabylon` | Run `./configure` in Babylon repo |
| `buildBabylonImages` | Run `make images` in Babylon repo |

## Project Structure

- **app** - Main application with NVML (NVIDIA GPU) integration via FFM
- **babylon-runtime** - Build orchestration for Babylon JDK (not a Java library)
- **snakeburger-core/cli** - Babylon code model reflection (requires Java 26)
- **snakegrinder-core/cli** - GraalPy polyglot context (requires GraalVM 25)
- **list** - Linked list data structure
- **utilities** - String manipulation utilities
- **build-logic** - Gradle convention plugins
- **holmes-lab/mark1/ci-scripts** - Hardware CI orchestration scripts

## Design Philosophy: "It Just Works"

Inspired by Steve Jobs' NeXT and Apple philosophy: **software should just work**.

There must be zero developer configuration, environment variables, secondary installs, futzing around, twiddling, head scratching, or other headache inducing shenanigans in this code.

When a developer downloads WarpForge tools, they should be able to run them immediately without any setup steps beyond extracting the archive.

### Self-Contained Build Artifacts

All build artifacts must be self-contained. The executable and all libraries it depends on must reside in the same build output directory.

"Eating your own dogfood" doesn't mean spreading it all over the kitchen floor for the dog to eat. It means putting it neatly in a bowl.

Even WarpForge developers should have a clean, encapsulated testing and verification path:

```
build/snakegrinder-dist/
├── bin/
│   └── snakegrinder          # Native executable (or wrapper script)
└── lib/
    └── *.dylib               # All dependent libraries
```

Run from the build directory: `./build/snakegrinder-dist/bin/snakegrinder`

No environment variables. No paths pointing elsewhere. One directory contains everything needed to run.

## SnakeGrinder vs SnakeBurger Architecture

These two projects have distinct architectural goals and distribution paths:

### SnakeGrinder (GraalPy + Real PyTorch + native-image)

- **Runtime**: GraalPy 25.0.1 with **real PyTorch 2.7.0** (built from source with GraalPy patches)
- **Tracing**: `torch.fx.symbolic_trace` for full-fidelity model capture
- **Distribution**: Self-contained directory with native executable + bundled libs
- **StableHLO output**: Text format (MLIR), converted from FX graph

#### Validated Capabilities (January 2025)

| Feature | Status |
|---------|--------|
| PyTorch 2.7.0 on GraalPy | ✅ Works (builds from source) |
| `torch.fx.symbolic_trace` | ✅ Full support |
| `torch.jit.trace` | ✅ Full support |
| `torch.jit.script` | ✅ Full support |
| `torch.export` (dynamo) | ❌ Not supported on GraalPy |
| FX → StableHLO conversion | ✅ Working prototype |
| Native-image build | ✅ Works (~4 min build) |

#### Build Requirements (for WarpForge developers only)

PyTorch must be built from source for GraalPy (no prebuilt wheels available):
- cmake >= 3.18
- C++ compiler (Xcode on macOS, gcc on Linux)
- ~30-60 minutes first-time build
- ~2-5GB disk space

#### Distribution Structure (end-user receives this)

```
snakegrinder/
├── bin/
│   └── snakegrinder          # Native executable
└── lib/
    ├── libtorch_cpu.dylib    # PyTorch native libs
    ├── libc10.dylib
    └── ...
```

**Usage**: `./snakegrinder/bin/snakegrinder --help`

No environment variables. No PATH changes. No pip install. It just works.

### SnakeBurger (Babylon JDK)

- **Runtime**: Babylon JDK (Java 26 with `jdk.incubator.code` module)
- **Distribution**: `jlink` / `jpackage` → single-file executable
- **Key constraint**: Babylon is experimental, lives only in Oracle's GitHub repo, and subject to API flux. Cannot use native-image path because Babylon APIs are incubator/preview.
- **Future**: May integrate with HAT (Heterogeneous Accelerator Toolkit) when stable
- **StableHLO input**: Parses text format from snakegrinder, converts to Babylon Code Reflection IR

### Data Flow

```
PyTorch Model (nn.Module)
         │
         ▼
┌─────────────────────────────────────┐
│  SnakeGrinder                       │
│  ┌───────────────────────────────┐  │
│  │ torch.fx.symbolic_trace       │  │
│  │         ↓                     │  │
│  │ FX Graph (real PyTorch ops)   │  │
│  │         ↓                     │  │
│  │ FXToStableHLO converter       │  │
│  └───────────────────────────────┘  │
│  Native executable + libs           │
└──────────────┬──────────────────────┘
               │ .mlir file (StableHLO text)
               ▼
┌─────────────────────────────────────┐
│  SnakeBurger                        │
│  ┌───────────────────────────────┐  │
│  │ StableHLO Parser              │  │
│  │         ↓                     │  │
│  │ Type Checker                  │  │
│  │         ↓                     │  │
│  │ Babylon Code Reflection IR    │  │
│  └───────────────────────────────┘  │
│  jlink/jpackage executable          │
└─────────────────────────────────────┘
```

### Design Implications

- SnakeGrinder uses **real PyTorch** for full-fidelity tracing (not mock modules)
- Distribution is **self-contained directory** (executable finds its libs automatically)
- End users need **zero configuration** - extract and run
- SnakeBurger Java code must handle **Babylon API instability** gracefully
- The two projects communicate via **StableHLO text format** as a stable interface
- Build system handles all complexity; users never see cmake, pip, or env vars

## Testing

- Framework: JUnit 5
- Tag tests with `@Tag("cpu")`, `@Tag("nvidia")`, or `@Tag("amd")` for hardware-specific execution
- GPU tests have 60-second SSH/wake timeouts

## CI/CD

GitHub Actions workflow (`holmes-mark1-ci.yml`) orchestrates:
1. SSH to NUC (self-hosted)
2. Build and test on NUC
3. Wake and test on NVIDIA box
4. Wake and test on AMD box

Required secrets: `HOLMES_NUC_HOST`, `HOLMES_NUC_USER`, `HOLMES_NUC_SSH_KEY`

### Full Integration Test Path (Target Architecture)

The NUC orchestrates the complete ML-to-GPU pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  NUC (Linux x86_64) - Orchestrator                                      │
│                                                                         │
│  1. SnakeGrinder traces PyTorch model → StableHLO MLIR                 │
│  2. SnakeBurger parses StableHLO → Babylon Code Reflection IR          │
│  3. WarpForge generates GPU kernels from IR                            │
│                                                                         │
└─────────────────────┬───────────────────────┬───────────────────────────┘
                      │                       │
                      ▼                       ▼
        ┌─────────────────────┐   ┌─────────────────────┐
        │  Node A (AMD GPU)   │   │  Node B (NVIDIA GPU)│
        │  - ROCm runtime     │   │  - CUDA runtime     │
        │  - Execute kernels  │   │  - Execute kernels  │
        │  - Verify results   │   │  - Verify results   │
        └─────────────────────┘   └─────────────────────┘
```

**Critical dependency**: SnakeGrinder must build and run on Linux (NUC) for this pipeline to work. This requires:
- GraalPy 25.0.1 for Linux x86_64
- PyTorch 2.7.0 built from source with GraalPy patches
- Native-image compilation on Linux

## SnakeGrinder Distribution Build

### Quick Start

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

### Version Configuration

All versions are centralized in `snakegrinder-dist/versions.env`:

```bash
GRAALPY_VERSION="25.0.1"
PYTORCH_VERSION="2.7.0"
PYTHON_VERSION="3.12"
```

### Upgrading PyTorch/GraalPy

**Constraint**: PyTorch version must have a matching GraalPy patch. Check `$GRAALPY_HOME/lib/graalpy*/patches/torch-*.patch` for available versions.

1. Check GraalPy releases: https://github.com/oracle/graalpython/releases
2. Check available PyTorch patches in the new GraalPy release
3. Update `snakegrinder-dist/versions.env`
4. Rebuild:
   ```bash
   ./gradlew :snakegrinder-dist:rebuildPytorchVenv
   ./gradlew :snakegrinder-dist:assembleDist
   ```

### Build Dependencies

The build script checks dependencies and provides install instructions:

| macOS | Linux |
|-------|-------|
| `brew install cmake ninja` | `sudo apt install cmake ninja-build build-essential` |
| Xcode CLI: `xcode-select --install` | |

### Checking for New PyTorch Patches

GraalPy periodically adds patches for newer PyTorch versions. Check for updates:

```bash
./snakegrinder-dist/scripts/check-graalpy-patches.sh
```

Track the upstream issue for PyTorch 2.8+ support: https://github.com/oracle/graalpython/issues/588

### macOS Packaging (DMG, PKG)

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

## Development Workflow: Fixes Must Survive Cleanup

When fixing build issues, **never make manual edits to generated or downloaded artifacts** (e.g., files inside `.pytorch-venv/`, `build/`, or any directory that gets deleted on clean rebuild).

**Wrong approach:**
```bash
# Editing a file that will be deleted on rebuild
vim .pytorch-venv/pytorch-src/some/file.cpp  # BAD - lost on rm -rf .pytorch-venv
```

**Correct approach:**
1. Create patch files in a permanent location (e.g., `scripts/patches/`)
2. Update the build script to apply patches automatically
3. Verify by doing a clean rebuild: `rm -rf <artifact-dir> && ./gradlew build`

This ensures fixes are:
- **Repeatable** - Works for other developers and CI
- **Documented** - Patch files show exactly what changed and why
- **Resilient** - Survives `clean` operations

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

## Git Commit Preferences

- **Never include Co-Authored-By lines** in commit messages under any circumstances

## Test Isolation: No Repo Modifications

**Tests must NEVER create, modify, or delete files in the Git repository.**

All test types (unit, integration, performance, fixture validation, etc.) must:
- Write output only to `build/` directories or system temp directories
- Never modify source files, resource files, or fixtures
- Leave `git status` clean after execution

**Fixture generators are NOT tests** - they are on-demand tools that intentionally modify fixtures:
- Must be tagged separately (e.g., `@Tag("fixture-generator")`)
- Must be excluded from all test tasks (`excludeTags 'fixture-generator'`)
- Should only run via explicit task (e.g., `./gradlew generateFixtures`)
- Generated changes must be reviewed and committed intentionally

If `git status` shows modified files after running `./gradlew test`, this is a bug that must be fixed immediately.

## E2E Fixture Versioning: Principle of Least Developer Surprise

**Generated test fixtures must stay synchronized with their generator's dependencies.**

E2E test fixtures (in `warpforge-core/src/test/resources/fixtures/e2e/`) are generated by running PyTorch models through the snakegrinder native binary. These fixtures capture expected outputs that WarpForge must match.

**Problem**: If PyTorch is upgraded but fixtures aren't regenerated, tests may pass/fail for wrong reasons—the expected values no longer reflect current PyTorch behavior.

**Solution**: Version-stamped fixtures with automatic validation.

- `fixtures/e2e/VERSION` contains the PyTorch version used to generate fixtures
- The `test` task depends on `checkE2EFixturesVersion` which compares against `snakegrinder-dist/versions.env`
- If versions mismatch, the build fails with clear instructions:
  ```
  E2E fixtures are stale: generated with PyTorch 2.7.0, current is 2.8.0.
  Run: ./gradlew :warpforge-core:generateE2EFixtures
  ```

**Workflow when upgrading PyTorch**:
1. Update `snakegrinder-dist/versions.env` with new version
2. Rebuild venv: `./gradlew :snakegrinder-dist:buildPytorchVenv`
3. Rebuild distribution: `./gradlew :snakegrinder-dist:assembleDist`
4. Regenerate fixtures: `./gradlew :warpforge-core:generateE2EFixtures`
5. Review and commit the updated fixtures

This ensures developers are never surprised by stale test data—the build system enforces consistency.

## Polyglot Verification Testing

**When implementing the same functionality in multiple languages, both implementations must produce identical output.**

This is a core WarpForge principle: if you write a tool in Python, and a matching tool in Java, they must be byte-for-byte identical in their output. This proves correctness across language boundaries and is essential for a project that bridges Python (PyTorch) and Java (Babylon).

### Example: Logo Generator

The `assets/` directory contains matching Python and Java implementations:

```bash
# Python version
python3 assets/generate-logo.py --all --svg-only --output ./test-py

# Java version
java assets/GenerateLogo.java --all --svg-only --output ./test-java

# Verify identical output
diff -r test-py test-java
```

### Test Script

Run the automated polyglot verification:

```bash
./assets/test-logo-generators.sh
```

This script:
1. Runs both Python and Java generators
2. Compares all SVG outputs
3. Fails if any differences are found

### When to Apply This Pattern

Use polyglot verification when:
- Implementing CLI tools that could be written in either language
- Creating code generators or formatters
- Building serialization/deserialization logic
- Any functionality that crosses the Python↔Java boundary

### Implementation Guidelines

1. **Same CLI interface** - Both implementations must accept identical command-line arguments
2. **Same output format** - Byte-for-byte identical output (normalize whitespace if needed)
3. **Automated test** - Always include a verification script
4. **Document both** - Keep implementations in sync when making changes
