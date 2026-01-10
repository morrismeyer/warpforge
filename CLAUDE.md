# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WarpForge is a multi-module Java build system integrating:
- **Babylon JDK** (Java 26) - Locally-built JDK with incubator code model APIs
- **SnakeBurger** - Tools for Babylon's code reflection API
- **SnakeGrinder** - GraalPy (Python on GraalVM) polyglot integration
- **Hardware CI** - Distributed testing across NUC orchestrator, NVIDIA, and AMD GPU boxes

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

Example: PyTorch GraalPy patches live in `snakegrinder-dist/scripts/patches/` and are applied by `build-pytorch-venv.sh` after the main GraalPy patch.

## Git Commit Preferences

- **Never include Co-Authored-By lines** in commit messages under any circumstances
