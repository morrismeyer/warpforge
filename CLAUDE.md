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

## SnakeGrinder vs SnakeBurger Architecture

These two projects have distinct architectural goals and distribution paths:

### SnakeGrinder (GraalPy + Real PyTorch + native-image)

- **Runtime**: GraalPy 25.0.1 with **real PyTorch 2.7.0** (built from source with GraalPy patches)
- **Tracing**: `torch.fx.symbolic_trace` for full-fidelity model capture
- **Distribution**: GraalVM `native-image` executable (~736MB) + PyTorch native libs (~210MB)
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
| Native executable runs | ✅ Works (with DYLD_LIBRARY_PATH) |

#### Build Requirements

PyTorch must be built from source for GraalPy (no prebuilt wheels available):
- cmake >= 3.18
- C++ compiler (Xcode on macOS, gcc on Linux)
- ~30-60 minutes first-time build
- ~2-5GB disk space

#### Distribution Structure

```
snakegrinder-dist/
├── snakegrinder              # Native executable (736MB)
└── lib/
    ├── libtorch_cpu.dylib    # PyTorch native libs (~210MB total)
    ├── libc10.dylib
    ├── libtorch_python.dylib
    └── ...
```

Run with: `DYLD_LIBRARY_PATH=lib ./snakegrinder`

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
- Distribution is **executable + lib folder** (not single file, but still "just works")
- SnakeBurger Java code must handle **Babylon API instability** gracefully
- The two projects communicate via **StableHLO text format** as a stable interface
- Build requires cmake/C++ toolchain (one-time setup, then cached)

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

## Git Commit Preferences

- **Never include Co-Authored-By lines** in commit messages under any circumstances
