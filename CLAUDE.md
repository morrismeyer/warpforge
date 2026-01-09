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

Use `ensureBabylonReady` - it handles everything automatically:
```bash
# Pull latest, check deps, build (with auto-recovery on failure)
./gradlew :babylon-runtime:ensureBabylonReady -Pbabylon.bootJdk=$(/usr/libexec/java_home)

# Now snakeburger tasks will work
./gradlew :snakeburger-cli:run
```

#### Auto-Update Mode

For development, enable auto-update to always use the latest Babylon:
```bash
# Automatically pulls and rebuilds Babylon before compiling snakeburger
./gradlew :snakeburger-cli:run -Pbabylon.autoUpdate=true -Pbabylon.bootJdk=$(/usr/libexec/java_home)
```

#### Manual Setup (Step-by-Step)

```bash
# 0. Check/install build dependencies (macOS)
./gradlew :babylon-runtime:checkBabylonDeps

# 1. Build Babylon (requires boot JDK path)
./gradlew :babylon-runtime:buildBabylonImages -Pbabylon.bootJdk=/path/to/jdk

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

### SnakeGrinder (GraalPy + native-image)

- **Runtime**: GraalPy (Python on GraalVM)
- **Distribution**: GraalVM `native-image` → single-file executable
- **Key constraint**: **No pip install dependencies**. All Python code must be pure Python bundled as resources. This eliminates Python configuration hassle and enables true single-file distribution.
- **Why mock tracer**: The mock torch/jax modules exist because real PyTorch/JAX require pip install and native extensions incompatible with native-image bundling.
- **StableHLO output**: Text format (MLIR), emitted by pure Python code

### SnakeBurger (Babylon JDK)

- **Runtime**: Babylon JDK (Java 26 with `jdk.incubator.code` module)
- **Distribution**: `jlink` / `jpackage` → single-file executable
- **Key constraint**: Babylon is experimental, lives only in Oracle's GitHub repo, and subject to API flux. Cannot use native-image path because Babylon APIs are incubator/preview.
- **Future**: May integrate with HAT (Heterogeneous Accelerator Toolkit) when stable
- **StableHLO input**: Parses text format from snakegrinder, converts to Babylon Code Reflection IR

### Data Flow

```
Python ML code
      │
      ▼
┌─────────────────┐
│  SnakeGrinder   │  GraalPy + mock tracer
│  (native-image) │  Emits StableHLO text
└────────┬────────┘
         │ .mlir file
         ▼
┌─────────────────┐
│  SnakeBurger    │  Babylon JDK
│  (jlink)        │  Parses StableHLO → Babylon IR
└─────────────────┘
```

### Design Implications

- SnakeGrinder Python code must remain **pure Python** (no C extensions, no pip packages)
- SnakeBurger Java code must handle **Babylon API instability** gracefully
- The two projects communicate via **StableHLO text format** as a stable interface
- Both aim for **single-file executable** distribution but via different toolchains

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
