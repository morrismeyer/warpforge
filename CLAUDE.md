# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WarpForge is a multi-module Java build system integrating:
- **Babylon JDK** (Java 26) - Locally-built JDK with incubator code model APIs
- **SnakeBurger** - Tools for Babylon's code reflection API
- **SnakeGrinder** - GraalPy (Python on GraalVM) polyglot integration
- **Hardware CI** - Distributed testing across NUC orchestrator, NVIDIA, and AMD GPU boxes

## Architecture Documentation

For detailed architectural decisions and implementation roadmaps, see the `architecture/` directory:

- **[ARCHITECTURE.md](architecture/ARCHITECTURE.md)** - High-level system overview, module map, design principles
- **[BACKEND-PHASES.md](architecture/BACKEND-PHASES.md)** - Phased approach to GPU backend development

This file (CLAUDE.md) covers build commands, development workflow, and code style. The architecture docs cover **what to build and why**.

### Architecture Doc Naming Convention

All architecture documentation files use **ALL-CAPS-WITH-HYPHENS.md** naming:
- `ARCHITECTURE.md` not `architecture.md`
- `BACKEND-PHASES.md` not `backend-phases.md`
- `GPU-SCHEDULING.md` not `gpu_scheduling.md`

This makes architecture docs visually distinct from code and config files.

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

**Baseline JDK: Java 25 (GraalVM 25)**

The project uses JDK 25 as the baseline for all modules except SnakeBurger, which requires the experimental Babylon JDK (Java 26). GraalVM 25 is preferred for the baseline because it provides:
- Native-image compilation for high-performance executables
- GraalPy for Python interop (SnakeGrinder)
- Modern FFM (Foreign Function & Memory) API support

The build routes projects to different JDKs via `gradle/jdk-routing.gradle`:

| Project Pattern | JDK Required | Setup |
|-----------------|--------------|-------|
| `snakeburger-*` | Java 26 (Babylon) | Build from `../babylon`, source env file |
| `snakegrinder-*` | Java 25 (GraalVM) | Auto-discovered via toolchain |
| `warpforge-io` | Java 25 (GraalVM) | Native-image for performance tests |
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

- **architecture/** - Architectural documentation and roadmaps
- **app** - Main application with NVML (NVIDIA GPU) integration via FFM
- **babylon-runtime** - Build orchestration for Babylon JDK (not a Java library)
- **snakeburger-core/cli/codegen** - Babylon code model reflection (requires Java 26)
- **snakegrinder-core/cli/dist** - GraalPy polyglot context (requires GraalVM 25)
- **warpforge-core** - Core IR and analysis
- **warpforge-backend-cpu** - CPU reference backend
- **warpforge-backend-nvidia** - NVIDIA GPU backend (cuBLAS/cuDNN via FFM)
- **warpforge-backend-amd** - AMD GPU backend (hipBLAS/MIOpen via FFM)
- **warpforge-codegen-api** - Kernel codegen interfaces
- **warpforge-io** - I/O utilities
- **warpforge-launch-core/cli** - Distributed launch infrastructure
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

### Zero Tolerance for Test Failures

**The build must fail if even a single test fails.** This is non-negotiable.

Allowing test failures to accumulate is a dangerous development practice that leads to:
- Silent regressions that compound over time
- Loss of confidence in the test suite
- "It's probably fine" mentality that masks real bugs
- Technical debt that becomes impossible to pay down

**Rules:**
- A failing test is a build-breaking event—treat it with the same urgency as a compilation error
- Never commit code that causes test failures
- Never disable or skip tests to make the build pass (fix the test or fix the code)
- CI enforces minimum test counts per module to detect accidentally skipped tests

**If a test is flaky:**
- Fix the flakiness immediately—flaky tests are worse than no tests
- If the test cannot be fixed quickly, delete it and file an issue
- Never leave flaky tests in the suite "for now"

**Module test minimums (enforced by CI):**
- `snakeburger-core`: 300+ tests
- `warpforge-io`: 150+ tests
- `warpforge-core`: 100+ tests

### Fix the Root Cause, Not the Symptom

**When a build fails due to missing dependencies or environment issues, fix the underlying problem—never skip or disable tests to make the build pass.**

This applies to:
- Missing tools or binaries (e.g., GraalPy, native libraries)
- Missing environment configuration
- Unavailable external dependencies

**Wrong approach:**
```bash
# BAD: Skip tests when dependency is missing
if [[ ! -f "$GRAALPY_BIN" ]]; then
  log "Skipping tests because GraalPy is missing"
fi
```

**Correct approach:**
```bash
# GOOD: Download the dependency as part of the build
./gradlew :snakegrinder-dist:downloadGraalPy
./gradlew :snakegrinder-dist:testDist
```

The build system should be self-healing: if a required dependency is missing, download or build it automatically. Tests should always run—if they can't run due to missing infrastructure, that's a build system bug to fix, not a test to skip.

### Build Pass Without Tests = Not Validated

**If the build and unit tests completed but the unit tests did not actually run (for any reason), the code is not validated.**

A "successful" build that skips tests is worse than a failing build because:
- It provides false confidence that the code works
- Regressions slip through undetected
- The problem compounds as more changes are built on unvalidated code

**This counts as a failed build:**
- Tests skipped due to missing dependencies or configuration
- Tests skipped due to early exit in test setup
- Zero tests executed (test task "passed" trivially)
- Test count significantly lower than expected minimums

**What to do:**
1. If tests didn't run, investigate immediately—do not proceed with other work
2. Check the test output for "0 tests" or "skipped" messages
3. Fix the root cause (missing dependencies, broken setup, etc.)
4. Re-run and verify the expected number of tests executed

**Prevention:**
- CI enforces minimum test counts per module
- Test tasks should fail-fast if prerequisites are missing
- Never treat "no tests found" as a passing condition

## CI/CD

### Nightly Full Build: Clone-to-Run Verification

**The nightly build must verify that a new developer can clone WarpForge and build everything from scratch.**

This is the ultimate "It Just Works" test. Every night, CI performs a complete clean build that:

1. **Wipes everything** - All build artifacts, Gradle caches, and generated files (keeps only PyTorch source cache for speed)
2. **Builds PyTorch venv from scratch** - Verifies GraalPy + PyTorch builds correctly
3. **Builds all modules** - Full `./gradlew clean assemble` with no caching
4. **Runs all tests** - Unit, integration, and performance tests
5. **Validates performance baselines** - Catches regressions in collective operations

**Why this matters:**
- Ensures reproducible builds - if nightly passes, any developer can build
- Catches "works on my machine" issues before they accumulate
- Validates that build scripts, patches, and dependencies are correctly configured
- Performance baselines prevent silent throughput regressions

**Nightly workflow location:** `.github/workflows/nightly-full-build.yml`

### Weekly Fresh Clone Build

**The weekly build is the ultimate "It Just Works" test.**

Every Sunday, CI performs a completely fresh clone into a new directory (`~/surfworks/warpforge-weekly`) and builds everything as a new developer would. This catches issues that the nightly build might miss:

- Hardcoded paths that only work on existing setups
- Dependencies on files outside the repository
- Build scripts that assume certain directories exist
- Any "works on my machine" issues

**What it tests:**
1. Fresh `git clone` from GitHub (no existing checkout)
2. Build dependencies are documented and available
3. PyTorch venv builds from scratch
4. All modules compile
5. All tests pass

**After success, the fresh clone is deleted** to save disk space.

**Weekly workflow location:** `.github/workflows/weekly-fresh-clone.yml`

**Manual trigger for testing:**
```bash
# Via GitHub CLI
gh workflow run "Nightly Full Build" --ref main
gh workflow run "Weekly Fresh Clone Build" --ref main

# Or via GitHub UI: Actions → [workflow name] → Run workflow
```

**What gets tested:**

| Phase | Description | Duration |
|-------|-------------|----------|
| Full Clean | Wipe all artifacts and caches | ~1 min |
| PyTorch Venv | Build GraalPy + PyTorch from source | ~30-60 min |
| Full Build | All modules, no cache | ~5 min |
| Unit Tests | All module tests | ~5 min |
| SnakeGrinder | Distribution tests | ~2 min |
| Native Image | Build `ucc-perf-test` | ~4 min |
| GPU Tests | NVIDIA + AMD box tests | ~10 min |
| Two-Node Perf | NVIDIA↔AMD collective benchmarks | ~5 min |
| Baseline Check | Compare against performance baselines | ~1 min |

**Performance baselines:** `holmes-lab/mark1/baselines/`

### Push-Triggered CI

GitHub Actions workflow (`orchestrated-ci.yml`) orchestrates:
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

## Pre-Commit Checklist

**Before every commit, complete these steps in order:**

1. **Run `git status`** - Review both staged AND unstaged changes
2. **Verify no unstaged dependencies** - If unstaged files are modified, ask: "Does my staged code depend on any of these?" If yes, stage them too.
3. **Run `./gradlew clean assemble`** - A clean build catches missing files because it compiles only committed/staged code
4. **Run `./gradlew test`** - Verify tests pass
5. **Commit and push**

**The most common CI failure is committing code that calls methods/classes in files you forgot to stage.** Step 2 catches this by inspection, step 3 catches it by compilation failure. Both are needed—step 2 is faster, step 3 is more thorough.

```bash
# Full pre-commit sequence
git status                    # Step 1-2: Review staged AND unstaged
./gradlew clean assemble      # Step 3: Clean rebuild
./gradlew test                # Step 4: Run tests
git commit -m "message"       # Step 5: Commit
git push                      # Push
```

**Why this matters:**
- A failed CI build that could have been caught locally wastes everyone's time
- CI infrastructure has limited capacity; don't use it as your first line of defense
- Other developers may be blocked waiting for a green build
- Missing imports, typos, and basic test failures should never reach CI

**If the local build passes but CI still fails:**
- Check for platform-specific issues (macOS vs Linux)
- Check for environment differences (JDK version, dependencies)
- Check for flaky tests that need fixing

This discipline ensures CI remains a verification step, not a discovery step.

## Post-Push CI Monitoring: Own Your Push

**After every `git push`, you are responsible for monitoring CI and fixing any failures automatically.**

When you push code, you must:
1. Check CI status using `gh run list --limit 5`
2. If a run is in progress, wait for it to complete or check back
3. If CI fails, immediately investigate with `gh run view <run-id> --log-failed`
4. Fix the issue and push the fix without waiting for human intervention

**This is non-negotiable.** A push is not complete until CI is green. Do not:
- Push and walk away assuming it will pass
- Wait for a human to tell you CI failed
- Leave the main branch broken for someone else to fix

**Automated fix workflow:**
```bash
# 1. Push your changes
git push

# 2. Check CI status
gh run list --limit 3

# 3. If failed, investigate
gh run view <run-id> --log-failed | tail -100

# 4. Fix locally, test, commit, and push the fix
./gradlew clean assemble test
git add -A && git commit -m "Fix: <description of CI failure>"
git push

# 5. Repeat until CI is green
```

**Common CI-specific failures to watch for:**
- Platform differences (macOS builds locally, Linux runs on CI)
- Missing native libraries on CI that exist locally
- Path differences between local dev and CI environment
- Tests that pass in isolation but fail in full suite

The goal is zero human intervention for routine CI failures. If you broke it, you fix it—immediately.

## Java Code Style

- **Single blank line after package statement** before imports
- **Explicit imports only** - never use wildcard imports like `import foo.bar.*`
  - Every import must specify the exact class being imported
  - This improves code readability and makes dependencies explicit
  - Static imports must also be explicit (e.g., `import static org.junit.jupiter.api.Assertions.assertEquals`)
- **Use "EndToEnd" not "E2E"** in class names, method names, and documentation
  - `EndToEndBenchmark` not `E2EBenchmark`
  - `runEndToEndTests()` not `runE2ETests()`
  - This improves readability and avoids cryptic abbreviations
- Example:
  ```java
  package io.surfworks.warpforge.example;

  import java.util.List;
  import java.util.Map;
  import java.util.Optional;

  import static org.junit.jupiter.api.Assertions.assertEquals;
  import static org.junit.jupiter.api.Assertions.assertNotNull;
  ```

## PTX/CUDA Code Style

When writing or fixing PTX (Parallel Thread Execution) code in `CudaKernels.java`:

### Scan for Similar Issues

**When you fix an error in PTX code, immediately scan the entire PTX codebase for other instances of the same error pattern.** Do not commit a fix for one function without checking all similar functions.

Example: If you find that `copysign.f32` doesn't work on some GPU architectures and fix it with `setp` + `neg.f32`, search for ALL uses of `copysign.f32` in the codebase and fix them all in the same commit.

### Known PTX Portability Issues

| Instruction | Issue | Fix |
|-------------|-------|-----|
| `copysign.f32` | Not available on all architectures | Use `setp.lt.f32` + `@%p neg.f32` |
| `atom.global.add.u64` | Wrong for timing accumulation | Use `red.global.add.u64` |
| Unicode in PTX comments | CUDA driver rejects (ptxas accepts) | Use ASCII only in PTX strings |
| Double predicates (`@%p1 @%p2`) | Invalid PTX syntax | Use single predicate per instruction |

### PTX Comment Character Encoding

**PTX string literals must contain only ASCII characters.** While `ptxas` (the PTX assembler) may accept Unicode characters in comments, the CUDA driver's `cuModuleLoadData` will reject them with `CUDA_ERROR_INVALID_PTX (218)`.

```java
// BAD - Unicode in PTX string
String ptxOps = """
    // Compute base angle in [0, π/2]
    // atan(z) ≈ z * (1 - z^2/3)""";

// GOOD - ASCII only in PTX string
String ptxOps = """
    // Compute base angle in [0, pi/2]
    // atan(z) ~ z * (1 - z^2/3)""";
```

Unicode is fine in Java comments and Javadoc outside the PTX string literals.

## Documentation Style: Tables

**Use ASCII box-drawing tables**, not GitHub-flavored markdown tables. Tables must have:
- Vertical lines between all columns
- Horizontal lines on top, below header, and at bottom

**Correct format:**
```
+---------------+------------+-------------+
| Collective    | Throughput | % Line Rate |
+---------------+------------+-------------+
| ReduceScatter | 40.15 Gbps | 68.2%       |
| AllReduce     | 36.90 Gbps | 62.6%       |
+---------------+------------+-------------+
```

**Wrong format (do not use):**
```
| Collective | Throughput | % Line Rate |
|------------|------------|-------------|
| ReduceScatter | 40.15 Gbps | 68.2% |
```

This ensures tables render consistently across all viewers and terminals.

## Configuration Preference: Minimize Environment Variables

**Avoid environment variables for configuration wherever possible.** This aligns with the "It Just Works" philosophy.

Preferred configuration sources (in order):
1. **Auto-detection** - Discover paths, settings, and dependencies automatically
2. **Sibling directories** - Look for related tools/configs in predictable locations (e.g., `../babylon`, `./lib/`)
3. **Config files** - Use `~/.config/warpforge/*.json` for persistent user preferences
4. **CLI arguments** - Explicit flags for one-time overrides

Environment variables should only be used when:
- Required by external tools (e.g., `JAVA_HOME` for JDK discovery)
- Needed for CI/CD secrets that can't be stored in files
- Overriding auto-detected values in unusual deployment scenarios

**Bad**: `WARPFORGE_SCHEDULER=ray ./warpforge-launch submit ...`
**Good**: `./warpforge-launch submit --scheduler ray ...` or auto-detect from config file

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
