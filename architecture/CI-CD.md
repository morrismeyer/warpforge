# CI/CD Architecture

This document covers WarpForge's continuous integration and deployment infrastructure.

## Nightly Full Build: Clone-to-Run Verification

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

## Weekly Fresh Clone Build

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

## Push-Triggered CI

GitHub Actions workflow (`orchestrated-ci.yml`) orchestrates:
1. SSH to NUC (self-hosted)
2. Build and test on NUC
3. Wake and test on NVIDIA box
4. Wake and test on AMD box

Required secrets: `HOLMES_NUC_HOST`, `HOLMES_NUC_USER`, `HOLMES_NUC_SSH_KEY`

## Hardware Test Distribution

**CRITICAL: The NUC has NO GPU.** Test tasks must run on the correct hardware:

| Test Task | Runs On | Hardware |
|-----------|---------|----------|
| `./gradlew test` | NUC | CPU only - no GPU |
| `./gradlew cpuTest` | NUC | CPU only |
| `./gradlew nvidiaTest` | NVIDIA box | RTX GPU + CUDA runtime |
| `./gradlew amdTest` | AMD box | Radeon GPU + ROCm runtime |

The NUC can compile GPU backend code and run FFM linkage tests (verifying native library bindings), but it **cannot execute GPU kernels**. Any test that actually runs CUDA/HIP kernels must be tagged appropriately and run on the corresponding GPU box.

- Tests tagged `@Tag("nvidia")` → run only via `nvidiaTest` on NVIDIA box
- Tests tagged `@Tag("amd")` → run only via `amdTest` on AMD box
- Tests tagged `@Tag("cpu")` or untagged → run on NUC via `test` or `cpuTest`

## Full Integration Test Path (Target Architecture)

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
