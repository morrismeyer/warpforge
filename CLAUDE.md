# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WarpForge is a multi-module Java build system integrating:
- **Babylon JDK** (Java 26) - Locally-built JDK with incubator code model APIs
- **SnakeBurger** - Tools for Babylon's code reflection API
- **SnakeGrinder** - GraalPy (Python on GraalVM) polyglot integration
- **Hardware CI** - Distributed testing across NUC orchestrator, NVIDIA, and AMD GPU boxes

## CRITICAL: No Data Files in Repository

**NEVER commit data files to this git repository.** This is a hard rule with zero exceptions.

Git repositories are for:
- Source code
- Configuration files
- Documentation

Git repositories are NOT for:
- Test fixtures with tensor data (`.npy`, `.bin`, `.safetensors`)
- Model weights
- Datasets (COCO, ImageNet, SQuAD, etc.)
- Any binary files larger than 100KB
- Generated outputs or artifacts

**Why this matters:**
- Large files bloat the repository permanently (even after deletion, they remain in git history)
- ML datasets can be 10GB-100GB+ which makes the repository unusable
- Cloning becomes slow and expensive
- CI/CD pipelines time out

**Where to store data instead:**
- Cloud storage (S3, GCS, Azure Blob) with download-on-demand
- Gradle task that downloads fixtures from a URL
- Git LFS for files that absolutely must be versioned (rare)
- Local cache directories outside the repository

**If you need test data:**
1. Create a Gradle task that downloads data from a URL
2. Store downloaded data in `build/` or a gitignored cache directory
3. Add the download URL to a manifest file (not the data itself)
4. Tests should skip gracefully if data isn't available

This rule exists because data was accidentally committed to this repository in the past, requiring a full repository re-clone to fix. Do not repeat this mistake.

## Architecture Documentation

For detailed architectural decisions and implementation roadmaps, see the `architecture/` directory:

- **[ARCHITECTURE.md](architecture/ARCHITECTURE.md)** - High-level system overview, module map, design principles
- **[BACKEND-PHASES.md](architecture/BACKEND-PHASES.md)** - Phased approach to GPU backend development

This file (CLAUDE.md) covers build commands, development workflow, and code style. The architecture docs cover **what to build and why**.

## Task Tracking

For ongoing work items and audit results, see the `tasks/` directory:

- **[GPU-TEST-AUDIT.md](tasks/GPU-TEST-AUDIT.md)** - Audit of GPU tests for real hardware verification
- **[RESEARCH-VALIDATION-AUDIT.md](tasks/RESEARCH-VALIDATION-AUDIT.md)** - Audit of research validation implementations

Task files track:
- What needs to be done (requirements)
- What was audited (findings)
- What passed/failed verification
- Specific fixes needed

**Before starting implementation work**, check `tasks/` for relevant audit findings and requirements.

### Architecture Doc Naming Convention

All architecture documentation files use **ALL-CAPS-WITH-HYPHENS.md** naming:
- `ARCHITECTURE.md` not `architecture.md`
- `BACKEND-PHASES.md` not `backend-phases.md`
- `GPU-SCHEDULING.md` not `gpu_scheduling.md`

This makes architecture docs visually distinct from code and config files.

## Pre-Implementation Requirements

**Before writing any implementation code, you MUST read the relevant architecture documents.**

This is not optional. Use the Read tool to read the actual files - do not rely on memory or summaries.

| Feature Area | Required Reading |
|--------------|------------------|
| GPU operations | `architecture/JFR-GPU.md`, `architecture/GPU-SCHEDULING.md` |
| Concurrency | `architecture/STRUCTURED-CONCURRENCY-RESEARCH.md` |
| Backend development | `architecture/BACKEND-PHASES.md` |
| Overall architecture | `architecture/ARCHITECTURE.md` |

**Implementation checklist:**
1. Read the relevant architecture docs (use Read tool, not memory)
2. Identify specific sections that apply to your task
3. Note exact API names, field names, and patterns from the docs
4. Implement according to those specifications exactly
5. Cross-reference your code against the docs before committing

**After context compaction (which you cannot detect), assume:**

- You have lost detailed implementation context from earlier in the conversation
- Re-read architecture documents before continuing ANY implementation work
- Do not rely on summaries - read the source files with the Read tool
- If a task seems partially complete, re-read the docs before continuing
- When in doubt, ask the user: "Should I re-read the architecture docs before proceeding?"

**A "validation" or "test" must:**

- Use assertions (JUnit, AssertJ) or JFR event verification
- Measure real hardware metrics, not simulated values
- Never declare success via `System.out.println("PASSED")` - that is not a test
- Query actual GPU state via NVML/CUDA APIs when testing GPU functionality

## Enforcement of Style Rules

The following rules in this document are **NON-NEGOTIABLE**. Violations should be fixed immediately upon detection:

- **EndToEnd, not E2E** - applies to all code, comments, documentation, and task names
- **Explicit imports only** - never use wildcard imports
- **No time estimates** - never predict durations for tasks
- **Architecture docs first** - read specifications before implementing

If you notice existing violations of these rules, fix them proactively.

## Compaction-Safe Summaries

**After completing any substantial implementation task, provide a compaction-safe summary.**

This summary should be structured to survive context compaction and enable work to continue correctly. Include:

1. **What was implemented** - Specific files created/modified with their purposes
2. **Key specifications followed** - Which architecture docs and which sections guided the implementation
3. **Critical details** - API names, field names, constants, or patterns that must not drift
4. **What remains** - Next steps with enough detail to continue without re-reading everything
5. **Verification commands** - How to verify the work is correct

Example format:
```
## Compaction-Safe Summary: [Task Name]

**Implemented:** GpuKernelEvent JFR integration in warpforge-core/src/main/java/.../jfr/

**Per specifications:** JFR-GPU.md lines 99-154 (event fields), GPU-SCHEDULING.md lines 248-265 (NVML utilization)

**Critical details:**
- Must use `nvmlDeviceGetUtilizationRates()` not simulated values
- GpuKernelEvent.teraflops = (2.0 * M * N * K) / (elapsedMs * 1e9)
- Backend field values: "cuBLAS", "PTX", "rocBLAS", "HIP"

**Remaining:** Add NVML bindings for nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

**Verify:** ./gradlew :warpforge-core:test --tests "*GpuKernel*"
```

This practice ensures that if compaction occurs, the summary in conversation history preserves actionable detail rather than vague descriptions.

## Language Preference

**When all else is equal, prefer Java over other languages.** This is the ethos of the WarpForge project.

If a tool, script, or component can be implemented in either Java or another language (e.g., Node.js, Python, Bash) with comparable effort and functionality, choose Java. This keeps the codebase consistent, reduces the number of runtime dependencies, and aligns with the project's core competency.

Exceptions are acceptable when:
- The task requires language-specific capabilities (e.g., Python for PyTorch integration in SnakeGrinder)
- A Java solution would require significantly more effort or complexity
- External dependencies mandate a specific language

## Vectorization: Use Java Vector API

**Every loop is a potential vector loop.** When writing code that processes arrays or performs numerical computation, always consider using the Java Vector API (`jdk.incubator.vector`).

The Vector API enables SIMD (Single Instruction, Multiple Data) operations that can provide 4-16x speedups over scalar loops. WarpForge is a high-performance ML compiler - vectorized code is not optional, it's expected.

**When to vectorize:**
- Any loop over float/double/int arrays
- Matrix operations (transpose, element-wise ops, reductions)
- Data transformation (normalization, scaling, type conversion)
- Batch processing of tensors

**Pattern to follow:**
```java
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

// Vectorized loop
int i = 0;
int upperBound = SPECIES.loopBound(data.length);
for (; i < upperBound; i += SPECIES.length()) {
    FloatVector v = FloatVector.fromArray(SPECIES, data, i);
    v = v.mul(scale);  // or other vector operations
    v.intoArray(result, i);
}
// Scalar tail
for (; i < data.length; i++) {
    result[i] = data[i] * scale;
}
```

**Do NOT write scalar loops like this when vectors are possible:**
```java
// BAD - scalar loop that should be vectorized
for (int i = 0; i < data.length; i++) {
    result[i] = data[i] * scale;
}
```

If you write a loop over numerical data, ask yourself: "Can this be vectorized?" The answer is usually yes.

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

## Backend Parity: NVIDIA ↔ AMD

**Every feature implemented for NVIDIA must have an equivalent implementation for AMD, and vice versa.** This is the entire ethos of WarpForge.

WarpForge is not an "NVIDIA-first" or "AMD-first" project. It is a **GPU-agnostic ML compiler** that treats both vendors as first-class citizens. When you add a capability to one backend, you must add the equivalent capability to the other.

### What Parity Means

| NVIDIA Component | AMD Equivalent | Notes |
|------------------|----------------|-------|
| CUDA Runtime | HIP Runtime | FFM bindings in respective backend modules |
| cuBLAS | hipBLAS/rocBLAS | Matrix operations |
| cuDNN | MIOpen | Deep learning primitives |
| NVML (monitoring) | ROCm SMI | GPU utilization, temperature, power |
| PTX (assembly) | GCN/RDNA ISA | Low-level kernel code |
| NCCL (collectives) | RCCL | Multi-GPU communication |

### Implementation Rules

1. **No NVIDIA-only features** - If you add NVML monitoring to NvidiaBackend, you must add ROCm SMI monitoring to AmdBackend in the same PR or immediately after.

2. **No AMD-only features** - The same applies in reverse. Parity goes both ways.

3. **Shared abstractions** - Create interfaces in `warpforge-core` that both backends implement:
   - `GpuBackend` - core GPU operations
   - `GpuMonitoring` - utilization/temperature/power queries
   - Future: `CollectiveBackend`, `TensorCoreBackend`, etc.

4. **Test parity** - If NVIDIA has 50 hardware execution tests, AMD must have 50 equivalent tests. See "Hardware Tests Must Run on Hardware" below.

5. **Documentation parity** - Architecture docs must cover both backends equally.

### Why This Matters

- **Vendor lock-in is the enemy** - Users should be able to switch between NVIDIA and AMD GPUs without changing their code
- **Competition drives innovation** - Supporting both backends keeps WarpForge relevant as GPU markets evolve
- **Correctness verification** - Running the same workload on both architectures catches subtle bugs that single-vendor testing misses
- **Research reproducibility** - ML research should not depend on specific hardware vendors

### Enforcement

When reviewing code changes:
- If a PR adds NVIDIA-specific functionality, ask: "Where is the AMD equivalent?"
- If a PR adds AMD-specific functionality, ask: "Where is the NVIDIA equivalent?"
- Exceptions require explicit justification (e.g., vendor-specific debugging tools)

**The audit documents in `tasks/` track parity status.** If one backend falls behind, it becomes a high-priority work item.

### Halt If Parity Is Not Achievable

**When implementing a GPU feature, always explore implementing for BOTH backends simultaneously.**

If you cannot implement a feature for both NVIDIA and AMD:
1. **STOP** - Do not proceed with a single-vendor implementation
2. **Explain** - Document why parity isn't currently possible (missing API, hardware limitation, etc.)
3. **Discuss** - Work with the user to decide how to proceed:
   - Can we design around the limitation?
   - Is there an alternative approach that works on both?
   - Should we defer the feature until both backends can support it?
   - Is this a rare exception that justifies vendor-specific code?

**Never silently ship NVIDIA-only or AMD-only code.** The architecture must evolve carefully to maintain parity. A half-implemented feature creates technical debt and violates the core WarpForge principle.

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

**Full details:** [architecture/TESTING.md](architecture/TESTING.md)

**Key rules:**
- Framework: JUnit 5 with `@Tag("cpu")`, `@Tag("nvidia")`, `@Tag("amd")`
- **Zero tolerance for test failures** - a failing test is a build-breaking event
- **Hardware tests must run on hardware** - code generation tests alone are insufficient
- **Build pass without tests = not validated** - if tests didn't run, investigate immediately
- **Test parity required** - if NVIDIA has tests, AMD must have equivalent tests

**Module test minimums (enforced by CI):**
- `snakeburger-core`: 300+ tests
- `warpforge-io`: 150+ tests
- `warpforge-core`: 100+ tests

## CI/CD

**Full details:** [architecture/CI-CD.md](architecture/CI-CD.md)

**Key points:**
- **Nightly build** verifies clone-to-run (fresh developer experience)
- **Weekly fresh clone** catches "works on my machine" issues
- **NUC has NO GPU** - use `nvidiaTest`/`amdTest` for GPU execution tests
- Tests tagged `@Tag("nvidia")` → NVIDIA box only
- Tests tagged `@Tag("amd")` → AMD box only

## SnakeGrinder Distribution Build

**Full details:** [architecture/SNAKEGRINDER-BUILD.md](architecture/SNAKEGRINDER-BUILD.md)

**Quick start:**
```bash
./gradlew :snakegrinder-dist:buildPytorchVenv   # ~30-60 min first time
./gradlew :snakegrinder-dist:assembleDist       # Build native distribution
```

Versions in `snakegrinder-dist/versions.env`: GraalPy 25.0.1, PyTorch 2.7.0

## Development Workflow: Fixes Must Survive Cleanup

When fixing build issues, **never make manual edits to generated or downloaded artifacts** (e.g., files inside `.pytorch-venv/`, `build/`). Create patch files in a permanent location and apply automatically.

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

## FFM (Foreign Function & Memory) API

When writing FFM bindings to native libraries (CUDA, HIP, cuBLAS, rocBLAS, etc.):

### invokeExact Requires Exact Return Type Matching

**FFM's `invokeExact` enforces exact type matching for BOTH parameters AND return types.** If a native function returns a value, you MUST capture it—even if you don't need it.

```java
// BAD - invokeExact will throw WrongMethodTypeException
// The function returns int but we're not capturing it
hiprtcDestroyProgram.invokeExact(progPtr);

// GOOD - capture the return value even if unused
@SuppressWarnings("unused")
int result = (int) hiprtcDestroyProgram.invokeExact(progPtr);
```

This differs from regular Java method calls where you can ignore return values. With `invokeExact`, the call site signature must exactly match the `MethodHandle`'s type, including the return type.

**Common symptoms of this bug:**
- `WrongMethodTypeException` at runtime
- Error mentions type mismatch between `()V` (void) and `()I` (returns int)
- Works in unit tests but fails in integration tests (if test mocking bypasses FFM)

### Known FFM Portability Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing return capture | `WrongMethodTypeException` | Always capture return value with correct type |
| Pointer type mismatch | Segfault or wrong data | Use `ADDRESS` for pointer types, `JAVA_LONG` for handles |
| Arena lifetime | Use-after-free crashes | Ensure arena outlives all allocated segments |
| String encoding | Garbled text or crashes | Use `arena.allocateFrom(str + "\0", StandardCharsets.UTF_8)` |

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
