# Backend Kernel Instrumentation Strategy

This document describes WarpForge's approach to GPU kernel instrumentation, designed to provide meaningful performance observability without introducing Heisenbug scenarios.

## Three-Tier Kernel Architecture

WarpForge provides three execution tiers that balance performance against observability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WarpForge Kernel Tiers                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRODUCTION          ████████████████████████████████████████  100% perf   │
│  (cuBLAS/rocBLAS)    │ Vendor libraries, external timing only │            │
│                                                                             │
│  OPTIMIZED_          ██████████████████████████████████████    ~93% perf   │
│  OBSERVABLE          │ Boehm-style PTX with salt instrumentation│          │
│  (Optimized PTX)                                                            │
│                                                                             │
│  CORRECTNESS         ████                                      ~1% perf    │
│  (Naive PTX)         │ Full tracing, numerical verification   │            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier Selection Guide

| Tier | Performance | Observability | Use Case |
|------|-------------|---------------|----------|
| **PRODUCTION** | 100% | External timing (JFR + CUDA Events) | Training at scale, inference |
| **OPTIMIZED_OBSERVABLE** | ~93% | Salt instrumentation (kernel internals) | Performance tuning, profiling |
| **CORRECTNESS** | ~1% | Full tracing, step-by-step | Debugging numerical issues |

### Why Three Tiers?

Current ML tooling has a fundamental visibility gap:

- **PyTorch Profiler**: Shows operator times, but not kernel internals
- **Nsight Compute**: Shows everything, but with 2-10x overhead

The OPTIMIZED_OBSERVABLE tier fills this gap: **kernel-internal visibility at near-production performance**.

This architecture is inspired by [Simon Boehm's work](https://siboehm.com/articles/22/CUDA-MMM) showing that
optimized CUDA (with coalescing, shared memory, register blocking, warptiling) can achieve 93.7% of cuBLAS performance.
By implementing this optimization level ourselves, we can add salt instrumentation while maintaining performance.

### Tier Implementation

```java
public enum KernelTier {
    /** Maximum performance via vendor libraries. External timing only. */
    PRODUCTION,

    /** Near-production performance with salt instrumentation for kernel internals. */
    OPTIMIZED_OBSERVABLE,

    /** Full observability at any cost. For debugging numerical issues. */
    CORRECTNESS
}
```

## Core Principle: One Implementation, Instrumentation as Salt

**Within each tier, there is exactly ONE implementation of each kernel operation.** Instrumentation variants are generated from the same code template with conditional sections enabled based on the "salt level."

This avoids the fundamental Heisenbug problem: if you have separate "production" and "debug" implementations, measuring the debug path tells you nothing about production performance. The performance characteristics of different code are non-determinative.

## Salt Levels

| Salt Level | Constant | Description | Overhead |
|------------|----------|-------------|----------|
| **SALT_NONE** | 0 | Production kernel, no instrumentation | Baseline |
| **SALT_TIMING** | 1 | Cycle counters around compute sections | ~8 instructions |
| **SALT_TRACE** | 2 | Memory access patterns, warp divergence | Higher (TBD) |

## Why "Salt"?

The term "salt" is borrowed from cryptography where a salt is added to input to produce different but related outputs. Here, the salt value produces different but related kernels:

- Same algorithm
- Same memory access patterns
- Same register allocation strategy
- Different: presence/absence of timing/tracing instructions

The overhead is **quantifiable and consistent**, allowing:
```
actual_performance ≈ measured_performance - known_overhead
```

## PTX Generation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CudaKernels.generateAddF32(salt)                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PTX Template                                            │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  // Header, registers                               │ │   │
│  │  │  ld.global.f32 %f1, [%rd6];  // Load a[i]          │ │   │
│  │  │  ld.global.f32 %f2, [%rd7];  // Load b[i]          │ │   │
│  │  │                                                     │ │   │
│  │  │  #if SALT >= SALT_TIMING                           │ │   │
│  │  │  mov.u64 %rd_t0, %globaltimer;  // Start timer     │ │   │
│  │  │  #endif                                             │ │   │
│  │  │                                                     │ │   │
│  │  │  add.f32 %f3, %f1, %f2;  // THE ACTUAL OPERATION   │ │   │
│  │  │                                                     │ │   │
│  │  │  #if SALT >= SALT_TIMING                           │ │   │
│  │  │  mov.u64 %rd_t1, %globaltimer;  // End timer       │ │   │
│  │  │  sub.u64 %rd_delta, %rd_t1, %rd_t0;                │ │   │
│  │  │  atom.global.add.u64 [timing], %rd_delta;          │ │   │
│  │  │  #endif                                             │ │   │
│  │  │                                                     │ │   │
│  │  │  st.global.f32 [%rd8], %f3;  // Store result       │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Timing Accumulator

When `SALT_TIMING` is enabled, the kernel receives an additional parameter pointing to a device memory location for accumulating timing data:

```ptx
.visible .entry add_f32(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u64 timing_ptr    // Only present when salt >= SALT_TIMING
)
```

Each thread atomically adds its cycle count to the accumulator. The host can read this after kernel completion to get total cycles across all threads.

## Overhead Quantification

The SALT_TIMING instrumentation adds exactly:
- 2 `mov.u64` instructions (read globaltimer)
- 1 `sub.u64` instruction (compute delta)
- 1 `ld.param.u64` instruction (load timing pointer)
- 1 `atom.global.add.u64` instruction (accumulate)

**Total: 5 instructions, ~8-12 cycles per thread** (architecture dependent)

This overhead is:
1. **Constant** - same for every element
2. **Measurable** - can be calibrated per GPU architecture
3. **Subtractable** - real performance = measured - (overhead × thread_count)

## Integration with warpforge-core-jfr (Future)

The salt level will be configurable via:
1. Backend configuration at construction time
2. Per-operation override for targeted profiling
3. JFR event correlation

```java
// Future API sketch
NvidiaBackend backend = new NvidiaBackend.Builder()
    .device(0)
    .instrumentationSalt(CudaKernels.SALT_TIMING)
    .build();

// Or per-operation
backend.withInstrumentation(SALT_TIMING, () -> {
    backend.execute(addOp, inputs);
});
```

## Vendor Libraries in the Three-Tier Architecture

Vendor libraries (cuBLAS, cuDNN, rocBLAS, MIOpen) are used in the **PRODUCTION** tier for maximum performance. They are black boxes - we can only observe them externally via CUDA/HIP Events and JFR.

### What We CAN Measure (PRODUCTION Tier)

- Wall-clock execution time via CUDA/HIP Events (~0.5μs resolution)
- TFLOPS throughput (computed from dimensions and time)
- Memory transfer bandwidth
- JFR integration for unified Java+GPU observability

### What We CANNOT Measure (Vendor Libraries)

- Per-thread timing
- Memory stall cycles
- Warp divergence
- Register pressure
- Internal algorithmic choices

### Why Not Just Use cuBLAS for Everything?

While cuBLAS provides maximum performance, it lacks observability. We investigated using NVBit to instrument cuBLAS kernels, but the overhead (1.5-5x) defeats the purpose.

The OPTIMIZED_OBSERVABLE tier exists because:

1. **Kernel-internal visibility** - Salt instrumentation shows exactly where cycles are spent
2. **Near-production performance** - Boehm-style optimizations achieve ~93% of cuBLAS
3. **Same code path** - Measurements are directly applicable to tuning
4. **Foundation for fusion** - Custom kernels can be fused; cuBLAS calls cannot

### When to Use Each Tier

```
"My training is slow"           → Start with PRODUCTION (baseline)
"Which op is the bottleneck?"   → PRODUCTION + JFR events
"Why is this GEMM slow?"        → OPTIMIZED_OBSERVABLE (kernel internals)
"Results don't match expected"  → CORRECTNESS (full numerical trace)
```

By implementing custom PTX with salt-based instrumentation alongside vendor library calls, we get:
- Maximum performance when needed (PRODUCTION)
- Full observability when needed (OPTIMIZED_OBSERVABLE)
- Numerical verification when needed (CORRECTNESS)

## Validation Strategy

The CPU backend (`warpforge-backend-cpu`) is the source of truth for correctness. NVIDIA backend output is validated against CPU backend within floating-point tolerance.

```
CPU Backend (reference) ──────────────────────────────────────►
                                                                │
                                                                │ Compare
                                                                │ (tolerance)
                                                                ▼
NVIDIA Backend (custom PTX) ──────────────────────────────────►
```

We do NOT validate NVIDIA against cuBLAS because:
1. cuBLAS uses different parallelization → different FP rounding
2. Would create dependency on black-box implementation
3. Can't instrument cuBLAS to understand differences

## File Locations

### PRODUCTION Tier (Vendor Libraries)

| File | Purpose |
|------|---------|
| `.../cublas/CublasRuntime.java` | FFM bindings to cuBLAS library |
| `.../ops/CublasDotKernel.java` | cuBLAS SGEMM implementation |

### OPTIMIZED_OBSERVABLE / CORRECTNESS Tiers (Custom PTX)

| File | Purpose |
|------|---------|
| `.../cuda/CudaKernels.java` | PTX generation with salt |
| `.../cuda/CudaRuntime.java` | FFM bindings to CUDA Driver API |
| `.../cuda/CudaContext.java` | Context, kernel, and cuBLAS handle management |
| `.../ops/DotKernel.java` | PTX-based matrix multiply (salted) |
| `.../ops/AddKernel.java` | PTX-based elementwise add (salted) |

### Tests

| File | Purpose |
|------|---------|
| `.../AddKernelTest.java` | Unit tests for Add kernel |
| `.../CpuNvidiaComparisonTest.java` | CPU vs NVIDIA integration tests |

## Testing Strategy

### Unit Tests (No CUDA Required)

Tests in `AddKernelTest.java` that don't have `@Tag("nvidia")`:
- PTX generation produces valid output
- PTX includes timing instrumentation when SALT_TIMING is set
- Grid size calculations are correct

### CUDA Hardware Tests

Tests with `@Tag("nvidia")` require actual CUDA hardware:

```bash
./gradlew :warpforge-backend-nvidia:nvidiaBackendTest
```

### CPU vs NVIDIA Comparison Tests

`CpuNvidiaComparisonTest.java` validates NVIDIA against the CPU backend:
- Small, medium, and large tensors
- 2D and 3D tensor shapes
- Edge cases (zeros, negatives, very small/large numbers)
- Timing instrumentation preserves results

The CPU backend is the **source of truth**. If NVIDIA results differ from CPU beyond tolerance, it's a bug in the NVIDIA implementation.

## Running on NVIDIA Hardware

From the NUC or any machine with NVIDIA GPU:

```bash
# Run all NVIDIA tests
./gradlew :warpforge-backend-nvidia:nvidiaBackendTest

# Or via the top-level task
./gradlew nvidiaTest
```

Ensure CUDA is installed and `libcuda.so` is accessible. The test automatically sets `LD_LIBRARY_PATH` to include `/usr/local/cuda/lib64`.
