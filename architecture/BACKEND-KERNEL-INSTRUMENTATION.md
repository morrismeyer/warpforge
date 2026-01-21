# Backend Kernel Instrumentation Strategy

This document describes WarpForge's approach to GPU kernel instrumentation, designed to provide meaningful performance observability without introducing Heisenbug scenarios.

## Core Principle: One Implementation, Instrumentation as Salt

**There is exactly ONE implementation of each kernel operation.** Instrumentation variants are generated from the same code template with conditional sections enabled based on the "salt level."

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

## Why Not cuBLAS/cuDNN?

Vendor libraries (cuBLAS, cuDNN) are black boxes. We can wrap them in JFR events to measure call latency, but:

- No visibility into internal kernel behavior
- No ability to instrument the actual computation
- No path to fusion (Phase 2+)
- Different code path = Heisenbug potential if we have both

By implementing all operations as custom PTX with salt-based instrumentation, we get:
- Full observability at any granularity
- Single code path for production and profiling
- Foundation for Babylon-based kernel fusion
- Consistent, comparable measurements

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

| File | Purpose |
|------|---------|
| `warpforge-backend-nvidia/src/main/java/.../cuda/CudaKernels.java` | PTX generation with salt |
| `warpforge-backend-nvidia/src/main/java/.../cuda/CudaRuntime.java` | FFM bindings to CUDA Driver API |
| `warpforge-backend-nvidia/src/main/java/.../cuda/CudaContext.java` | Context and kernel management |
| `warpforge-backend-nvidia/src/main/java/.../ops/AddKernel.java` | Add operation implementation |
| `warpforge-backend-nvidia/src/test/java/.../AddKernelTest.java` | Unit tests for Add kernel |
| `warpforge-backend-nvidia/src/test/java/.../CpuNvidiaComparisonTest.java` | CPU vs NVIDIA integration tests |

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
