# Backend Kernel Instrumentation Strategy

This document describes WarpForge's approach to GPU kernel instrumentation, designed to provide meaningful performance observability without introducing Heisenbug scenarios.

**Supported Backends:**
- **NVIDIA** - CUDA Driver API + PTX assembly
- **AMD** - HIP/ROCm + AMDGCN assembly

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
│  OBSERVABLE          │ Boehm-style PTX/AMDGCN with salt        │           │
│  (Custom ISA)        │ instrumentation                         │           │
│                                                                             │
│  CORRECTNESS         ████                                      ~1% perf    │
│  (Naive ISA)         │ Full tracing, numerical verification   │            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier Selection Guide

| Tier | Performance | Observability | Use Case |
|------|-------------|---------------|----------|
| **PRODUCTION** | 100% | External timing (JFR + CUDA/HIP Events) | Training at scale, inference |
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

## NVIDIA PTX Generation Architecture

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

## AMD AMDGCN Generation Architecture

AMD GPUs use AMDGCN (AMD GCN ISA) assembly instead of PTX. Key architectural differences:

| Concept | NVIDIA | AMD |
|---------|--------|-----|
| Thread group | Warp (32 threads) | Wavefront (32 or 64 threads) |
| Execution unit | Streaming Multiprocessor (SM) | Compute Unit (CU) |
| Register file | Unified | Separate SGPR (scalar) + VGPR (vector) |
| ISA format | PTX (virtual) → SASS (native) | AMDGCN (native) |
| Timer access | `%globaltimer` | `s_memrealtime` |

```
┌─────────────────────────────────────────────────────────────────┐
│  AmdKernels.generateAddF32(salt)                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AMDGCN Template                                         │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  // Kernel setup, register allocation               │ │   │
│  │  │  s_load_dwordx2 s[0:1], s[4:5], 0x0  // Load a_ptr │ │   │
│  │  │  s_load_dwordx2 s[2:3], s[4:5], 0x8  // Load b_ptr │ │   │
│  │  │  s_waitcnt lgkmcnt(0)                               │ │   │
│  │  │                                                     │ │   │
│  │  │  global_load_dword v1, v0, s[0:1]    // Load a[i]  │ │   │
│  │  │  global_load_dword v2, v0, s[2:3]    // Load b[i]  │ │   │
│  │  │  s_waitcnt vmcnt(0)                                 │ │   │
│  │  │                                                     │ │   │
│  │  │  #if SALT >= SALT_TIMING                           │ │   │
│  │  │  s_memrealtime s[8:9]                // Start time │ │   │
│  │  │  s_waitcnt lgkmcnt(0)                              │ │   │
│  │  │  #endif                                             │ │   │
│  │  │                                                     │ │   │
│  │  │  v_add_f32 v3, v1, v2                // THE OP     │ │   │
│  │  │                                                     │ │   │
│  │  │  #if SALT >= SALT_TIMING                           │ │   │
│  │  │  s_memrealtime s[10:11]              // End time   │ │   │
│  │  │  s_waitcnt lgkmcnt(0)                              │ │   │
│  │  │  s_sub_u32 s12, s10, s8              // Delta low  │ │   │
│  │  │  s_subb_u32 s13, s11, s9             // Delta high │ │   │
│  │  │  global_atomic_add_x2 v[4:5], v0, s[12:13], ... // Acc │ │
│  │  │  #endif                                             │ │   │
│  │  │                                                     │ │   │
│  │  │  global_store_dword v0, v3, s[6:7]   // Store out  │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### AMD Register Architecture

AMD GCN/RDNA uses a **split register file**:

```
┌─────────────────────────────────────────────────────────────────┐
│  AMD Compute Unit Register Files                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SGPR (Scalar General Purpose Registers)                        │
│  ├── s0-s103: 104 registers per wavefront (RDNA3)              │
│  ├── Used for: addresses, constants, control flow              │
│  └── Shared across all lanes in wavefront                       │
│                                                                 │
│  VGPR (Vector General Purpose Registers)                        │
│  ├── v0-v255: 256 registers per work-item                      │
│  ├── Used for: per-lane data, ALU operands                     │
│  └── Each lane has its own copy                                │
│                                                                 │
│  Special Registers                                              │
│  ├── exec: Execution mask (which lanes are active)             │
│  ├── vcc: Vector condition code (comparison results)           │
│  └── scc: Scalar condition code                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### AMD Wait Count Semantics

AMD requires explicit wait instructions for memory operations:

| Wait Instruction | Purpose |
|------------------|---------|
| `s_waitcnt vmcnt(N)` | Wait until N or fewer vector memory ops pending |
| `s_waitcnt lgkmcnt(N)` | Wait until N or fewer LDS/GDS/scalar-memory ops pending |
| `s_waitcnt expcnt(N)` | Wait until N or fewer exports pending |

**Critical difference from NVIDIA**: PTX memory operations have implicit ordering; AMDGCN requires explicit waits. The salt instrumentation must respect these semantics.

## Timing Accumulator

When `SALT_TIMING` is enabled, the kernel receives an additional parameter pointing to a device memory location for accumulating timing data.

### NVIDIA (PTX)

```ptx
.visible .entry add_f32(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u64 timing_ptr    // Only present when salt >= SALT_TIMING
)
```

### AMD (AMDGCN)

```asm
; Kernel arguments passed via scalar registers or kernarg segment
; s[4:5] = kernarg pointer
; At offset 0x20: timing_ptr (only when salt >= SALT_TIMING)

s_load_dwordx2 s[14:15], s[4:5], 0x20   // Load timing_ptr
s_waitcnt lgkmcnt(0)
```

Each thread/work-item atomically adds its cycle count to the accumulator. The host can read this after kernel completion to get total cycles across all threads/wavefronts.

## Overhead Quantification

### NVIDIA SALT_TIMING Overhead

The instrumentation adds exactly:
- 2 `mov.u64` instructions (read globaltimer)
- 1 `sub.u64` instruction (compute delta)
- 1 `ld.param.u64` instruction (load timing pointer)
- 1 `atom.global.add.u64` instruction (accumulate)

**Total: 5 instructions, ~8-12 cycles per thread**

### AMD SALT_TIMING Overhead

The instrumentation adds exactly:
- 2 `s_memrealtime` instructions (read timer)
- 2 `s_waitcnt lgkmcnt(0)` instructions (wait for timer)
- 2 scalar subtract instructions (compute 64-bit delta)
- 1 `s_load_dwordx2` instruction (load timing pointer)
- 1 `global_atomic_add_x2` instruction (accumulate)

**Total: 8 instructions + 2 waits, ~15-25 cycles per wavefront**

Note: AMD timing is per-wavefront (one result for 32/64 work-items) while NVIDIA is per-thread. Adjust accumulator reads accordingly.

### Properties (Both Vendors)

This overhead is:
1. **Constant** - same for every element
2. **Measurable** - can be calibrated per GPU architecture
3. **Subtractable** - real performance = measured - (overhead × thread_count)

## Integration with warpforge-core-jfr (Future)

The salt level will be configurable via:
1. Backend configuration at construction time
2. Per-operation override for targeted profiling
3. JFR event correlation

### NVIDIA Backend

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

### AMD Backend

```java
// Future API sketch
AmdBackend backend = new AmdBackend.Builder()
    .device(0)
    .instrumentationSalt(AmdKernels.SALT_TIMING)
    .build();

// Or per-operation
backend.withInstrumentation(SALT_TIMING, () -> {
    backend.execute(addOp, inputs);
});
```

### Unified API

```java
// Backend-agnostic profiling
GpuBackend backend = GpuBackend.create(deviceId);  // Auto-detects NVIDIA or AMD
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

### Why Not Just Use cuBLAS/rocBLAS for Everything?

While vendor libraries provide maximum performance, they lack observability:

| Approach | Overhead | Reason |
|----------|----------|--------|
| NVBit (NVIDIA) | 1.5-5x | Dynamic binary instrumentation |
| rocprofiler (AMD) | 1.2-3x | Counter collection overhead |

The OPTIMIZED_OBSERVABLE tier exists because:

1. **Kernel-internal visibility** - Salt instrumentation shows exactly where cycles are spent
2. **Near-production performance** - Boehm-style optimizations achieve ~93% of vendor libs
3. **Same code path** - Measurements are directly applicable to tuning
4. **Foundation for fusion** - Custom kernels can be fused; vendor library calls cannot

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

The CPU backend (`warpforge-backend-cpu`) is the source of truth for correctness. All GPU backends are validated against CPU backend within floating-point tolerance.

```
                            ┌───────────────────────────────────┐
                            │   CPU Backend (reference)         │
                            │   warpforge-backend-cpu           │
                            └───────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │ Compare               │ Compare               │
                    │ (tolerance)           │ (tolerance)           │
                    ▼                       ▼                       │
┌───────────────────────────┐   ┌───────────────────────────┐      │
│ NVIDIA Backend            │   │ AMD Backend               │      │
│ (custom PTX)              │   │ (custom AMDGCN)           │      │
│ warpforge-backend-nvidia  │   │ warpforge-backend-amd     │      │
└───────────────────────────┘   └───────────────────────────┘      │
```

We do NOT validate GPU backends against vendor libraries because:
1. Vendor libs use different parallelization → different FP rounding
2. Would create dependency on black-box implementation
3. Can't instrument vendor libs to understand differences

We do NOT validate NVIDIA against AMD because:
1. Different architectures have different FP rounding behavior
2. Neither is "more correct" than the other
3. CPU backend provides deterministic reference

## File Locations

### NVIDIA Backend

#### PRODUCTION Tier (Vendor Libraries)

| File | Purpose |
|------|---------|
| `.../nvidia/cublas/CublasRuntime.java` | FFM bindings to cuBLAS library |
| `.../nvidia/ops/CublasDotKernel.java` | cuBLAS SGEMM implementation |

#### OPTIMIZED_OBSERVABLE / CORRECTNESS Tiers (Custom PTX)

| File | Purpose |
|------|---------|
| `.../nvidia/cuda/CudaKernels.java` | PTX generation with salt |
| `.../nvidia/cuda/CudaRuntime.java` | FFM bindings to CUDA Driver API |
| `.../nvidia/cuda/CudaContext.java` | Context, kernel, and cuBLAS handle management |
| `.../nvidia/ops/DotKernel.java` | PTX-based matrix multiply (salted) |
| `.../nvidia/ops/AddKernel.java` | PTX-based elementwise add (salted) |

### AMD Backend

#### PRODUCTION Tier (Vendor Libraries)

| File | Purpose |
|------|---------|
| `.../amd/rocblas/RocblasRuntime.java` | FFM bindings to rocBLAS library |
| `.../amd/ops/RocblasDotKernel.java` | rocBLAS SGEMM implementation |

#### OPTIMIZED_OBSERVABLE / CORRECTNESS Tiers (Custom AMDGCN)

| File | Purpose |
|------|---------|
| `.../amd/hip/AmdKernels.java` | AMDGCN generation with salt |
| `.../amd/hip/HipRuntime.java` | FFM bindings to HIP Runtime API |
| `.../amd/hip/HipContext.java` | Context, kernel, and rocBLAS handle management |
| `.../amd/ops/DotKernel.java` | AMDGCN-based matrix multiply (salted) |
| `.../amd/ops/AddKernel.java` | AMDGCN-based elementwise add (salted) |

### Tests

| File | Purpose |
|------|---------|
| `.../nvidia/AddKernelTest.java` | Unit tests for NVIDIA Add kernel |
| `.../nvidia/CpuNvidiaComparisonTest.java` | CPU vs NVIDIA integration tests |
| `.../amd/AddKernelTest.java` | Unit tests for AMD Add kernel |
| `.../amd/CpuAmdComparisonTest.java` | CPU vs AMD integration tests |

## Testing Strategy

### Unit Tests (No GPU Required)

Tests without `@Tag("nvidia")` or `@Tag("amd")`:
- ISA generation produces valid output (PTX for NVIDIA, AMDGCN for AMD)
- Timing instrumentation is correctly inserted when SALT_TIMING is set
- Grid/workgroup size calculations are correct

### NVIDIA Hardware Tests

Tests with `@Tag("nvidia")` require actual CUDA hardware:

```bash
./gradlew :warpforge-backend-nvidia:nvidiaBackendTest
```

### AMD Hardware Tests

Tests with `@Tag("amd")` require actual ROCm hardware:

```bash
./gradlew :warpforge-backend-amd:amdBackendTest
```

### CPU vs GPU Comparison Tests

Both `CpuNvidiaComparisonTest.java` and `CpuAmdComparisonTest.java` validate GPU against the CPU backend:
- Small, medium, and large tensors
- 2D and 3D tensor shapes
- Edge cases (zeros, negatives, very small/large numbers)
- Timing instrumentation preserves results

The CPU backend is the **source of truth**. If GPU results differ from CPU beyond tolerance, it's a bug in the GPU implementation.

## Running on GPU Hardware

### NVIDIA

From the NUC or any machine with NVIDIA GPU:

```bash
# Run all NVIDIA tests
./gradlew :warpforge-backend-nvidia:nvidiaBackendTest

# Or via the top-level task
./gradlew nvidiaTest
```

Ensure CUDA is installed and `libcuda.so` is accessible. The test automatically sets `LD_LIBRARY_PATH` to include `/usr/local/cuda/lib64`.

### AMD

From any machine with AMD GPU and ROCm:

```bash
# Run all AMD tests
./gradlew :warpforge-backend-amd:amdBackendTest

# Or via the top-level task
./gradlew amdTest
```

Ensure ROCm is installed and `libamdhip64.so` is accessible. The test automatically sets `LD_LIBRARY_PATH` to include `/opt/rocm/lib`.

### Required Libraries

| Vendor | Library | Typical Location |
|--------|---------|------------------|
| NVIDIA | libcuda.so | /usr/local/cuda/lib64 |
| NVIDIA | libcublas.so | /usr/local/cuda/lib64 |
| AMD | libamdhip64.so | /opt/rocm/lib |
| AMD | librocblas.so | /opt/rocm/lib |

## AMD-Specific Considerations

### Wavefront Size

AMD GPUs support two wavefront sizes:
- **Wave32** (RDNA/RDNA2/RDNA3): 32 work-items per wavefront
- **Wave64** (GCN/CDNA): 64 work-items per wavefront

The kernel generator must query the device and generate appropriate code:

```java
int wavefrontSize = hipDevice.getWavefrontSize();  // Returns 32 or 64
String isa = AmdKernels.generateAddF32(salt, wavefrontSize);
```

### Memory Ordering

AMD requires explicit memory barriers more often than NVIDIA:

```asm
; After stores that must be visible to other wavefronts
s_waitcnt vmcnt(0)
s_dcache_wb             ; Write back data cache (GCN)
; or
buffer_gl0_inv          ; Invalidate GL0 (RDNA)
```

### Occupancy Differences

| Metric | NVIDIA (Ada) | AMD (RDNA3) |
|--------|--------------|-------------|
| Max waves per CU/SM | 64 warps | 32 wavefronts |
| Registers per CU/SM | 65536 | 65536 |
| Shared mem per CU/SM | 100KB | 64KB LDS |
| Typical occupancy target | 50-75% | 50-75% |

Salt instrumentation register usage must stay within occupancy-friendly limits on both architectures.
