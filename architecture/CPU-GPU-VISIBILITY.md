# CPU-GPU Boundary Visibility Architecture

This document defines WarpForge's approach to expressing and visualizing the CPU↔GPU boundary - the mapping from Java virtual threads to GPU kernel execution. This is a core competitive differentiator: **understanding what the GPU is actually doing when Java code runs**.

## The Problem: Half-Blind Execution

Current state of GPU profiling from Java:

```
Java VThread → GpuTaskScope → Stream → Kernel → [wall-clock time]
                                                        ↓
                                              "It took 1,234μs"
                                              "It achieved 27.3 TFLOPS"
                                                        ↓
                                              ??? (black box)
```

We know *how long* and *how fast*, but not:
- How many GPU threads were spawned
- Which SMs executed the work
- What percentage of GPU capacity was used
- Whether compute-bound or memory-bound
- How it interacted with concurrent kernels

## The Vision: Full Boundary Expression

```
+-------------------------------------------------------------------------+
|  Java World                    |  GPU World                             |
+--------------------------------+----------------------------------------+
|  1 Virtual Thread              |  16,777,216 GPU threads                |
|  1 GpuTaskScope                |  1-N CUDA/HIP streams                  |
|  1 kernel.execute() call       |  256×256 blocks × 256 threads/block    |
|  ~10μs CPU time (launch)       |  1,234μs GPU time (compute)            |
|  Blocking on scope.join()      |  SMs 0-99 active, warps interleaved    |
+--------------------------------+----------------------------------------+
```

Every Java virtual thread's GPU work should be fully characterized, not just timed.

## Enhanced JFR Event Schema

### GpuKernelEvent (Enhanced)

```java
@Label("GPU Kernel Execution")
@Category({"WarpForge", "GPU"})
@Description("Records GPU kernel execution with full launch configuration")
public class GpuKernelEvent extends jdk.jfr.Event {

    // ==================== Identity ====================

    @Label("Operation")
    public String operation;                    // "GEMM", "Conv2D", "Add"

    @Label("Shape")
    public String shape;                        // "4096x4096 * 4096x4096"

    @Label("Backend")
    public String backend;                      // "cuBLAS", "PTX", "rocBLAS", "HIP"

    @Label("Tier")
    public String tier;                         // "PRODUCTION", "OPTIMIZED_OBSERVABLE"

    @Label("Device Index")
    public int deviceIndex;

    // ==================== Timing ====================

    @Label("GPU Time")
    @Timespan(Timespan.MICROSECONDS)
    public long gpuTimeMicros;                  // Kernel execution time

    @Label("Launch Latency")
    @Timespan(Timespan.NANOSECONDS)
    public long launchLatencyNanos;             // CPU→GPU submission overhead

    @Label("Queue Delay")
    @Timespan(Timespan.NANOSECONDS)
    public long queueDelayNanos;                // Time waiting in stream queue

    // ==================== Throughput ====================

    @Label("Throughput TFLOPS")
    public double teraflops;

    @Label("Memory Bandwidth GB/s")
    public double memoryBandwidthGBps;

    @Label("Bytes Transferred")
    @DataAmount
    public long bytesTransferred;

    // ==================== Launch Configuration ====================

    @Label("Grid Dim X")
    public int gridDimX;                        // Number of blocks in X

    @Label("Grid Dim Y")
    public int gridDimY;                        // Number of blocks in Y

    @Label("Grid Dim Z")
    public int gridDimZ;                        // Number of blocks in Z

    @Label("Block Dim X")
    public int blockDimX;                       // Threads per block in X

    @Label("Block Dim Y")
    public int blockDimY;                       // Threads per block in Y

    @Label("Block Dim Z")
    public int blockDimZ;                       // Threads per block in Z

    @Label("Total Threads")
    public long totalThreads;                   // gridDim * blockDim (computed)

    @Label("Total Warps")
    public long totalWarps;                     // totalThreads / 32 (NVIDIA) or /64 (AMD)

    @Label("Total Blocks")
    public int totalBlocks;                     // gridDimX * gridDimY * gridDimZ

    // ==================== Resource Usage ====================

    @Label("Registers Per Thread")
    public int registersPerThread;              // From cudaFuncGetAttributes

    @Label("Static Shared Memory")
    @DataAmount
    public int staticSharedMemoryBytes;         // Compiled into kernel

    @Label("Dynamic Shared Memory")
    @DataAmount
    public int dynamicSharedMemoryBytes;        // Passed at launch

    @Label("Local Memory Per Thread")
    @DataAmount
    public int localMemoryPerThread;            // Spilled registers

    // ==================== Occupancy ====================

    @Label("Theoretical Occupancy %")
    public int theoreticalOccupancyPercent;     // From cudaOccupancyMaxActiveBlocksPerMultiprocessor

    @Label("Max Active Blocks Per SM")
    public int maxActiveBlocksPerSM;            // Hardware limit for this kernel

    @Label("Estimated Active SMs")
    public int estimatedActiveSMs;              // min(totalBlocks, smCount)

    @Label("Estimated Active Warps")
    public long estimatedActiveWarps;           // Concurrent warps across GPU

    // ==================== Stream Context ====================

    @Label("Stream ID")
    public long streamId;                       // CUDA/HIP stream handle

    @Label("Stream Priority")
    public int streamPriority;                  // -1 (high) to 0 (default)

    @Label("Concurrent Kernels")
    public int concurrentKernels;               // Other kernels on other streams

    // ==================== Java Context ====================

    @Label("Virtual Thread ID")
    public long virtualThreadId;                // Thread.currentThread().threadId()

    @Label("Scope ID")
    public String scopeId;                      // GpuTaskScope identifier

    @Label("Scope Name")
    public String scopeName;                    // Human-readable scope name
}
```

### GpuOccupancyEvent (New)

For tracking occupancy changes over time:

```java
@Label("GPU Occupancy Snapshot")
@Category({"WarpForge", "GPU", "Occupancy"})
@Description("Periodic snapshot of GPU occupancy state")
public class GpuOccupancyEvent extends jdk.jfr.Event {

    @Label("Device Index")
    public int deviceIndex;

    @Label("Active Streams")
    public int activeStreams;

    @Label("Active Kernels")
    public int activeKernels;

    @Label("Estimated Total Occupancy %")
    public int estimatedTotalOccupancyPercent;

    @Label("NVML GPU Utilization %")
    public int nvmlGpuUtilization;              // From NVML/SMI

    @Label("NVML Memory Utilization %")
    public int nvmlMemoryUtilization;

    @Label("Active Warps Estimate")
    public long activeWarpsEstimate;

    @Label("SM Count")
    public int smCount;

    @Label("Max Warps Per SM")
    public int maxWarpsPerSM;
}
```

## FFM Bindings Required

### NVIDIA (CudaRuntime additions)

```java
// ==================== Kernel Attributes ====================

/**
 * Get kernel resource requirements.
 *
 * struct cudaFuncAttributes {
 *     size_t sharedSizeBytes;      // Static shared memory
 *     size_t constSizeBytes;       // Constant memory
 *     size_t localSizeBytes;       // Local memory per thread
 *     int maxThreadsPerBlock;      // Max threads for this kernel
 *     int numRegs;                 // Registers per thread
 *     int ptxVersion;              // PTX version
 *     int binaryVersion;           // Binary (cubin) version
 *     int cacheModeCA;             // Cache mode
 *     int maxDynamicSharedSizeBytes;
 *     int preferredShmemCarveout;
 * };
 */
public record KernelAttributes(
    long sharedSizeBytes,
    long localSizeBytes,
    int maxThreadsPerBlock,
    int numRegs,
    int ptxVersion,
    int binaryVersion
) {}

public static KernelAttributes funcGetAttributes(long kernelFunction);

// ==================== Occupancy Calculation ====================

/**
 * Calculate max active blocks per SM for given kernel and block size.
 *
 * This is the key API for occupancy estimation.
 */
public static int occupancyMaxActiveBlocksPerMultiprocessor(
    long kernelFunction,
    int blockSize,
    long dynamicSharedMemBytes
);

/**
 * Suggest optimal block size for maximum occupancy.
 */
public static int occupancyMaxPotentialBlockSize(
    long kernelFunction,
    long dynamicSharedMemBytesPerBlock,
    int blockSizeLimit
);

// ==================== Device Properties ====================

/**
 * Get SM count for device.
 */
public static int deviceGetAttribute(int attribute, int device);

// Attribute constants
public static final int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
public static final int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39;
public static final int CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82;
public static final int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81;
```

### AMD (HipRuntime additions) - PARITY REQUIRED

```java
// ==================== Kernel Attributes ====================

/**
 * HIP equivalent of cudaFuncGetAttributes.
 */
public record HipFuncAttributes(
    long sharedSizeBytes,
    long localSizeBytes,
    int maxThreadsPerBlock,
    int numRegs,
    int ptxVersion,      // Actually GCN version for AMD
    int binaryVersion
) {}

public static HipFuncAttributes funcGetAttributes(long kernelFunction);

// ==================== Occupancy Calculation ====================

/**
 * HIP equivalent of cudaOccupancyMaxActiveBlocksPerMultiprocessor.
 */
public static int occupancyMaxActiveBlocksPerMultiprocessor(
    long kernelFunction,
    int blockSize,
    long dynamicSharedMemBytes
);

/**
 * Suggest optimal block size.
 */
public static int occupancyMaxPotentialBlockSize(
    long kernelFunction,
    long dynamicSharedMemBytesPerBlock,
    int blockSizeLimit
);

// ==================== Device Properties ====================

public static int deviceGetAttribute(int attribute, int device);

// AMD uses Compute Units (CUs) instead of SMs
public static final int HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = ...;
public static final int HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = ...;
```

## Occupancy Calculation

### Theoretical Occupancy Formula

```
Occupancy = (Active Warps per SM) / (Max Warps per SM)

Active Warps per SM = min(
    floor(Max Warps per SM),
    floor(Registers per SM / (Registers per Thread × Warp Size)),
    floor(Shared Memory per SM / Shared Memory per Block) × (Block Size / Warp Size),
    Max Blocks per SM × (Block Size / Warp Size)
)
```

### Java Implementation

```java
public class OccupancyCalculator {

    private final int smCount;
    private final int maxWarpsPerSM;
    private final int maxRegistersPerSM;
    private final int maxSharedMemoryPerSM;
    private final int maxBlocksPerSM;
    private final int warpSize;  // 32 for NVIDIA, 32 or 64 for AMD

    public OccupancyCalculator(GpuBackend backend) {
        this.smCount = backend.getSmCount();
        this.maxWarpsPerSM = backend.getMaxWarpsPerSM();
        this.maxRegistersPerSM = backend.getMaxRegistersPerSM();
        this.maxSharedMemoryPerSM = backend.getMaxSharedMemoryPerSM();
        this.maxBlocksPerSM = backend.getMaxBlocksPerSM();
        this.warpSize = backend.getWarpSize();
    }

    public OccupancyInfo calculate(KernelAttributes attrs, int blockSize, int dynamicSharedMem) {
        int warpsPerBlock = (blockSize + warpSize - 1) / warpSize;
        int totalSharedMem = (int) attrs.sharedSizeBytes() + dynamicSharedMem;

        // Constraint 1: Register limit
        int warpsLimitedByRegs = maxRegistersPerSM / (attrs.numRegs() * warpSize);

        // Constraint 2: Shared memory limit
        int blocksLimitedByShared = totalSharedMem > 0
            ? maxSharedMemoryPerSM / totalSharedMem
            : maxBlocksPerSM;
        int warpsLimitedByShared = blocksLimitedByShared * warpsPerBlock;

        // Constraint 3: Block count limit
        int warpsLimitedByBlocks = maxBlocksPerSM * warpsPerBlock;

        // Constraint 4: Warp limit
        int warpsLimitedByMax = maxWarpsPerSM;

        // Take minimum
        int activeWarpsPerSM = Math.min(
            Math.min(warpsLimitedByRegs, warpsLimitedByShared),
            Math.min(warpsLimitedByBlocks, warpsLimitedByMax)
        );

        float occupancy = (float) activeWarpsPerSM / maxWarpsPerSM;

        return new OccupancyInfo(
            (int) (occupancy * 100),
            activeWarpsPerSM,
            activeWarpsPerSM / warpsPerBlock,  // maxActiveBlocksPerSM
            getLimitingFactor(warpsLimitedByRegs, warpsLimitedByShared,
                             warpsLimitedByBlocks, warpsLimitedByMax)
        );
    }

    public record OccupancyInfo(
        int occupancyPercent,
        int activeWarpsPerSM,
        int maxActiveBlocksPerSM,
        String limitingFactor  // "registers", "shared_memory", "blocks", "warps"
    ) {}
}
```

## Visualization Model

### Timeline View: VThread → GPU Mapping

```
Time (ms)     0    1    2    3    4    5    6    7    8    9   10
              │    │    │    │    │    │    │    │    │    │    │
VThread-1:    ▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓
              launch │←── waiting ──────→│ continue
                     │
                     ▼
Stream-0:            ████████████████████
                     │← GEMM 4096×4096 →│

SM View:             ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░
                     75% occupancy
                     100/132 SMs active
                     512K warps
```

### Concurrent Kernel View

```
Time (ms)     0    1    2    3    4    5    6
              │    │    │    │    │    │    │
Stream-0:     ████████████████████          GEMM (75% occ)
Stream-1:          ▓▓▓▓▓▓▓▓                 Add (10% occ)
Stream-2:               ▒▒▒▒▒▒▒▒▒▒▒▒        Softmax (15% occ)
              │    │    │    │    │    │    │
Total Occ:    75%  85%  100% 100% 90%  15%  0%
              ─────────────────────────────────
              │         DANGER ZONE          │
              │    GPU may be oversubscribed │
```

### JDK Mission Control Integration

Custom views for JMC showing GPU kernel details:

```
+------------------------------------------------------------------+
|  GPU Kernel Details                                               |
+------------------------------------------------------------------+
|  Operation: GEMM                                                  |
|  Shape: 4096×4096 × 4096×4096                                     |
|  Backend: cuBLAS                                                  |
+------------------------------------------------------------------+
|  Launch Configuration                                             |
|  ┌────────────────────────────────────────────────────────────┐  |
|  │  Grid:  256 × 256 × 1 = 65,536 blocks                      │  |
|  │  Block: 256 × 1 × 1   = 256 threads                        │  |
|  │  Total: 16,777,216 threads (524,288 warps)                 │  |
|  └────────────────────────────────────────────────────────────┘  |
+------------------------------------------------------------------+
|  Resource Usage                                                   |
|  ┌────────────────────────────────────────────────────────────┐  |
|  │  Registers:    32 per thread                               │  |
|  │  Shared Mem:   48 KB per block (static) + 0 (dynamic)      │  |
|  │  Local Mem:    0 bytes per thread                          │  |
|  └────────────────────────────────────────────────────────────┘  |
+------------------------------------------------------------------+
|  Occupancy Analysis                                               |
|  ┌────────────────────────────────────────────────────────────┐  |
|  │  Theoretical: 75% ████████████████████░░░░░░░              │  |
|  │  Limiting Factor: Registers (32 regs × 256 threads)        │  |
|  │  Active Blocks/SM: 6 of 8 max                              │  |
|  │  Active Warps/SM: 48 of 64 max                             │  |
|  │  Active SMs: ~100 of 132 (blocks spread across SMs)        │  |
|  └────────────────────────────────────────────────────────────┘  |
+------------------------------------------------------------------+
|  Timing                                                           |
|  ┌────────────────────────────────────────────────────────────┐  |
|  │  Launch Latency:  8.2 μs (CPU → GPU submission)            │  |
|  │  Queue Delay:     0.3 μs (waiting in stream)               │  |
|  │  GPU Execution:   1,234 μs                                 │  |
|  │  Total:           1,242.5 μs                               │  |
|  │                                                            │  |
|  │  Throughput:      27.3 TFLOPS (82% of peak)                │  |
|  │  Memory BW:       2.1 TB/s (63% of peak)                   │  |
|  └────────────────────────────────────────────────────────────┘  |
+------------------------------------------------------------------+
|  Java Context                                                     |
|  ┌────────────────────────────────────────────────────────────┐  |
|  │  Virtual Thread: VThread-47 (id=1247)                      │  |
|  │  Task Scope:     inference-batch-12                        │  |
|  │  Stream:         0x7f8a4c000800 (priority: default)        │  |
|  └────────────────────────────────────────────────────────────┘  |
+------------------------------------------------------------------+
```

## Implementation Phases

### Phase 1: Basic Launch Configuration Capture

**Goal:** Capture grid/block dimensions and compute thread counts.

1. Modify kernel dispatch wrappers to capture launch config
2. Add fields to GpuKernelEvent
3. Compute totalThreads, totalWarps, totalBlocks
4. **Both NVIDIA and AMD** - enforce parity

### Phase 2: Kernel Attributes and Occupancy

**Goal:** Query kernel resource usage and estimate occupancy.

1. Add `cudaFuncGetAttributes` / `hipFuncGetAttributes` FFM bindings
2. Add `cudaOccupancyMaxActiveBlocksPerMultiprocessor` / HIP equivalent
3. Implement OccupancyCalculator
4. Add occupancy fields to GpuKernelEvent
5. **Both NVIDIA and AMD** - enforce parity

### Phase 3: Launch Overhead Measurement

**Goal:** Measure CPU→GPU submission latency.

1. Timestamp before kernel launch (System.nanoTime())
2. Record CUDA/HIP event immediately after launch
3. Calculate launch latency
4. Add launchLatencyNanos to GpuKernelEvent

### Phase 4: Stream Context and Concurrency

**Goal:** Track concurrent kernel execution.

1. Track stream IDs in GpuKernelEvent
2. Maintain count of active kernels across streams
3. Add GpuOccupancyEvent for periodic snapshots
4. Enable concurrent kernel visualization

### Phase 5: Advanced CUPTI/roctracer Integration

**Goal:** Hardware counter-based achieved occupancy.

1. CUPTI activity callbacks for actual SM metrics
2. roctracer equivalent for AMD
3. L1/L2 cache hit rates
4. Warp stall analysis

## Competitive Differentiation

This CPU↔GPU visibility is a **core competitive advantage** for WarpForge:

| Competitor | GPU Visibility |
|------------|----------------|
| PyTorch | torch.profiler shows kernel names, times; no Java integration |
| TensorFlow | XLA profiler; separate from application profiling |
| JAX | XLA profiler; no JVM visibility |
| ONNX Runtime | Basic timing; no launch config |
| **WarpForge** | Full JFR integration, launch config, occupancy, VThread mapping |

**Why this matters:**
- Developers can correlate Java virtual thread behavior with GPU kernel execution
- Optimization is guided by actual resource usage, not guesswork
- Debugging "why is my model slow?" becomes tractable
- Multi-tenant GPU scheduling is informed by real occupancy data

## Critical: Virtual Thread Naming

**Research finding from loom-dev mailing list:** JFR events recorded on virtual threads have empty 'Event Thread' fields unless the virtual thread is explicitly named. This causes Java Mission Control to collapse all virtual thread events into a single unnamed thread.

**WarpForge MUST name all virtual threads meaningfully:**

```java
// REQUIRED: Name virtual threads with scope and task context
public static Thread createGpuTask(String scopeId, int taskId, Runnable task) {
    return Thread.ofVirtual()
        .name("warpforge-gpu-" + scopeId + "-task-" + taskId)
        .start(task);
}

// BAD: Anonymous virtual threads - JFR events will have empty thread field
Thread.ofVirtual().start(task);  // Don't do this!
```

This naming convention enables:
1. JFR events to be correlated with specific GPU operations
2. Mission Control to group events by virtual thread
3. Debugging of concurrent GPU work across many virtual threads

See [LOOM-DEBUGGING.md](LOOM-DEBUGGING.md) for the full research findings on virtual thread debugging limitations.

---

## Related Documents

- [JFR-GPU.md](JFR-GPU.md) - JFR event definitions and integration
- [GPU-SCHEDULING.md](GPU-SCHEDULING.md) - Capacity-aware scheduling architecture
- [BACKEND-KERNEL-INSTRUMENTATION.md](BACKEND-KERNEL-INSTRUMENTATION.md) - Salt-based kernel instrumentation
- [STRUCTURED-CONCURRENCY-RESEARCH.md](STRUCTURED-CONCURRENCY-RESEARCH.md) - Virtual thread integration
- [LOOM-DEBUGGING.md](LOOM-DEBUGGING.md) - Virtual thread debugging challenges (companion doc)

## References

### NVIDIA Documentation
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
- [CUDA Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)
- [cudaFuncGetAttributes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)
- [CUPTI User's Guide](https://docs.nvidia.com/cupti/main/main.html)
- [Nsight Compute Occupancy](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy)

### AMD Documentation
- [HIP Programming Guide - Occupancy](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/occupancy.html)
- [ROCm Profiler](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [hipFuncGetAttributes](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group__Execution.html)

### Research Papers
- "Demystifying GPU Microarchitecture through Microbenchmarking" (ISPASS 2010)
- "Occupancy-based Compilation" (CGO 2020)
- "Understanding GPU Scheduling" (MICRO 2016)
