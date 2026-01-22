# GPU Scheduling Architecture

This document describes WarpForge's approach to GPU resource scheduling, leveraging Java's Virtual Threads and Structured Concurrency to maximize GPU utilization. This is a core architectural differentiator: **higher GPU utilization directly translates to lower costs for AI/ML workloads**.

## The Vision: Java-Controlled GPU Parallelism

Traditional GPU programming treats the GPU as a black box: submit work, wait for completion. WarpForge aims to bring the sophistication of modern CPU schedulers to GPU resource management, using Java 21+ concurrency primitives:

```
+-------------------------------------------------------------------------+
|  Java Structured Concurrency (Virtual Threads)                          |
|  +-------------------------------------------------------------------+  |
|  |  StructuredTaskScope.ShutdownOnFailure                            |  |
|  |  +---------+ +---------+ +---------+ +---------+ +---------+      |  |
|  |  |VThread 1| |VThread 2| |VThread 3| |VThread 4| |VThread N|      |  |
|  |  |Model A  | |Model B  | |Batch 1  | |Batch 2  | |  ...    |      |  |
|  |  +----+----+ +----+----+ +----+----+ +----+----+ +----+----+      |  |
|  +-------+----------+----------+----------+----------+---------------+  |
|          |          |          |          |          |                  |
|  +-------v----------v----------v----------v----------v---------------+  |
|  |  GpuScheduler (Capacity-Aware)                                    |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  |  Resource Model:                                            |  |  |
|  |  |  - SM allocation tracking                                   |  |  |
|  |  |  - Memory pressure monitoring                               |  |  |
|  |  |  - Stream pool management                                   |  |  |
|  |  |  - Occupancy estimation per kernel type                     |  |  |
|  |  |  - Real-time utilization feedback (NVML/SMI)                |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |                                                                   |  |
|  |  acquireResources(kernelType, shape) -> GpuLease | BLOCK          |  |
|  |  releaseResources(lease)                                          |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |  CUDA/HIP Streams (Concurrent Execution Channels)                 |  |
|  |  +--------+ +--------+ +--------+ +--------+ +--------+           |  |
|  |  |Stream 0| |Stream 1| |Stream 2| |Stream 3| |  ...   |           |  |
|  |  |[GEMM]  | |[Conv]  | |[Add]   | |[idle]  | |        |           |  |
|  |  +--------+ +--------+ +--------+ +--------+ +--------+           |  |
|  +-------------------------------------------------------------------+  |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |  GPU Hardware                                                     |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  |  Streaming Multiprocessors (SMs)                            |  |  |
|  |  |  [SM0][SM1][SM2]...[SM131]  (H100: 132 SMs)                  |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  +-------------------------------------------------------------+  |  |
|  |  |  HBM Memory: 80GB (H100)                                    |  |  |
|  |  |  Memory Bandwidth: 3.35 TB/s                                |  |  |
|  |  +-------------------------------------------------------------+  |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

## GPU Resource Model

Understanding GPU resources is essential for intelligent scheduling.

### Streaming Multiprocessors (SMs)

The SM is the fundamental compute unit. Modern GPUs have many SMs:

|--------|---------|----------|------------|---------------|
| GPU    | SMs     | Warps/SM | Threads/SM | Total Threads |
|--------|---------|----------|------------|---------------|
| A100   | 108     | 64       | 2048       | 221,184       |
| H100   | 132     | 64       | 2048       | 270,336       |
| MI300X | 304 CUs | 64       | 2048       | 622,592       |
|--------|---------|----------|------------|---------------|

Each SM has:
- **Warp schedulers** (4 per SM on modern NVIDIA)
- **CUDA cores** (FP32, FP64, INT32 units)
- **Tensor cores** (for matrix operations)
- **Shared memory** (configurable, up to 228KB on H100)
- **Register file** (65536 32-bit registers per SM)

### Occupancy

[Occupancy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html) is the ratio of active warps to maximum warps per SM. It depends on:

1. **Registers per thread** - More registers → fewer concurrent threads
2. **Shared memory per block** - More shared memory → fewer concurrent blocks
3. **Block size** - Must be multiple of warp size (32)

```java
// CUDA Occupancy API
int maxActiveBlocksPerSM = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    kernel, blockSize, dynamicSharedMem);
float occupancy = (float)(maxActiveBlocksPerSM * blockSize) / maxThreadsPerSM;
```

**Key insight**: A kernel with 50% occupancy uses only half the SM's warp slots, leaving room for concurrent kernels.

### Memory Hierarchy

|-------------|--------------|--------------|---------------------|
| Level       | Capacity     | Latency      | Bandwidth           |
|-------------|--------------|--------------|---------------------|
| Registers   | 256KB/SM     | 0 cycles     | -                   |
| L1/Shared   | 228KB/SM     | ~30 cycles   | ~19 TB/s aggregate  |
| L2 Cache    | 50MB (H100)  | ~200 cycles  | ~12 TB/s            |
| HBM         | 80GB (H100)  | ~400 cycles  | 3.35 TB/s           |
| PCIe/NVLink | Host RAM     | ~10us        | 64-900 GB/s         |
|-------------|--------------|--------------|---------------------|

Memory bandwidth is often the bottleneck for ML workloads, not compute.

## Hardware Partitioning Mechanisms

### NVIDIA MIG (Multi-Instance GPU)

[MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) provides hardware-level isolation on A100/H100:

```
+-----------------------------------------------------------------+
|  H100 GPU (132 SMs, 80GB HBM)                                   |
+-----------------------------------------------------------------+
|  MIG Profiles:                                                  |
|  +-------------+ +-------------+ +-------------+                |
|  | 1g.10gb     | | 1g.10gb     | | 1g.10gb     |  ... (7x)      |
|  | ~19 SMs     | | ~19 SMs     | | ~19 SMs     |                |
|  | 10GB HBM    | | 10GB HBM    | | 10GB HBM    |                |
|  +-------------+ +-------------+ +-------------+                |
|                                                                 |
|  Or:                                                            |
|  +-------------------------+ +-------------------------+        |
|  | 3g.40gb                 | | 3g.40gb                 |        |
|  | ~56 SMs                 | | ~56 SMs                 |        |
|  | 40GB HBM                | | 40GB HBM                |        |
|  +-------------------------+ +-------------------------+        |
+-----------------------------------------------------------------+
```

**Benefits**:
- Hardware isolation (one instance can't starve others)
- Guaranteed memory bandwidth
- Fault isolation

**Limitations**:
- Fixed partition sizes
- Must be configured at boot/admin time
- Not dynamic

### NVIDIA MPS (Multi-Process Service)

[MPS](https://docs.nvidia.com/deploy/mps/index.html) enables software-level sharing:

```
+-----------------------------------------------------------------+
|  MPS Server (Single CUDA Context)                               |
|  +----------+ +----------+ +----------+ +----------+            |
|  | Client 1 | | Client 2 | | Client 3 | | Client 4 |  ...       |
|  | (20% SM) | | (30% SM) | | (25% SM) | | (25% SM) |            |
|  +----------+ +----------+ +----------+ +----------+            |
|                     |                                           |
|                     v                                           |
|  +-------------------------------------------------------------+|
|  |  GPU Hardware (shared context, concurrent execution)        ||
|  +-------------------------------------------------------------+|
+-----------------------------------------------------------------+
```

**Key capability**: `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` controls SM allocation per client.

**Benefits**:
- Arbitrary partition ratios (e.g., 70%/30%)
- Dynamic (no reboot required)
- Up to 48-60 concurrent clients (Volta+)

**WarpForge opportunity**: Use MPS to implement fine-grained SM allocation controlled by the Java scheduler.

### AMD Equivalent

AMD provides similar capabilities via:
- **Compute Partitioning** on MI300X
- **Resource Management** via ROCm runtime

## Software Scheduling Strategies

### Stream-Based Concurrency

CUDA streams enable [concurrent kernel execution](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf):

```java
// Multiple streams for concurrent execution
CudaStream stream1 = context.createStream();
CudaStream stream2 = context.createStream();
CudaStream stream3 = context.createStream();

// Kernels on different streams can execute concurrently
// if GPU has available resources
launchKernel(gemmKernel, stream1, ...);  // Uses ~60% of SMs
launchKernel(addKernel, stream2, ...);   // Uses ~10% of SMs
launchKernel(softmax, stream3, ...);     // Uses ~20% of SMs
// All three may run concurrently!
```

**Concurrency limits** (architecture-dependent):
- Fermi: 16-way concurrency
- Kepler+: 32-way concurrency
- Actual concurrency depends on resource availability

### CUDA Graphs

[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) reduce launch overhead by batching operations:

```
+-----------------------------------------------------------------+
|  CUDA Graph (captured workflow)                                 |
|                                                                 |
|  +------+     +------+     +------+                             |
|  | H2D  |---->| GEMM |---->| D2H  |                             |
|  +------+     +------+     +------+                             |
|      |            |            |                                |
|      |        +------+         |                                |
|      +------->| Add  |---------+                                |
|               +------+                                          |
|                                                                 |
|  Single launch: ~2.5us + ~1ns/node (vs ~10us per kernel)        |
+-----------------------------------------------------------------+
```

**Benefits**:
- 60ns/node reduction in inter-kernel latency
- Optimal for iterative workloads (training loops)
- Batch 50-100 nodes per graph for best performance

### Preemption and Time-Slicing

Recent research enables fine-grained GPU preemption:

|-------------------------------------------------------------|------------------------|-----------------|
| System                                                      | Preemption Granularity | Approach        |
|-------------------------------------------------------------|------------------------|-----------------|
| [LithOS](https://www.cs.cmu.edu/~csd-phd-blog/2025/lithos/) | 500us quantum          | Kernel slicing  |
| [Hummingbird](https://arxiv.org/html/2601.04071v1)          | Microseconds           | SLO-aware sched |
| [GCAPS](https://arxiv.org/html/2406.05221v1)                | Context-based          | Driver-level    |
|-------------------------------------------------------------|------------------------|-----------------|

**WarpForge opportunity**: Implement kernel slicing for long-running operations to enable responsive preemption.

## Monitoring and Feedback

### NVML (NVIDIA Management Library)

[NVML](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) provides real-time GPU metrics:

```java
// Get GPU utilization (% of time kernels were running)
nvmlDeviceGetUtilizationRates(device, utilization);
int gpuUtil = utilization.gpu;    // 0-100%
int memUtil = utilization.memory; // 0-100%

// Get memory info
nvmlDeviceGetMemoryInfo(device, memory);
long used = memory.used;
long free = memory.free;
long total = memory.total;
```

**Sampling period**: 166ms - 1 second (device-dependent)

**Important caveat**: NVML's "utilization" means "% of time any kernel was running", not "% of compute capacity used". A kernel using 10% of SMs still shows 100% utilization while running.

### AMD SMI

[AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.1.0/doxygen/docBin/html/group__gpumon.html) provides equivalent metrics:

```java
// Get GPU busy percentage
rsmi_dev_busy_percent_get(device, &busy_percent);

// Get memory usage
rsmi_dev_memory_usage_get(device, RSMI_MEM_TYPE_VRAM, &used);
rsmi_dev_memory_total_get(device, RSMI_MEM_TYPE_VRAM, &total);
```

### Hardware Performance Counters (CUPTI)

For deeper insight, [CUPTI](https://docs.nvidia.com/cupti/main/main.html) provides hardware counters:

|------------------|------------------------------------------|---------------------------|
| Counter Category | Examples                                 | Use Case                  |
|------------------|------------------------------------------|---------------------------|
| SM Occupancy     | active_warps, theoretical_warps          | Capacity estimation       |
| Memory           | dram_read_transactions, l2_hit_rate      | Bandwidth analysis        |
| Compute          | sm_efficiency, achieved_occupancy        | Utilization analysis      |
| Stalls           | stall_memory_dependency, stall_barrier   | Bottleneck identification |
|------------------|------------------------------------------|---------------------------|

**Warp stall reasons** reveal why performance is lost:
- **Memory dependency**: Waiting for memory operations
- **Barrier**: Waiting for `__syncthreads()`
- **Execution dependency**: Waiting for previous instruction result
- **Not selected**: Eligible but scheduler chose another warp

### JFR Integration

All metrics flow into JFR for unified observability (see `JFR-GPU.md`):

```java
@Label("GPU Scheduler Decision")
@Category({"WarpForge", "GPU", "Scheduler"})
public class GpuSchedulerEvent extends jdk.jfr.Event {
    @Label("Action") public String action;  // "acquire", "release", "block"
    @Label("Kernel Type") public String kernelType;
    @Label("Estimated Occupancy") public float estimatedOccupancy;
    @Label("Current GPU Util %") public int currentUtilization;
    @Label("Active Streams") public int activeStreams;
    @Label("Wait Time") @Timespan public long waitTimeNanos;
}
```

## WarpForge Scheduler Architecture

### Core Components

```java
public class GpuScheduler implements AutoCloseable {

    // Resource tracking
    private final AtomicInteger activeStreams = new AtomicInteger();
    private final AtomicLong allocatedMemory = new AtomicLong();
    private final Semaphore streamPermits;

    // Resource pools
    private final BlockingQueue<CudaStream> streamPool;
    private final Map<String, OccupancyProfile> occupancyProfiles;

    // Monitoring
    private final NvmlMonitor nvmlMonitor;  // Or AmdSmiMonitor
    private final ScheduledExecutorService monitorExecutor;

    // Configuration
    private final GpuSchedulerConfig config;
}
```

### Occupancy Profiling

Pre-compute occupancy for each kernel type:

```java
public class OccupancyProfile {
    private final String kernelName;
    private final int registersPerThread;
    private final int sharedMemoryPerBlock;
    private final int blockSize;

    // Computed
    private final float theoreticalOccupancy;
    private final int maxBlocksPerSM;

    public static OccupancyProfile analyze(CudaKernel kernel) {
        // Use cudaOccupancyMaxActiveBlocksPerMultiprocessor
        int maxBlocks = CudaRuntime.occupancyMaxActiveBlocksPerSM(
            kernel.getFunction(), kernel.getBlockSize(),
            kernel.getSharedMemory());

        float occupancy = (float)(maxBlocks * kernel.getBlockSize())
            / CudaRuntime.getMaxThreadsPerSM();

        return new OccupancyProfile(kernel.getName(), ..., occupancy, maxBlocks);
    }
}
```

### Capacity-Aware Acquisition

```java
public class GpuLease implements AutoCloseable {
    private final CudaStream stream;
    private final String kernelType;
    private final float estimatedOccupancy;
    private final long acquireTimeNanos;
    private final GpuScheduler scheduler;

    @Override
    public void close() {
        scheduler.release(this);
    }
}

public GpuLease acquire(String kernelType, long[] shape) throws InterruptedException {
    OccupancyProfile profile = occupancyProfiles.get(kernelType);
    float estimatedOccupancy = profile.getTheoreticalOccupancy();

    // Check if we should block based on current load
    while (shouldBlock(estimatedOccupancy)) {
        // Emit JFR event for visibility
        emitBlockEvent(kernelType, estimatedOccupancy);

        // Wait for capacity (with timeout)
        capacityAvailable.await(config.getBlockTimeoutMs(), TimeUnit.MILLISECONDS);
    }

    // Acquire stream from pool
    CudaStream stream = streamPool.poll(config.getStreamTimeoutMs(), TimeUnit.MILLISECONDS);
    if (stream == null) {
        throw new GpuResourceException("No stream available");
    }

    activeStreams.incrementAndGet();

    return new GpuLease(stream, kernelType, estimatedOccupancy, System.nanoTime(), this);
}

private boolean shouldBlock(float incomingOccupancy) {
    // Strategy 1: Simple stream count limit
    if (activeStreams.get() >= config.getMaxConcurrentStreams()) {
        return true;
    }

    // Strategy 2: Occupancy-based (estimate total occupancy)
    float totalEstimatedOccupancy = getCurrentEstimatedOccupancy() + incomingOccupancy;
    if (totalEstimatedOccupancy > config.getMaxTotalOccupancy()) {
        return true;
    }

    // Strategy 3: NVML-based (actual utilization feedback)
    if (nvmlMonitor.getUtilization() > config.getTargetUtilization()
        && activeStreams.get() > 0) {
        return true;
    }

    return false;
}
```

### Work Stealing for Multi-GPU

For multi-GPU systems, implement [work stealing](https://en.wikipedia.org/wiki/Work_stealing):

```java
public class MultiGpuScheduler {
    private final List<GpuScheduler> gpuSchedulers;
    private final ThreadLocal<Integer> preferredGpu = ThreadLocal.withInitial(() -> -1);

    public GpuLease acquire(String kernelType, long[] shape) throws InterruptedException {
        // Try preferred GPU first (locality)
        int preferred = preferredGpu.get();
        if (preferred >= 0) {
            GpuLease lease = gpuSchedulers.get(preferred).tryAcquire(kernelType, shape);
            if (lease != null) return lease;
        }

        // Work stealing: find least loaded GPU
        GpuScheduler leastLoaded = findLeastLoaded();
        GpuLease lease = leastLoaded.acquire(kernelType, shape);

        // Update affinity
        preferredGpu.set(leastLoaded.getDeviceIndex());

        return lease;
    }

    private GpuScheduler findLeastLoaded() {
        return gpuSchedulers.stream()
            .min(Comparator.comparingInt(s -> s.getActiveStreams()))
            .orElseThrow();
    }
}
```

### Integration with Virtual Threads

```java
public class GpuBatchProcessor {
    private final GpuScheduler scheduler;
    private final GpuBackend backend;

    public List<Tensor> processBatch(List<Model> models, List<Tensor> inputs)
            throws InterruptedException, ExecutionException {

        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            List<Supplier<Tensor>> tasks = new ArrayList<>();

            for (int i = 0; i < models.size(); i++) {
                final Model model = models.get(i);
                final Tensor input = inputs.get(i);

                tasks.add(scope.fork(() -> {
                    // Virtual thread blocks here if GPU is saturated
                    // Other virtual threads continue processing
                    try (GpuLease lease = scheduler.acquire("inference", input.shape())) {
                        return backend.execute(model, input, lease.getStream());
                    }
                }));
            }

            scope.join().throwIfFailed();

            return tasks.stream()
                .map(Supplier::get)
                .toList();
        }
    }
}
```

## Memory Management Integration

### Memory Pressure Tracking

```java
public class GpuMemoryManager {
    private final AtomicLong allocatedBytes = new AtomicLong();
    private final long totalMemory;
    private final long reservedMemory;  // For CUDA overhead, fragmentation

    public long getAvailableMemory() {
        return totalMemory - reservedMemory - allocatedBytes.get();
    }

    public boolean canAllocate(long bytes) {
        return getAvailableMemory() >= bytes;
    }

    public long allocate(long bytes) {
        if (!canAllocate(bytes)) {
            throw new GpuOutOfMemoryException("Cannot allocate " + bytes + " bytes");
        }
        allocatedBytes.addAndGet(bytes);
        return CudaRuntime.memAlloc(bytes);
    }

    public void free(long ptr, long bytes) {
        CudaRuntime.memFree(ptr);
        allocatedBytes.addAndGet(-bytes);
    }
}
```

### Memory-Aware Scheduling

```java
private boolean shouldBlock(String kernelType, long[] shape) {
    // Estimate memory requirement
    long estimatedMemory = estimateMemoryUsage(kernelType, shape);

    // Check memory availability
    if (!memoryManager.canAllocate(estimatedMemory)) {
        // Wait for memory to be freed
        return true;
    }

    // Continue with compute-based checks...
}
```

### Unified Virtual Memory Considerations

When using [UVM](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/), page faults add latency:

```java
public class UvmAwareScheduler extends GpuScheduler {

    @Override
    public GpuLease acquire(String kernelType, long[] shape) throws InterruptedException {
        // Prefetch data to GPU before acquiring compute resources
        prefetchToGpu(inputTensors);

        // Then acquire compute
        return super.acquire(kernelType, shape);
    }

    private void prefetchToGpu(List<Tensor> tensors) {
        for (Tensor t : tensors) {
            if (t.isUvmManaged()) {
                CudaRuntime.memPrefetchAsync(t.getDevicePtr(), t.byteSize(),
                    deviceIndex, prefetchStream);
            }
        }
        // Wait for prefetch to complete
        CudaRuntime.streamSynchronize(prefetchStream);
    }
}
```

## Inference Serving: SLO-Aware Scheduling

For inference workloads, meeting SLOs (Service Level Objectives) is critical:

```java
public class SloAwareScheduler extends GpuScheduler {

    public GpuLease acquire(InferenceRequest request) throws InterruptedException {
        long deadline = request.getDeadlineNanos();
        long now = System.nanoTime();
        long slack = deadline - now - estimateExecutionTime(request);

        // High priority if close to deadline
        if (slack < config.getUrgentThresholdNanos()) {
            return acquireUrgent(request);
        }

        // Normal priority
        return acquireNormal(request);
    }

    private GpuLease acquireUrgent(InferenceRequest request) throws InterruptedException {
        // Use priority stream
        CudaStream stream = priorityStreamPool.poll();
        if (stream != null) {
            // Set high priority
            CudaRuntime.streamSetPriority(stream, CUDA_STREAM_PRIORITY_HIGH);
            return new GpuLease(stream, ...);
        }

        // Preempt low-priority work if necessary
        return preemptAndAcquire(request);
    }
}
```

### Batching for Throughput

```java
public class AdaptiveBatcher {
    private final BlockingQueue<InferenceRequest> pendingRequests;
    private final ScheduledExecutorService batchTimer;

    public void submit(InferenceRequest request) {
        pendingRequests.add(request);

        // Check if we should batch now
        if (shouldBatchNow()) {
            executeBatch();
        }
    }

    private boolean shouldBatchNow() {
        int pending = pendingRequests.size();

        // Batch when we have enough requests
        if (pending >= config.getMaxBatchSize()) return true;

        // Or when oldest request is approaching deadline
        InferenceRequest oldest = pendingRequests.peek();
        if (oldest != null && oldest.getSlackNanos() < config.getBatchTimeoutNanos()) {
            return true;
        }

        return false;
    }

    private void executeBatch() {
        List<InferenceRequest> batch = new ArrayList<>();
        pendingRequests.drainTo(batch, config.getMaxBatchSize());

        // Execute batch on GPU
        try (GpuLease lease = scheduler.acquire("batch_inference", computeBatchShape(batch))) {
            backend.executeBatch(batch, lease.getStream());
        }
    }
}
```

## Advanced: Kernel Slicing for Preemption

For long-running kernels, implement slicing to enable preemption:

```java
public class SliceableKernel {
    private final long totalElements;
    private final int sliceSize;

    public void execute(CudaStream stream, long[] inputs, long output) {
        long remaining = totalElements;
        long offset = 0;

        while (remaining > 0) {
            long slice = Math.min(remaining, sliceSize);

            // Check for preemption between slices
            if (Thread.currentThread().isInterrupted()) {
                // Save state for resumption
                saveCheckpoint(offset);
                throw new PreemptedException(offset);
            }

            // Execute slice
            launchKernelSlice(stream, inputs, output, offset, slice);
            CudaRuntime.streamSynchronize(stream);

            offset += slice;
            remaining -= slice;
        }
    }
}
```

## Configuration

```java
public class GpuSchedulerConfig {
    // Concurrency limits
    private int maxConcurrentStreams = 16;
    private float maxTotalOccupancy = 0.9f;

    // NVML feedback
    private int targetUtilization = 80;  // percent
    private int utilizationPollingMs = 200;

    // Timeouts
    private long blockTimeoutMs = 5000;
    private long streamTimeoutMs = 1000;

    // Memory
    private float memoryReserveFraction = 0.1f;  // Reserve 10% for overhead

    // Batching (for inference)
    private int maxBatchSize = 32;
    private long batchTimeoutNanos = 10_000_000;  // 10ms

    // Preemption
    private long kernelSliceElements = 1_000_000;
    private long urgentThresholdNanos = 1_000_000;  // 1ms
}
```

## Implementation Phases

### Phase 1: Basic Stream Pool (Current Target)
- Fixed-size stream pool with semaphore
- Simple acquisition/release
- JFR events for visibility

### Phase 2: Occupancy-Aware Scheduling
- Pre-compute occupancy profiles
- Block based on estimated total occupancy
- Memory pressure tracking

### Phase 3: NVML/SMI Feedback Loop
- Real-time utilization monitoring
- Adaptive concurrency limits
- Multi-GPU work stealing

### Phase 4: SLO-Aware Scheduling
- Priority streams
- Adaptive batching
- Deadline-aware admission control

### Phase 5: Kernel Slicing and Preemption
- Sliceable kernel wrappers
- Checkpoint/resume for long operations
- True preemptive scheduling

## Research References

### GPU Scheduling and Partitioning
- [Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach](https://arxiv.org/html/2405.08754v1) (2024)
- [Hardware Compute Partitioning on NVIDIA GPUs](https://www.researchgate.net/publication/371906237_Hardware_Compute_Partitioning_on_NVIDIA_GPUs) (Bakita & Anderson, 2023)
- [Mitigating GPU Core Partitioning Performance Effects](https://engineering.purdue.edu/tgrogers/publication/barnes-hpca-2023/barnes-hpca-2023.pdf) (HPCA 2023)
- [Contention-Aware GPU Partitioning](https://hal.science/hal-03641750v1/document)

### Preemption and Real-Time
- [LithOS: An Operating System for Efficient Machine Learning on GPUs](https://www.cs.cmu.edu/~csd-phd-blog/2025/lithos/)
- [Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale](https://arxiv.org/html/2601.04071v1)
- [GCAPS: GPU Context-Aware Preemptive Priority-based Scheduling](https://arxiv.org/html/2406.05221v1)
- [Unleashing the Power of Preemptive Priority-based Scheduling for Real-Time GPU Tasks](https://arxiv.org/html/2401.16529v1)

### Inference Serving
- [ML Inference Scheduling with Predictable Latency](https://arxiv.org/html/2512.18725)
- [Deep Learning Workload Scheduling in GPU Datacenters: A Survey](https://dl.acm.org/doi/full/10.1145/3638757)
- [LLM Inference Scheduling: A Survey](https://www.techrxiv.org/users/994660/articles/1355915)

### Memory Management
- [Improving GPU Memory Oversubscription Performance](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)
- [Oversubscribing GPU Unified Virtual Memory](https://dl.acm.org/doi/10.1145/3489525.3511691)
- [MSched: GPU Multitasking via Proactive Memory Scheduling](https://arxiv.org/html/2512.24637v1)

### CUDA Graphs and Optimization
- [Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs](https://arxiv.org/html/2501.09398v1)
- [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)

### Hardware and APIs
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html)
- [CUDA Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)
- [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)
- [CUPTI Documentation](https://docs.nvidia.com/cupti/main/main.html)
- [AMD SMI Documentation](https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.1.0/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

### Work Stealing and Heterogeneous Computing
- [Dynamic Task Parallelism with a GPU Work-Stealing Runtime System](https://link.springer.com/chapter/10.1007/978-3-642-36036-7_14)
- [Exploiting Concurrent GPU Operations for Efficient Work Stealing on Multi-GPUs](https://inria.hal.science/hal-00735470/document)
