# JFR-Integrated GPU Profiling

This document describes WarpForge's approach to GPU performance observability using Java Flight Recorder (JFR) custom events, enabling unified profiling across both vendor libraries (cuBLAS, MIOpen) and generated kernels (PTX, SPIR-V).

## The Observability Challenge

WarpForge's hybrid backend architecture uses:
- **Vendor libraries** (cuBLAS, rocBLAS, cuDNN, MIOpen) for heavy operations (GEMM, convolution)
- **Generated kernels** (PTX, SPIR-V) for elementwise and custom operations

The existing salt-based instrumentation (see `BACKEND-KERNEL-INSTRUMENTATION.md`) provides in-kernel visibility for generated kernels. However, vendor libraries are black boxes—we cannot instrument their internal execution. We need a different approach for unified observability.

## Solution: GPU Events + JFR Custom Events

```
+---------------------------------------------------------------------+
|                    WarpForge GPU Operation                          |
|  +---------------------------------------------------------------+  |
|  |  GpuKernelEvent (JFR Custom Event)                            |  |
|  |    - operation: "GEMM"                                        |  |
|  |    - shape: "4096x4096 * 4096x4096"                           |  |
|  |    - gpuTimeMicros: 1234                                      |  |
|  |    - teraflops: 27.3                                          |  |
|  |    - backend: "cuBLAS"                                        |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  +-------------+    +-------------+    +-------------+              |
|  | CUDA Event  |    |   cuBLAS    |    | CUDA Event  |              |
|  |   start     |--->|   SGEMM     |--->|    stop     |              |
|  +-------------+    +-------------+    +-------------+              |
|         |                                      |                    |
|         +-------- cudaEventElapsedTime() ------+                    |
|                           |                                         |
|                           v                                         |
|                   GpuKernelEvent.commit()                           |
+---------------------------------------------------------------------+
```

## Profiling API Landscape

### NVIDIA

|------------------------|------------------------|-------------------|-----------------|------------|
| Approach               | Granularity            | Overhead          | JFR Integration | Complexity |
|------------------------|------------------------|-------------------|-----------------|------------|
| **CUDA Events**        | Per-kernel             | ~0.5us resolution | Easy            | Low        |
| **CUPTI Activity API** | Per-kernel + transfers | Low               | Medium          | High       |
| **NVTX Ranges**        | Code regions           | Very low          | Via Nsight      | Low        |
|------------------------|------------------------|-------------------|-----------------|------------|

### AMD

|------------------------|------------------------|-------------------|-----------------|------------|
| Approach               | Granularity            | Overhead          | JFR Integration | Complexity |
|------------------------|------------------------|-------------------|-----------------|------------|
| **HIP Events**         | Per-kernel             | Similar to CUDA   | Easy            | Low        |
| **roctracer**          | Per-kernel + transfers | Low               | Medium          | High       |
| **roctx**              | Code regions           | Very low          | Via rocprof     | Low        |
|------------------------|------------------------|-------------------|-----------------|------------|

## Recommended Approach: GPU Events

CUDA Events and HIP Events provide the best balance of accuracy, low overhead, and ease of integration. They work with both vendor libraries and generated kernels.

### CUDA Events API

```c
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);

// Execute kernel or cuBLAS call
cublasSgemm(...);

cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);  // ~0.5μs resolution
```

### HIP Events API (AMD equivalent)

```c
hipEventCreate(&start);
hipEventCreate(&stop);
hipEventRecord(start, stream);

// Execute kernel or rocBLAS call
rocblas_sgemm(...);

hipEventRecord(stop, stream);
hipEventSynchronize(stop);
hipEventElapsedTime(&milliseconds, start, stop);
```

## JFR Custom Events

### Event Definitions

```java
@Label("GPU Kernel Execution")
@Category({"WarpForge", "GPU"})
@Description("Records GPU kernel execution time and throughput")
public class GpuKernelEvent extends jdk.jfr.Event {
    @Label("Operation")
    public String operation;

    @Label("Shape")
    public String shape;

    @Label("GPU Time")
    @Timespan(Timespan.MICROSECONDS)
    public long gpuTimeMicros;

    @Label("Throughput")
    public double teraflops;

    @Label("Backend")
    public String backend;  // "cuBLAS", "PTX", "MIOpen", "SPIR-V"

    @Label("Device")
    public int deviceIndex;
}

@Label("GPU Memory Transfer")
@Category({"WarpForge", "GPU", "Memory"})
public class GpuMemoryEvent extends jdk.jfr.Event {
    @Label("Direction")
    public String direction;  // "H2D", "D2H", "D2D"

    @Label("Bytes")
    @DataAmount
    public long bytes;

    @Label("Bandwidth GB/s")
    public double bandwidthGBps;
}

@Label("GPU Kernel Compilation")
@Category({"WarpForge", "GPU", "Compilation"})
public class GpuCompilationEvent extends jdk.jfr.Event {
    @Label("Kernel")
    public String kernelName;

    @Label("Backend")
    public String backend;  // "PTX", "SPIR-V", "HIP"

    @Label("Compile Time")
    @Timespan(Timespan.MICROSECONDS)
    public long compileTimeMicros;

    @Label("Cached")
    public boolean wasCached;
}
```

### Instrumented Kernel Dispatch

```java
public class CublasGemmKernel implements CudaOpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        long eventStart = cudaRuntime.eventCreate();
        long eventStop = cudaRuntime.eventCreate();

        try {
            // Record start
            cudaRuntime.eventRecord(eventStart, stream);

            // Execute cuBLAS GEMM
            cublasRuntime.sgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                alpha, dA, lda, dB, ldb,
                beta, dC, ldc);

            // Record stop
            cudaRuntime.eventRecord(eventStop, stream);
            cudaRuntime.eventSynchronize(eventStop);

            // Calculate elapsed time
            float elapsedMs = cudaRuntime.eventElapsedTime(eventStart, eventStop);

            // Emit JFR event
            GpuKernelEvent event = new GpuKernelEvent();
            event.operation = "GEMM";
            event.shape = M + "x" + K + " * " + K + "x" + N;
            event.gpuTimeMicros = (long)(elapsedMs * 1000);
            event.teraflops = (2.0 * M * N * K) / (elapsedMs * 1e9);
            event.backend = "cuBLAS";
            event.deviceIndex = deviceIndex;
            event.commit();

            // ... return result
        } finally {
            cudaRuntime.eventDestroy(eventStart);
            cudaRuntime.eventDestroy(eventStop);
        }
    }
}
```

## JDK Mission Control Visualization

With JFR events, GPU operations appear in standard Java profiling tools:

```
+-----------------------------------------------------------------+
|  JDK Mission Control - GPU Kernel Events                        |
+-----------------------------------------------------------------+
|  Time        Operation  Shape           GPU(us)  TFLOPS Backend |
|  ----------  ---------  --------------  -------  ------ ------- |
|  10:23:01.1  GEMM       4096x4096       1,234    27.3   cuBLAS  |
|  10:23:01.2  Add        4096x4096       12       -      PTX     |
|  10:23:01.2  GEMM       4096x4096       1,198    28.1   cuBLAS  |
|  10:23:01.3  Softmax    4096x128        45       -      cuDNN   |
|  10:23:01.3  GEMM       4096x4096       1,201    28.0   cuBLAS  |
|  ...                                                            |
+-----------------------------------------------------------------+
|  [Histogram: GPU Time by Operation]  [Timeline View]            |
+-----------------------------------------------------------------+
```

## Integration with Salt-Based Instrumentation

The JFR approach complements, not replaces, the existing salt-based instrumentation:

|------------------------------------------|------------------------------------|
| Use Case                                 | Approach                           |
|------------------------------------------|------------------------------------|
| **Vendor library ops** (cuBLAS, MIOpen)  | CUDA/HIP Events -> JFR             |
| **Generated kernel wall-clock time**     | CUDA/HIP Events -> JFR             |
| **Generated kernel internal timing**     | Salt instrumentation (SALT_TIMING) |
| **Memory transfer timing**               | CUDA/HIP Events -> JFR             |
| **Kernel compilation timing**            | Host-side timing -> JFR            |
|------------------------------------------|------------------------------------|

### Unified Event Flow

```
+-----------------------------------------------------------------+
|  Heavy Ops (cuBLAS/MIOpen)    |  Light Ops (Generated Kernels)  |
+-------------------------------+---------------------------------+
|  - CUDA/HIP Events for timing |  - CUDA/HIP Events for timing   |
|  - JFR events with TFLOPS     |  - Salt instrumentation for     |
|  - NVTX/roctx for Nsight      |    internal visibility          |
|                               |  - JFR events for consistency   |
+-------------------------------+---------------------------------+
```

## CUPTI/roctracer (Advanced)

For deeper visibility, CUPTI (NVIDIA) and roctracer (AMD) provide:
- Automatic capture of all kernel launches and memory transfers
- Hardware performance counters
- Correlation with NVTX/roctx ranges

However, they require more complex integration:
- Callback registration during initialization
- Buffer management for activity records
- Thread-safe event correlation

These are deferred to a future phase. GPU Events provide sufficient visibility for initial implementation.

## FFM Bindings Required

### NVIDIA (CudaRuntime additions)

```java
// Event management
cudaEventCreate      // cudaError_t cudaEventCreate(cudaEvent_t *event)
cudaEventDestroy     // cudaError_t cudaEventDestroy(cudaEvent_t event)
cudaEventRecord      // cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
cudaEventSynchronize // cudaError_t cudaEventSynchronize(cudaEvent_t event)
cudaEventElapsedTime // cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop)
```

### AMD (HipRuntime)

```java
// Event management
hipEventCreate       // hipError_t hipEventCreate(hipEvent_t *event)
hipEventDestroy      // hipError_t hipEventDestroy(hipEvent_t event)
hipEventRecord       // hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
hipEventSynchronize  // hipError_t hipEventSynchronize(hipEvent_t event)
hipEventElapsedTime  // hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
```

## Optional: NVTX/roctx Integration

For correlation with vendor profilers (Nsight Systems, rocprof):

```java
// NVTX (NVIDIA)
nvtxRangePushA("GEMM 4096x4096");
cublasSgemm(...);
nvtxRangePop();

// roctx (AMD)
roctxRangePush("GEMM 4096x4096");
rocblas_sgemm(...);
roctxRangePop();
```

This allows GPU operations to appear in vendor profiler timelines, correlated with their internal kernel traces.

## Module Structure

```
warpforge-core/
├── src/main/java/.../core/jfr/
│   ├── GpuKernelEvent.java              # Kernel execution timing
│   ├── GpuMemoryEvent.java              # Memory transfer timing
│   └── GpuCompilationEvent.java         # Kernel compilation timing

warpforge-benchmark/
├── src/main/java/.../benchmark/jfr/
│   ├── GpuBenchmarkEvent.java           # Benchmark iteration results
│   └── GpuTierComparisonEvent.java      # Tier comparison results

warpforge-backend-nvidia/
├── src/main/java/.../cuda/
│   ├── CudaRuntime.java                 # Add event functions
│   └── CudaContext.java                 # Add event helper methods

warpforge-backend-amd/
├── src/main/java/.../hip/
│   ├── HipRuntime.java                  # Include event functions
│   └── HipContext.java                  # Include event helper methods
```

## Comparison: Salt vs GPU Events vs CUPTI

|----------------------------|----------|------------|-------------------|
| Feature                    | PTX Salt | GPU Events | CUPTI/roctracer   |
|----------------------------|----------|------------|-------------------|
| Works with cuBLAS/MIOpen   | No       | Yes        | Yes               |
| Per-thread timing          | Yes      | No         | Yes (with effort) |
| Memory transfer tracking   | No       | Manual     | Yes Automatic     |
| Low overhead               | Yes      | Yes        | Medium            |
| JFR integration            | Custom   | Easy       | Medium            |
| Hardware counters          | No       | No         | Yes               |
| Implementation effort      | Done     | ~200 LOC   | ~1000 LOC         |
|----------------------------|----------|------------|-------------------|

## Nightly/Weekly CI Performance Tracking

### Why GPU-Side JFR Matters for Babylon

JFR on the JVM side tracks Java execution, but **Babylon optimizations manifest on the GPU**. When Babylon transforms code (kernel fusion, memory layout optimization, operation reordering), the impact is measured in GPU kernel timing, not JVM metrics.

Without GPU-side JFR:
- Babylon optimization merged → JVM metrics unchanged → "Did it help?"
- Performance regression → No data on which kernel got slower

With GPU-side JFR:
- Every nightly run captures per-kernel timing with full context
- Babylon optimization merged → Compare kernel durations → Quantified impact
- Regression detected → JFR shows exactly which kernel, which shape, which backend

### CI Integration Architecture

```
+------------------------------------------------------------------+
|                    Nightly Performance CI                         |
+------------------------------------------------------------------+
|                                                                  |
|  1. Build WarpForge with JFR instrumentation enabled             |
|  2. Run ptest benchmarks on both GPU nodes (NVIDIA + AMD)        |
|  3. Collect JFR recordings with GPU events                       |
|  4. Upload artifacts tagged with commit SHA                      |
|  5. Compare against baseline, flag regressions                   |
|                                                                  |
+------------------------------------------------------------------+
         |                                    |
         v                                    v
+-------------------+              +-------------------+
| NVIDIA Node       |              | AMD Node          |
| - GpuKernelEvent  |              | - GpuKernelEvent  |
| - CUPTI traces    |              | - roctracer data  |
| - TFLOPS metrics  |              | - TFLOPS metrics  |
+-------------------+              +-------------------+
         |                                    |
         +----------------+-------------------+
                          |
                          v
              +------------------------+
              | Artifact Storage       |
              | performance-artifacts/ |
              |   {commit-sha}/        |
              |     jfr/               |
              |     baselines/         |
              +------------------------+
```

### ptest Directory Structure

The `ptest` directory contains performance tests with JFR integration:

```
ptest/
├── benchmarks/
│   ├── kernel/                    # Individual kernel performance
│   │   ├── GemmBenchmark.java     # Matrix multiply variants
│   │   ├── ConvBenchmark.java     # Convolution kernels
│   │   ├── SoftmaxBenchmark.java  # Softmax implementations
│   │   └── LayerNormBenchmark.java
│   ├── fusion/                    # Fused operation patterns
│   │   ├── GeluFusionBenchmark.java
│   │   ├── AttentionFusionBenchmark.java
│   │   └── ResidualNormBenchmark.java
│   └── endtoend/                  # Full model inference
│       ├── BertInferenceBenchmark.java
│       ├── LlamaInferenceBenchmark.java
│       └── VisionTransformerBenchmark.java
├── baselines/
│   ├── nvidia/
│   │   ├── gemm-baseline.json     # Per-kernel baseline metrics
│   │   ├── conv-baseline.json
│   │   └── endtoend-baseline.json
│   └── amd/
│       ├── gemm-baseline.json
│       └── ...
├── analysis/
│   ├── JfrDiffTool.java           # Compare two JFR recordings
│   ├── RegressionDetector.java    # Flag performance regressions
│   └── ReportGenerator.java       # Generate comparison reports
├── jfr-configs/
│   ├── ptest-profile.jfc          # JFR settings for benchmarks
│   └── ptest-gpu-detailed.jfc     # High-detail GPU profiling
└── reports/
    └── .gitignore                 # Generated reports excluded
```

### Benchmark Execution with JFR

```java
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {
    "-XX:StartFlightRecording=filename=gemm.jfr,settings=ptest-profile.jfc"
})
public class GemmBenchmark {

    @Param({"1024", "2048", "4096"})
    private int size;

    private CudaBackend backend;
    private Tensor a, b;

    @Setup
    public void setup() {
        backend = new CudaBackend(0);
        a = Tensor.randn(size, size, ScalarType.F32, backend);
        b = Tensor.randn(size, size, ScalarType.F32, backend);
    }

    @Benchmark
    public Tensor gemm() {
        // GpuKernelEvent automatically emitted
        return backend.matmul(a, b);
    }
}
```

### Baseline Comparison

```java
public class RegressionDetector {

    private static final double REGRESSION_THRESHOLD = 0.05; // 5%

    public RegressionReport compare(Path currentJfr, Path baselineJfr) {
        var current = loadGpuKernelEvents(currentJfr);
        var baseline = loadGpuKernelEvents(baselineJfr);

        var regressions = new ArrayList<Regression>();

        for (var entry : current.entrySet()) {
            String kernel = entry.getKey();
            double currentTime = entry.getValue().avgGpuTimeMicros();
            double baselineTime = baseline.get(kernel).avgGpuTimeMicros();

            double delta = (currentTime - baselineTime) / baselineTime;

            if (delta > REGRESSION_THRESHOLD) {
                regressions.add(new Regression(
                    kernel,
                    baselineTime,
                    currentTime,
                    delta
                ));
            }
        }

        return new RegressionReport(regressions);
    }
}
```

### Tracking Babylon Optimization Impact

When a Babylon optimization is merged:

1. **Before merge**: Baseline JFR shows kernel X takes 1.5ms
2. **PR merged**: Babylon fusion combines kernel X + Y
3. **Nightly run**: JFR shows fused kernel XY takes 1.1ms
4. **Report**: "Babylon fusion reduced attention compute by 27%"

This creates an auditable history:
```
performance-artifacts/
├── abc123/          # Before optimization
│   └── jfr/nvidia/attention.jfr  -> attention_scores: 1.5ms
├── def456/          # After optimization
│   └── jfr/nvidia/attention.jfr  -> attention_fused: 1.1ms
└── comparison-abc123-def456.html  # Generated diff report
```

## Implementation Phases

### Phase 1: GPU Events + JFR ✅ (Complete)
- ✅ Add CUDA Event FFM bindings (CudaRuntime.java)
- ✅ Add HIP Event FFM bindings (HipRuntime.java - already complete)
- ✅ Define JFR event classes (GpuKernelEvent, GpuMemoryEvent, GpuCompilationEvent)
- ✅ Add CudaContext event helper methods (createEvent, destroyEvent, timeOperation)
- ⏳ Wrap kernel dispatchers with timing (ready for integration)

### Phase 2: NVTX/roctx Integration
- Add NVTX FFM bindings
- Instrument key code paths
- Enable Nsight/rocprof correlation

### Phase 3: CUPTI/roctracer (Future)
- Full activity tracing
- Hardware counter collection
- Advanced profiling scenarios

## Usage

### Enabling JFR Recording

```bash
java -XX:StartFlightRecording=filename=gpu.jfr,settings=profile \
     -jar warpforge-cli.jar run model.mlir
```

### Analyzing with JDK Mission Control

1. Open `gpu.jfr` in JDK Mission Control
2. Navigate to Event Browser → WarpForge → GPU
3. View kernel execution times, throughput, and memory transfers
4. Correlate with Java thread activity and GC events

### Programmatic Access

```java
try (RecordingStream rs = new RecordingStream()) {
    rs.enable(GpuKernelEvent.class);
    rs.onEvent(GpuKernelEvent.class, event -> {
        System.out.printf("%s: %.2f TFLOPS (%s)%n",
            event.getString("operation"),
            event.getDouble("teraflops"),
            event.getString("backend"));
    });
    rs.start();
}
```
