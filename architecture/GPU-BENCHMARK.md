# GPU Kernel Benchmark Framework

This document describes WarpForge's GPU kernel benchmarking framework, designed to validate the performance claims of the three-tier kernel architecture with JFR integration.

## Motivation

The three-tier kernel architecture (see [BACKEND-KERNEL-INSTRUMENTATION.md](BACKEND-KERNEL-INSTRUMENTATION.md)) makes specific performance claims:

| Tier | Claimed Performance |
|------|---------------------|
| PRODUCTION | 100% (baseline) |
| OPTIMIZED_OBSERVABLE | ~93% of PRODUCTION |
| CORRECTNESS | ~1% of PRODUCTION |

**Trust but verify.** The benchmark framework proves these claims through systematic measurement, ensuring the OPTIMIZED_OBSERVABLE tier can truly be trusted as "near-production" speed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WarpForge GPU Benchmark Framework                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  @GpuBenchmark Annotations                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  @GpuBenchmark(                                                 │  │  │
│  │  │      operation = "GEMM",                                        │  │  │
│  │  │      shape = "4096x4096",                                       │  │  │
│  │  │      tiers = {PRODUCTION, OPTIMIZED_OBSERVABLE, CORRECTNESS}    │  │  │
│  │  │  )                                                              │  │  │
│  │  │  void benchmarkGemm(KernelTier tier) { ... }                    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  GpuBenchmarkRunner                                                   │  │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐  │  │
│  │  │  Warmup Phase     │  │  Measurement      │  │  Statistical      │  │  │
│  │  │  (JIT, cache,     │──│  Phase            │──│  Analysis         │  │  │
│  │  │   GPU init)       │  │  (N iterations)   │  │  (mean, stddev,   │  │  │
│  │  │                   │  │                   │  │   percentiles)    │  │  │
│  │  └───────────────────┘  └───────────────────┘  └───────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                     ┌──────────────┼──────────────┐                         │
│                     ▼              ▼              ▼                         │
│  ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐   │
│  │  JFR Events         │ │ Console Output  │ │ TierComparisonReport    │   │
│  │  ┌───────────────┐  │ │ (Real-time)     │ │ ┌─────────────────────┐ │   │
│  │  │ GpuKernelEvent│  │ │                 │ │ │ Overhead Analysis   │ │   │
│  │  │ GpuBenchmark- │  │ │                 │ │ │ Pass/Fail Status    │ │   │
│  │  │   Event       │  │ │                 │ │ │ Recommendations     │ │   │
│  │  │ GpuTierComp-  │  │ │                 │ │ │ JSON Export         │ │   │
│  │  │   arisonEvent │  │ │                 │ │ └─────────────────────┘ │   │
│  │  └───────────────┘  │ │                 │ │                         │   │
│  └─────────────────────┘ └─────────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## JMH-Style Methodology

The framework follows JMH (Java Microbenchmark Harness) best practices adapted for GPU measurement:

### Warmup Phase

GPU benchmarks require warmup for:
1. **Kernel compilation** - PTX → CUBIN JIT compilation
2. **Memory allocation** - First allocation may be slower
3. **Kernel caching** - Module/function handles cached
4. **GPU frequency scaling** - GPU needs to "spin up"

Default: 5 warmup iterations

### Measurement Phase

After warmup, measurement iterations collect timing data:
- Multiple iterations for statistical significance
- CUDA/HIP Events for accurate GPU timing
- Per-kernel and aggregate statistics

Default: 20 measurement iterations

### Statistical Analysis

Results include:
- **Mean** - Average execution time
- **Standard Deviation** - Measurement stability
- **Percentiles** - P50, P99 for latency distribution
- **CV (Coefficient of Variation)** - Stability indicator (<5% is good)

## JFR Profiler Integration

Following JMH conventions, JFR is used as a **profiler** (to capture what happens during benchmark execution), not as a result reporter. Benchmark results are output as text and JSON.

### JFR Profiler Events (from warpforge-core)

When running with `--prof jfr`, the benchmark emits `GpuKernelEvent` events that can be analyzed in JDK Mission Control:

```java
@Name("io.surfworks.warpforge.GpuKernel")
@Label("GPU Kernel Execution")
@Category({"WarpForge", "GPU"})
public class GpuKernelEvent extends Event {
    public String operation;
    @Timespan(Timespan.MICROSECONDS)
    public long gpuTimeMicros;
    public double teraflops;
    public String tier;
    public String backend;
}
```

### Result Reporting (JMH-style)

Results are reported via:
- **Text output** - Console/stdout with formatted tables
- **JSON export** - `TierComparisonReport.toJson()` for programmatic consumption

This follows JMH's pattern where `-rf json -rff results.json` outputs machine-readable results, separate from JFR profiler recordings.

## Usage

### Writing Benchmarks

```java
public class GemmBenchmarks {

    private CudaContext context;
    private Tensor matA, matB;

    @Setup(level = Setup.Level.TRIAL)
    public void setup() {
        context = CudaContext.create(0);
        matA = Tensor.random(DType.F32, new int[]{4096, 4096});
        matB = Tensor.random(DType.F32, new int[]{4096, 4096});
    }

    @TearDown(level = Setup.Level.TRIAL)
    public void teardown() {
        context.close();
    }

    @GpuBenchmark(
        operation = "GEMM",
        shape = "4096x4096 * 4096x4096",
        tiers = {KernelTier.PRODUCTION, KernelTier.OPTIMIZED_OBSERVABLE},
        warmupIterations = 5,
        measurementIterations = 20
    )
    public Tensor benchmarkGemm(KernelTier tier) {
        DotKernel kernel = new DotKernel(context, tier.saltLevel());
        return kernel.execute(gemmOp, List.of(matA, matB)).get(0);
    }
}
```

### Running Benchmarks

```bash
# Run benchmarks (text output to console)
java -jar warpforge-benchmark.jar --all-tiers io.surfworks.warpforge.benchmark.GemmBenchmarks

# Run with JFR profiler (like JMH -prof jfr)
java -jar warpforge-benchmark.jar --prof jfr --all-tiers io.surfworks.warpforge.benchmark.GemmBenchmarks

# Run with JFR recording via Gradle
./gradlew :warpforge-benchmark:runWithJfr

# Run with custom tolerance
java -jar warpforge-benchmark.jar --tolerance 5.0 io.surfworks.warpforge.benchmark.GemmBenchmarks
```

### Programmatic Usage

```java
GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
runner.include(".*Gemm.*")    // Only GEMM benchmarks
      .tolerance(3.0)          // 3% tolerance
      .jfrEvents(true);        // Enable JFR

runner.run(GemmBenchmarks.class);

TierComparisonReport report = runner.generateReport();
report.print();

if (!report.allPassed()) {
    System.exit(1);  // Fail CI if overhead exceeds expectations
}
```

## TierComparisonReport

The report validates overhead claims:

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    GPU KERNEL TIER COMPARISON REPORT                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Total Benchmarks: 4                                                       ║
║  Total Comparisons: 2                                                      ║
║  Tolerance: ±3.0%                                                          ║
║  Status: ✓ ALL PASSED                                                      ║
╚════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│              OPTIMIZED_OBSERVABLE TIER OVERHEAD ANALYSIS                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  Expected overhead: ~7% (achieving ~93% of PRODUCTION performance)          │
│  ────────────────────────────────────────────────────────────────────────    │
│  Observed Mean:    +6.85%                                                   │
│  Observed Min:     +5.92%                                                   │
│  Observed Max:     +7.78%                                                   │
│  Sample Count:         2                                                     │
│  ────────────────────────────────────────────────────────────────────────    │
│  Assessment: ✓ EXCELLENT: Overhead matches theoretical prediction           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Integration with CI

Add benchmark validation to CI pipeline:

```yaml
# .github/workflows/gpu-benchmark.yml
jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - name: Run GPU Benchmarks
        run: |
          ./gradlew :warpforge-benchmark:gpuBenchmarkTest

      - name: Validate Tier Overhead
        run: |
          java -jar build/libs/warpforge-benchmark.jar \
            --all-tiers \
            --tolerance 5.0 \
            io.surfworks.warpforge.benchmark.CoreOpBenchmarks
```

## Analyzing with JDK Mission Control

When running with `--jfr`:

1. Open `gpu-benchmark.jfr` in JDK Mission Control
2. Navigate to **Event Browser** → **WarpForge** → **GPU**
3. View kernel execution times, tier comparisons, throughput
4. Correlate with Java thread activity and GC events

## Implementation Files

### warpforge-benchmark

| File | Purpose |
|------|---------|
| `KernelTier.java` | Enum defining three execution tiers |
| `GpuBenchmarkRunner.java` | Main benchmark execution engine |
| `BenchmarkResult.java` | Statistical analysis of timing data |
| `TierComparisonReport.java` | Report generation (text + JSON), validation |
| `annotation/GpuBenchmark.java` | Benchmark method annotation |
| `annotation/Setup.java` | Setup method annotation |
| `annotation/TearDown.java` | Teardown method annotation |

### warpforge-core (JFR Profiler Events)

| File | Purpose |
|------|---------|
| `jfr/GpuKernelEvent.java` | JFR profiler event for kernel executions |
| `jfr/GpuMemoryEvent.java` | JFR profiler event for memory transfers |
| `jfr/GpuCompilationEvent.java` | JFR profiler event for kernel compilation |

## Relationship to Other Documents

- **[BACKEND-KERNEL-INSTRUMENTATION.md](BACKEND-KERNEL-INSTRUMENTATION.md)** - Defines the three-tier architecture being validated
- **[JFR-GPU.md](JFR-GPU.md)** - JFR event definitions and GPU profiling approach
- **[backend-phases.md](backend-phases.md)** - Overall backend development phases

## Future Enhancements

### Phase 1 (Current)
- Basic benchmark runner with JMH-style methodology
- JFR event emission
- Tier comparison validation

### Phase 2 (Planned)
- Integration with actual CUDA/HIP Events for GPU-accurate timing
- Real backend kernel execution (not simulated)
- Memory bandwidth measurement

### Phase 3 (Future)
- Automatic regression detection
- Historical trend analysis
- Performance baseline management
