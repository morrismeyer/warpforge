package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.golden.ComparisonResult;

import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Results from a benchmark run.
 *
 * <p>Contains timing statistics, accuracy metrics, and optional memory usage data.
 */
public record BenchmarkResult(
        String benchmarkName,
        String modelId,
        String backend,
        Instant startTime,
        Instant endTime,
        int warmupIterations,
        int measurementIterations,
        long[] latenciesNanos,
        List<ComparisonResult> validationResults,
        MemoryStats memoryStats,
        Status status,
        String errorMessage
) {

    public BenchmarkResult {
        Objects.requireNonNull(benchmarkName);
        Objects.requireNonNull(modelId);
        Objects.requireNonNull(backend);
        Objects.requireNonNull(startTime);
        Objects.requireNonNull(endTime);
        Objects.requireNonNull(latenciesNanos);
        Objects.requireNonNull(status);
        latenciesNanos = latenciesNanos.clone();
        validationResults = validationResults != null ? List.copyOf(validationResults) : List.of();
    }

    /**
     * Benchmark status.
     */
    public enum Status {
        SUCCESS,
        VALIDATION_FAILED,
        ERROR,
        SKIPPED
    }

    /**
     * Whether the benchmark completed successfully with valid outputs.
     */
    public boolean isSuccess() {
        return status == Status.SUCCESS;
    }

    /**
     * Whether all outputs matched the golden outputs.
     */
    public boolean allOutputsValid() {
        return validationResults.stream().allMatch(ComparisonResult::matches);
    }

    /**
     * Total benchmark duration including warmup.
     */
    public Duration totalDuration() {
        return Duration.between(startTime, endTime);
    }

    /**
     * Mean latency in nanoseconds.
     */
    public double meanLatencyNanos() {
        if (latenciesNanos.length == 0) return 0;
        return Arrays.stream(latenciesNanos).average().orElse(0);
    }

    /**
     * Mean latency in milliseconds.
     */
    public double meanLatencyMs() {
        return meanLatencyNanos() / 1_000_000.0;
    }

    /**
     * Minimum latency in nanoseconds.
     */
    public long minLatencyNanos() {
        return Arrays.stream(latenciesNanos).min().orElse(0);
    }

    /**
     * Maximum latency in nanoseconds.
     */
    public long maxLatencyNanos() {
        return Arrays.stream(latenciesNanos).max().orElse(0);
    }

    /**
     * Standard deviation of latency in nanoseconds.
     */
    public double stdLatencyNanos() {
        if (latenciesNanos.length < 2) return 0;
        double mean = meanLatencyNanos();
        double sumSq = 0;
        for (long lat : latenciesNanos) {
            sumSq += (lat - mean) * (lat - mean);
        }
        return Math.sqrt(sumSq / (latenciesNanos.length - 1));
    }

    /**
     * Percentile latency in nanoseconds.
     *
     * @param percentile Percentile (0-100), e.g., 50 for median, 95 for p95
     */
    public long percentileLatencyNanos(int percentile) {
        if (latenciesNanos.length == 0) return 0;
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("Percentile must be 0-100");
        }
        long[] sorted = latenciesNanos.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    /**
     * Median latency (p50) in nanoseconds.
     */
    public long medianLatencyNanos() {
        return percentileLatencyNanos(50);
    }

    /**
     * P95 latency in nanoseconds.
     */
    public long p95LatencyNanos() {
        return percentileLatencyNanos(95);
    }

    /**
     * P99 latency in nanoseconds.
     */
    public long p99LatencyNanos() {
        return percentileLatencyNanos(99);
    }

    /**
     * Throughput in iterations per second.
     */
    public double throughputPerSecond() {
        double meanNanos = meanLatencyNanos();
        if (meanNanos <= 0) return 0;
        return 1_000_000_000.0 / meanNanos;
    }

    /**
     * Generate a human-readable summary.
     */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Benchmark: %s%n", benchmarkName));
        sb.append(String.format("Model: %s | Backend: %s%n", modelId, backend));
        sb.append(String.format("Status: %s%n", status));

        if (status == Status.ERROR) {
            sb.append(String.format("Error: %s%n", errorMessage));
            return sb.toString();
        }

        sb.append(String.format("Iterations: %d warmup, %d measurement%n",
                warmupIterations, measurementIterations));
        sb.append(String.format("Latency: mean=%.2fms, median=%.2fms, p95=%.2fms, p99=%.2fms%n",
                meanLatencyMs(),
                medianLatencyNanos() / 1_000_000.0,
                p95LatencyNanos() / 1_000_000.0,
                p99LatencyNanos() / 1_000_000.0));
        sb.append(String.format("Range: min=%.2fms, max=%.2fms, std=%.2fms%n",
                minLatencyNanos() / 1_000_000.0,
                maxLatencyNanos() / 1_000_000.0,
                stdLatencyNanos() / 1_000_000.0));
        sb.append(String.format("Throughput: %.2f iter/sec%n", throughputPerSecond()));

        if (!validationResults.isEmpty()) {
            long passed = validationResults.stream().filter(ComparisonResult::matches).count();
            sb.append(String.format("Validation: %d/%d outputs matched golden%n",
                    passed, validationResults.size()));
        }

        if (memoryStats != null) {
            sb.append(String.format("Memory: peak=%s, allocated=%s%n",
                    formatBytes(memoryStats.peakUsageBytes()),
                    formatBytes(memoryStats.allocatedBytes())));
        }

        return sb.toString();
    }

    /**
     * Generate a detailed report including validation details.
     */
    public String detailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(summary());

        if (!validationResults.isEmpty()) {
            sb.append("\nValidation Details:\n");
            for (int i = 0; i < validationResults.size(); i++) {
                ComparisonResult vr = validationResults.get(i);
                sb.append(String.format("  Output %d: %s%n", i, vr.summary()));
            }
        }

        return sb.toString();
    }

    /**
     * Convert to a map for JSON serialization.
     */
    public Map<String, Object> toMap() {
        return Map.ofEntries(
                Map.entry("benchmark_name", benchmarkName),
                Map.entry("model_id", modelId),
                Map.entry("backend", backend),
                Map.entry("start_time", startTime.toString()),
                Map.entry("end_time", endTime.toString()),
                Map.entry("warmup_iterations", warmupIterations),
                Map.entry("measurement_iterations", measurementIterations),
                Map.entry("status", status.name()),
                Map.entry("mean_latency_ms", meanLatencyMs()),
                Map.entry("median_latency_ms", medianLatencyNanos() / 1_000_000.0),
                Map.entry("p95_latency_ms", p95LatencyNanos() / 1_000_000.0),
                Map.entry("p99_latency_ms", p99LatencyNanos() / 1_000_000.0),
                Map.entry("min_latency_ms", minLatencyNanos() / 1_000_000.0),
                Map.entry("max_latency_ms", maxLatencyNanos() / 1_000_000.0),
                Map.entry("throughput_per_sec", throughputPerSecond()),
                Map.entry("all_outputs_valid", allOutputsValid())
        );
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    /**
     * Memory usage statistics.
     */
    public record MemoryStats(
            long peakUsageBytes,
            long allocatedBytes,
            long freedBytes,
            int allocationCount
    ) {
        public static MemoryStats empty() {
            return new MemoryStats(0, 0, 0, 0);
        }
    }

    /**
     * Create a success result.
     */
    public static BenchmarkResult success(
            String benchmarkName, String modelId, String backend,
            Instant startTime, Instant endTime,
            int warmupIterations, int measurementIterations,
            long[] latenciesNanos,
            List<ComparisonResult> validationResults,
            MemoryStats memoryStats
    ) {
        return new BenchmarkResult(
                benchmarkName, modelId, backend, startTime, endTime,
                warmupIterations, measurementIterations, latenciesNanos,
                validationResults, memoryStats, Status.SUCCESS, null
        );
    }

    /**
     * Create an error result.
     */
    public static BenchmarkResult error(
            String benchmarkName, String modelId, String backend,
            Instant startTime, String errorMessage
    ) {
        return new BenchmarkResult(
                benchmarkName, modelId, backend, startTime, Instant.now(),
                0, 0, new long[0], List.of(), null, Status.ERROR, errorMessage
        );
    }

    /**
     * Create a skipped result.
     */
    public static BenchmarkResult skipped(
            String benchmarkName, String modelId, String backend,
            String reason
    ) {
        Instant now = Instant.now();
        return new BenchmarkResult(
                benchmarkName, modelId, backend, now, now,
                0, 0, new long[0], List.of(), null, Status.SKIPPED, reason
        );
    }
}
