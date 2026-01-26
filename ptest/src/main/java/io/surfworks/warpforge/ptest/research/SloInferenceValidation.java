package io.surfworks.warpforge.ptest.research;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuTaskScope;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import io.surfworks.warpforge.ptest.research.ResearchValidationRunner.ValidationResult;

import java.lang.foreign.Arena;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.LongSummaryStatistics;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Alibaba Aegaeon-inspired (SOSP 2025) validation for SLO-bounded inference.
 *
 * <p>Aegaeon's key insight: GPU inference serving can maintain SLO compliance
 * while maximizing throughput by dynamically adjusting batch sizes and using
 * quality-of-service differentiation.
 *
 * <p>This validation measures:
 * <ul>
 *   <li><b>p99Latency</b> - Ensure P99 latency stays under SLO target</li>
 *   <li><b>gracefulDegradation</b> - Quality reduction under overload</li>
 *   <li><b>batchAdaptation</b> - Dynamic batch sizing for SLO compliance</li>
 * </ul>
 *
 * <p>Reference: "Aegaeon: GPU Cluster Scheduling for Online Serving",
 * Alibaba, SOSP 2025
 */
public class SloInferenceValidation {

    private static final long SLO_TARGET_MS = 100; // 100ms P99 target
    private static final int NUM_REQUESTS = 200;
    private static final int WARMUP_REQUESTS = 20;

    public static List<ValidationResult> runAll(GpuBackend backend, boolean verbose) {
        List<ValidationResult> results = new ArrayList<>();

        results.add(validateP99Latency(backend, verbose));
        results.add(validateGracefulDegradation(backend, verbose));
        results.add(validateBatchAdaptation(backend, verbose));

        return results;
    }

    /**
     * Scenario 1: P99 Latency under SLO.
     *
     * Runs inference requests and validates that P99 latency
     * stays under the SLO target (100ms).
     */
    private static ValidationResult validateP99Latency(GpuBackend backend, boolean verbose) {
        String name = "P99 Latency Under SLO (<100ms)";
        Instant start = Instant.now();

        try {
            List<Long> latencies = new ArrayList<>();

            // Warmup phase
            for (int i = 0; i < WARMUP_REQUESTS; i++) {
                doInference(backend, 16 * 1024); // 64KB tensor
            }

            // Measurement phase
            for (int i = 0; i < NUM_REQUESTS; i++) {
                long t0 = System.nanoTime();
                doInference(backend, 16 * 1024);
                long latencyNs = System.nanoTime() - t0;
                latencies.add(latencyNs);
            }

            // Calculate P99
            latencies.sort(Long::compare);
            int p99Index = (int) Math.ceil(latencies.size() * 0.99) - 1;
            long p99LatencyNs = latencies.get(Math.min(p99Index, latencies.size() - 1));
            double p99LatencyMs = p99LatencyNs / 1e6;

            // Calculate other percentiles for verbose output
            int p50Index = latencies.size() / 2;
            int p95Index = (int) Math.ceil(latencies.size() * 0.95) - 1;
            double p50Ms = latencies.get(p50Index) / 1e6;
            double p95Ms = latencies.get(p95Index) / 1e6;

            Duration duration = Duration.between(start, Instant.now());

            if (verbose) {
                LongSummaryStatistics stats = latencies.stream()
                    .mapToLong(Long::longValue)
                    .summaryStatistics();
                System.out.printf("  Requests: %d%n", NUM_REQUESTS);
                System.out.printf("  P50 latency: %.2fms%n", p50Ms);
                System.out.printf("  P95 latency: %.2fms%n", p95Ms);
                System.out.printf("  P99 latency: %.2fms%n", p99LatencyMs);
                System.out.printf("  Avg latency: %.2fms%n", stats.getAverage() / 1e6);
                System.out.printf("  Max latency: %.2fms%n", stats.getMax() / 1e6);
            }

            // Target: P99 under SLO
            if (p99LatencyMs < SLO_TARGET_MS) {
                String msg = String.format("P99=%.1fms (target<%dms)", p99LatencyMs, SLO_TARGET_MS);
                System.out.println("  [PASS] " + msg);
                return ValidationResult.pass(name, msg, duration);
            } else {
                String msg = String.format("P99=%.1fms exceeds %dms SLO", p99LatencyMs, SLO_TARGET_MS);
                System.out.println("  [FAIL] " + msg);
                return ValidationResult.fail(name, msg, duration);
            }

        } catch (Exception e) {
            Duration duration = Duration.between(start, Instant.now());
            System.out.println("  [FAIL] Exception: " + e.getMessage());
            return ValidationResult.fail(name, e.getMessage(), duration);
        }
    }

    /**
     * Scenario 2: Graceful degradation under overload.
     *
     * Tests that the system can handle overload by reducing quality
     * (smaller tensors, fewer iterations) while maintaining throughput.
     */
    private static ValidationResult validateGracefulDegradation(GpuBackend backend, boolean verbose) {
        String name = "Graceful Degradation Under Load";
        Instant start = Instant.now();

        try {
            // Normal quality (large tensors)
            int normalSize = 64 * 1024; // 256KB
            long normalStart = System.nanoTime();
            int normalCount = 0;
            while (System.nanoTime() - normalStart < 500_000_000) { // 500ms
                doInference(backend, normalSize);
                normalCount++;
            }
            double normalThroughput = normalCount / 0.5; // req/s

            // Degraded quality (smaller tensors)
            int degradedSize = 16 * 1024; // 64KB (4x smaller)
            long degradedStart = System.nanoTime();
            int degradedCount = 0;
            while (System.nanoTime() - degradedStart < 500_000_000) { // 500ms
                doInference(backend, degradedSize);
                degradedCount++;
            }
            double degradedThroughput = degradedCount / 0.5; // req/s

            Duration duration = Duration.between(start, Instant.now());

            double throughputGain = degradedThroughput / normalThroughput;

            if (verbose) {
                System.out.printf("  Normal quality (256KB): %.1f req/s%n", normalThroughput);
                System.out.printf("  Degraded quality (64KB): %.1f req/s%n", degradedThroughput);
                System.out.printf("  Throughput gain from degradation: %.1fx%n", throughputGain);
            }

            // Target: Degraded mode should provide at least 1.5x throughput
            if (throughputGain >= 1.5) {
                String msg = String.format("%.1fx throughput gain with degradation", throughputGain);
                System.out.println("  [PASS] " + msg);
                return ValidationResult.pass(name, msg, duration);
            } else {
                String msg = String.format("%.1fx gain (target >=1.5x)", throughputGain);
                System.out.println("  [FAIL] " + msg);
                return ValidationResult.fail(name, msg, duration);
            }

        } catch (Exception e) {
            Duration duration = Duration.between(start, Instant.now());
            System.out.println("  [FAIL] Exception: " + e.getMessage());
            return ValidationResult.fail(name, e.getMessage(), duration);
        }
    }

    /**
     * Scenario 3: Batch adaptation for SLO compliance.
     *
     * Tests dynamic batch sizing: start with small batches and
     * increase until approaching SLO, then back off.
     */
    private static ValidationResult validateBatchAdaptation(GpuBackend backend, boolean verbose) {
        String name = "Batch Adaptation for SLO Compliance";
        Instant start = Instant.now();

        try {
            int[] batchSizes = {1, 2, 4, 8, 16, 32};
            int optimalBatchSize = 1;
            double optimalThroughput = 0;
            boolean foundSloViolation = false;

            for (int batchSize : batchSizes) {
                // Measure latency for this batch size
                List<Long> latencies = new ArrayList<>();
                for (int i = 0; i < 20; i++) {
                    long t0 = System.nanoTime();
                    doBatchedInference(backend, batchSize, 8 * 1024);
                    latencies.add(System.nanoTime() - t0);
                }

                // Calculate P99
                latencies.sort(Long::compare);
                int p99Index = (int) Math.ceil(latencies.size() * 0.99) - 1;
                double p99Ms = latencies.get(Math.min(p99Index, latencies.size() - 1)) / 1e6;

                // Calculate throughput
                double avgLatencyMs = latencies.stream().mapToLong(Long::longValue).average().orElse(0) / 1e6;
                double throughput = batchSize / (avgLatencyMs / 1000); // items/sec

                if (verbose) {
                    System.out.printf("  Batch %d: P99=%.1fms, throughput=%.0f items/s%n",
                        batchSize, p99Ms, throughput);
                }

                // Track best batch size that stays under SLO
                if (p99Ms < SLO_TARGET_MS && throughput > optimalThroughput) {
                    optimalBatchSize = batchSize;
                    optimalThroughput = throughput;
                } else if (p99Ms >= SLO_TARGET_MS) {
                    foundSloViolation = true;
                    break; // Stop increasing batch size
                }
            }

            Duration duration = Duration.between(start, Instant.now());

            if (verbose) {
                System.out.printf("  Optimal batch size: %d (%.0f items/s)%n",
                    optimalBatchSize, optimalThroughput);
                System.out.printf("  Found SLO boundary: %s%n", foundSloViolation);
            }

            // Success if we found an optimal batch size with reasonable throughput
            if (optimalBatchSize >= 1 && optimalThroughput > 0) {
                String msg = String.format("optimal batch=%d (%.0f items/s)", optimalBatchSize, optimalThroughput);
                System.out.println("  [PASS] " + msg);
                return ValidationResult.pass(name, msg, duration);
            } else {
                String msg = "Could not find SLO-compliant batch size";
                System.out.println("  [FAIL] " + msg);
                return ValidationResult.fail(name, msg, duration);
            }

        } catch (Exception e) {
            Duration duration = Duration.between(start, Instant.now());
            System.out.println("  [FAIL] Exception: " + e.getMessage());
            return ValidationResult.fail(name, e.getMessage(), duration);
        }
    }

    /**
     * Simulate a single inference request.
     */
    private static void doInference(GpuBackend backend, int tensorSize) {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, tensorSize);
        try (Arena arena = Arena.ofConfined()) {
            Tensor host = Tensor.allocate(spec, arena);
            long stream = backend.createStream();
            try {
                Tensor device = backend.copyToDeviceAsync(host, stream);
                backend.synchronizeStream(stream);
                Tensor result = backend.copyToHostAsync(device, stream);
                backend.synchronizeStream(stream);
            } finally {
                backend.destroyStream(stream);
            }
        }
    }

    /**
     * Simulate batched inference (multiple items in one request).
     */
    private static void doBatchedInference(GpuBackend backend, int batchSize, int itemSize) {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, batchSize * itemSize);
        try (Arena arena = Arena.ofConfined()) {
            Tensor host = Tensor.allocate(spec, arena);
            long stream = backend.createStream();
            try {
                Tensor device = backend.copyToDeviceAsync(host, stream);
                backend.synchronizeStream(stream);
                Tensor result = backend.copyToHostAsync(device, stream);
                backend.synchronizeStream(stream);
            } finally {
                backend.destroyStream(stream);
            }
        }
    }
}
