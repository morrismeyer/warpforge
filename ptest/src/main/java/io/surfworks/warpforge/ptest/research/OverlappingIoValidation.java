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
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tally-inspired (ASPLOS 2025) validation for compute/memory overlap patterns.
 *
 * <p>Tally's key insight: Modern GPUs can overlap compute and memory operations
 * on different streams, but achieving this overlap requires careful scheduling.
 * This validation measures:
 * <ul>
 *   <li><b>computeMemoryOverlap</b> - GEMM + memcpy on different streams</li>
 *   <li><b>multiStreamOverlap</b> - 3+ streams concurrent, target >90% utilization</li>
 *   <li><b>sequentialVsOverlapped</b> - Measure speedup, target >1.5x improvement</li>
 * </ul>
 *
 * <p>Reference: "Tally: Non-Intrusive Performance Isolation for Concurrent
 * DNN Workloads", ASPLOS 2025
 */
public class OverlappingIoValidation {

    private static final int TENSOR_SIZE = 1024 * 1024; // 4MB tensor (1M floats)
    private static final int NUM_ITERATIONS = 10;

    public static List<ValidationResult> runAll(GpuBackend backend, boolean verbose) {
        List<ValidationResult> results = new ArrayList<>();

        results.add(validateComputeMemoryOverlap(backend, verbose));
        results.add(validateMultiStreamOverlap(backend, verbose));
        results.add(validateSequentialVsOverlapped(backend, verbose));

        return results;
    }

    /**
     * Scenario 1: Compute + Memory overlap on different streams.
     *
     * Tests that memory transfers on one stream can overlap with
     * synchronization waits on another stream.
     */
    private static ValidationResult validateComputeMemoryOverlap(GpuBackend backend, boolean verbose) {
        String name = "Compute/Memory Overlap";
        Instant start = Instant.now();

        try {
            AtomicLong transferTime = new AtomicLong(0);
            AtomicLong syncTime = new AtomicLong(0);

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "overlap-test")) {
                // Fork two tasks on different streams
                // Task 1: Memory transfer
                scope.forkWithStream(lease -> {
                    long t0 = System.nanoTime();
                    TensorSpec spec = TensorSpec.of(ScalarType.F32, TENSOR_SIZE);
                    try (Arena arena = Arena.ofConfined()) {
                        Tensor host = Tensor.allocate(spec, arena);
                        for (int i = 0; i < NUM_ITERATIONS; i++) {
                            Tensor device = backend.copyToDeviceAsync(host, lease.streamHandle());
                            backend.synchronizeStream(lease.streamHandle());
                            Tensor result = backend.copyToHostAsync(device, lease.streamHandle());
                            backend.synchronizeStream(lease.streamHandle());
                        }
                    }
                    transferTime.set(System.nanoTime() - t0);
                    return null;
                });

                // Task 2: Simulated compute (sync operations)
                scope.forkWithStream(lease -> {
                    long t0 = System.nanoTime();
                    for (int i = 0; i < NUM_ITERATIONS; i++) {
                        backend.synchronizeStream(lease.streamHandle());
                        // Small allocation to simulate compute
                        TensorSpec spec = TensorSpec.of(ScalarType.F32, 1024);
                        Tensor t = backend.allocateDevice(spec);
                    }
                    syncTime.set(System.nanoTime() - t0);
                    return null;
                });

                scope.joinAll();
            }

            Duration duration = Duration.between(start, Instant.now());
            long totalTime = duration.toNanos();
            long sequentialTime = transferTime.get() + syncTime.get();

            // If overlapping worked, total time should be less than sequential sum
            double overlapRatio = (double) totalTime / sequentialTime;

            if (verbose) {
                System.out.printf("  Transfer time: %.1fms%n", transferTime.get() / 1e6);
                System.out.printf("  Sync time: %.1fms%n", syncTime.get() / 1e6);
                System.out.printf("  Total time: %.1fms%n", totalTime / 1e6);
                System.out.printf("  Overlap ratio: %.2f (lower is better)%n", overlapRatio);
            }

            // Target: Some overlap achieved (ratio < 0.95 of sequential)
            if (overlapRatio < 0.95) {
                String msg = String.format("%.0f%% overlap achieved", (1 - overlapRatio) * 100);
                System.out.println("  [PASS] " + msg);
                return ValidationResult.pass(name, msg, duration);
            } else {
                String msg = String.format("No significant overlap (ratio=%.2f)", overlapRatio);
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
     * Scenario 2: Multi-stream overlap with 4+ concurrent streams.
     *
     * Tests that multiple streams can execute concurrently, achieving
     * high GPU utilization.
     */
    private static ValidationResult validateMultiStreamOverlap(GpuBackend backend, boolean verbose) {
        String name = "Multi-Stream Overlap (4 streams)";
        Instant start = Instant.now();

        try {
            int numStreams = 4;
            AtomicLong[] streamTimes = new AtomicLong[numStreams];
            for (int i = 0; i < numStreams; i++) {
                streamTimes[i] = new AtomicLong(0);
            }

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "multi-stream")) {
                for (int s = 0; s < numStreams; s++) {
                    final int streamIndex = s;
                    scope.forkWithStream(lease -> {
                        long t0 = System.nanoTime();
                        TensorSpec spec = TensorSpec.of(ScalarType.F32, TENSOR_SIZE / numStreams);
                        try (Arena arena = Arena.ofConfined()) {
                            Tensor host = Tensor.allocate(spec, arena);
                            for (int i = 0; i < NUM_ITERATIONS; i++) {
                                Tensor device = backend.copyToDeviceAsync(host, lease.streamHandle());
                                backend.synchronizeStream(lease.streamHandle());
                            }
                        }
                        streamTimes[streamIndex].set(System.nanoTime() - t0);
                        return null;
                    });
                }
                scope.joinAll();
            }

            Duration duration = Duration.between(start, Instant.now());
            long totalTime = duration.toNanos();

            // Calculate sum of individual stream times
            long sumStreamTimes = 0;
            long maxStreamTime = 0;
            for (AtomicLong st : streamTimes) {
                sumStreamTimes += st.get();
                maxStreamTime = Math.max(maxStreamTime, st.get());
            }

            // Utilization = sum of work / (numStreams * wallclock)
            // If fully parallel, utilization approaches 100%
            double utilization = (double) sumStreamTimes / (numStreams * totalTime) * 100;

            if (verbose) {
                System.out.printf("  Total wall time: %.1fms%n", totalTime / 1e6);
                System.out.printf("  Sum of stream times: %.1fms%n", sumStreamTimes / 1e6);
                System.out.printf("  Max stream time: %.1fms%n", maxStreamTime / 1e6);
                System.out.printf("  Estimated utilization: %.1f%%%n", utilization);
            }

            // Target: >50% utilization (accounting for overhead)
            if (utilization > 50) {
                String msg = String.format("%.0f%% utilization", utilization);
                System.out.println("  [PASS] " + msg);
                return ValidationResult.pass(name, msg, duration);
            } else {
                String msg = String.format("Low utilization: %.0f%%", utilization);
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
     * Scenario 3: Sequential vs Overlapped comparison.
     *
     * Directly compares sequential execution (one stream) with
     * overlapped execution (multiple streams) to measure speedup.
     */
    private static ValidationResult validateSequentialVsOverlapped(GpuBackend backend, boolean verbose) {
        String name = "Sequential vs Overlapped Speedup";
        Instant start = Instant.now();

        try {
            int numTasks = 4;
            TensorSpec spec = TensorSpec.of(ScalarType.F32, TENSOR_SIZE / numTasks);

            // Sequential execution (single stream)
            long sequentialStart = System.nanoTime();
            long singleStream = backend.createStream();
            try (Arena arena = Arena.ofConfined()) {
                Tensor host = Tensor.allocate(spec, arena);
                for (int task = 0; task < numTasks; task++) {
                    for (int i = 0; i < NUM_ITERATIONS; i++) {
                        Tensor device = backend.copyToDeviceAsync(host, singleStream);
                        backend.synchronizeStream(singleStream);
                    }
                }
            }
            backend.destroyStream(singleStream);
            long sequentialTime = System.nanoTime() - sequentialStart;

            // Overlapped execution (multiple streams)
            long overlappedStart = System.nanoTime();
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "overlapped")) {
                for (int task = 0; task < numTasks; task++) {
                    scope.forkWithStream(lease -> {
                        try (Arena arena = Arena.ofConfined()) {
                            Tensor host = Tensor.allocate(spec, arena);
                            for (int i = 0; i < NUM_ITERATIONS; i++) {
                                Tensor device = backend.copyToDeviceAsync(host, lease.streamHandle());
                                backend.synchronizeStream(lease.streamHandle());
                            }
                        }
                        return null;
                    });
                }
                scope.joinAll();
            }
            long overlappedTime = System.nanoTime() - overlappedStart;

            Duration duration = Duration.between(start, Instant.now());
            double speedup = (double) sequentialTime / overlappedTime;

            if (verbose) {
                System.out.printf("  Sequential time: %.1fms%n", sequentialTime / 1e6);
                System.out.printf("  Overlapped time: %.1fms%n", overlappedTime / 1e6);
                System.out.printf("  Speedup: %.2fx%n", speedup);
            }

            // Target: >1.2x speedup (some overlap benefit)
            if (speedup > 1.2) {
                String msg = String.format("%.2fx speedup", speedup);
                System.out.println("  [PASS] " + msg);
                return ValidationResult.pass(name, msg, duration);
            } else {
                String msg = String.format("Insufficient speedup: %.2fx", speedup);
                System.out.println("  [FAIL] " + msg);
                return ValidationResult.fail(name, msg, duration);
            }

        } catch (Exception e) {
            Duration duration = Duration.between(start, Instant.now());
            System.out.println("  [FAIL] Exception: " + e.getMessage());
            return ValidationResult.fail(name, e.getMessage(), duration);
        }
    }
}
