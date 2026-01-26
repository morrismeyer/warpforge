package io.surfworks.warpforge.backend.nvidia.stress;

import io.surfworks.warpforge.backend.nvidia.NvidiaBackend;
import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuLease;
import io.surfworks.warpforge.core.concurrency.GpuTaskScope;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.LongSummaryStatistics;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Stress tests for NVIDIA GPU stream operations.
 *
 * <p>These tests validate the NVIDIA backend under high load conditions:
 * <ul>
 *   <li>Virtual thread scaling with GPU operations</li>
 *   <li>Stream creation limits and exhaustion recovery</li>
 *   <li>Concurrent kernel launches and memory transfers</li>
 *   <li>Sustained load over extended periods</li>
 *   <li>Latency distribution under load</li>
 * </ul>
 *
 * <p>Tests are tagged with {@code @Tag("nvidia")} and {@code @Tag("stress")}.
 * Run with: {@code ./gradlew :warpforge-backend-nvidia:stressTest}
 */
@Tag("nvidia")
@Tag("stress")
@DisplayName("NVIDIA Stream Stress Tests")
class NvidiaStreamStressTest {

    private NvidiaBackend backend;

    @BeforeEach
    void setUp() {
        assumeTrue(NvidiaBackend.isCudaAvailable(), "CUDA not available - skipping stress test");
        backend = new NvidiaBackend();
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== Virtual Thread Scaling Tests ====================

    @Nested
    @DisplayName("Virtual Thread Scaling")
    class VirtualThreadScalingTests {

        @Test
        @DisplayName("1000 virtual threads with GPU operations")
        void scale1000VirtualThreads() throws Exception {
            int threadCount = 1000;
            AtomicInteger completed = new AtomicInteger(0);
            AtomicInteger failed = new AtomicInteger(0);

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "stress-1000")) {
                for (int i = 0; i < threadCount; i++) {
                    scope.forkWithStream(lease -> {
                        try {
                            doSmallGpuWork(backend, lease);
                            completed.incrementAndGet();
                        } catch (Exception e) {
                            failed.incrementAndGet();
                        }
                        return null;
                    });
                }
                scope.joinAll();
            }

            System.out.printf("1000 VT stress: completed=%d, failed=%d%n", completed.get(), failed.get());
            assertEquals(threadCount, completed.get(), "All virtual threads should complete");
            assertEquals(0, failed.get(), "No failures expected");
        }

        @Test
        @DisplayName("10000 virtual threads with GPU operations")
        void scale10000VirtualThreads() throws Exception {
            int threadCount = 10000;
            AtomicInteger completed = new AtomicInteger(0);
            AtomicInteger failed = new AtomicInteger(0);

            long startTime = System.nanoTime();

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "stress-10000")) {
                for (int i = 0; i < threadCount; i++) {
                    scope.forkWithStream(lease -> {
                        try {
                            doSmallGpuWork(backend, lease);
                            completed.incrementAndGet();
                        } catch (Exception e) {
                            failed.incrementAndGet();
                        }
                        return null;
                    });
                }
                scope.joinAll();
            }

            long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;
            double throughput = threadCount * 1000.0 / elapsedMs;

            System.out.printf("10000 VT stress: completed=%d, failed=%d, elapsed=%dms, throughput=%.1f ops/sec%n",
                completed.get(), failed.get(), elapsedMs, throughput);

            assertTrue(completed.get() >= threadCount * 0.99, "At least 99% should complete");
        }

        @Test
        @DisplayName("Concurrent scope creation with virtual threads")
        void concurrentScopeCreation() throws Exception {
            int scopeCount = 100;
            int tasksPerScope = 50;
            AtomicInteger totalCompleted = new AtomicInteger(0);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(scopeCount);

            // Create multiple scopes concurrently
            for (int s = 0; s < scopeCount; s++) {
                final int scopeIndex = s;
                Thread.startVirtualThread(() -> {
                    try {
                        startLatch.await();
                        try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-" + scopeIndex)) {
                            for (int t = 0; t < tasksPerScope; t++) {
                                scope.fork(() -> {
                                    totalCompleted.incrementAndGet();
                                    return null;
                                });
                            }
                            scope.joinAll();
                        }
                    } catch (Exception e) {
                        System.err.println("Scope " + scopeIndex + " failed: " + e.getMessage());
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            // Start all scopes at once
            startLatch.countDown();

            // Wait for completion with timeout
            await().atMost(Duration.ofSeconds(60))
                   .until(() -> doneLatch.getCount() == 0);

            int expected = scopeCount * tasksPerScope;
            System.out.printf("Concurrent scope stress: completed=%d/%d%n", totalCompleted.get(), expected);
            assertTrue(totalCompleted.get() >= expected * 0.95, "At least 95% should complete");
        }
    }

    // ==================== Stream Limits Tests ====================

    @Nested
    @DisplayName("Stream Limits")
    class StreamLimitsTests {

        @Test
        @DisplayName("Find maximum stream count")
        void findStreamLimit() {
            List<Long> streamHandles = new ArrayList<>();
            int maxStreams = 0;

            try {
                // Try to create many streams until failure
                for (int i = 0; i < 10000; i++) {
                    long handle = backend.createStream();
                    streamHandles.add(handle);
                    maxStreams++;
                }
            } catch (Exception e) {
                // Expected - hit stream limit
                System.out.println("Stream limit reached: " + e.getMessage());
            } finally {
                // Clean up all created streams
                for (long handle : streamHandles) {
                    try {
                        backend.destroyStream(handle);
                    } catch (Exception ignored) {
                    }
                }
            }

            System.out.printf("Maximum streams created: %d%n", maxStreams);
            assertTrue(maxStreams >= 100, "Should support at least 100 concurrent streams");
        }

        @Test
        @DisplayName("Stream exhaustion recovery")
        void exhaustionRecovery() throws Exception {
            // First, use up streams in a scope
            int streamCount = 500;
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "exhaust")) {
                for (int i = 0; i < streamCount; i++) {
                    scope.forkWithStream(lease -> {
                        lease.synchronize();
                        return null;
                    });
                }
                scope.joinAll();
            }

            // Now create new scope - streams should be recovered
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "recover")) {
                AtomicInteger completed = new AtomicInteger(0);
                for (int i = 0; i < 100; i++) {
                    scope.forkWithStream(lease -> {
                        lease.synchronize();
                        completed.incrementAndGet();
                        return null;
                    });
                }
                scope.joinAll();

                assertEquals(100, completed.get(), "Recovery after exhaustion should work");
            }
        }

        @Test
        @DisplayName("Rapid stream create/destroy cycle")
        void rapidCreateDestroyCycle() {
            int cycles = 5000;
            long startTime = System.nanoTime();

            for (int i = 0; i < cycles; i++) {
                long stream = backend.createStream();
                backend.synchronizeStream(stream);
                backend.destroyStream(stream);
            }

            long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;
            double cyclesPerSec = cycles * 1000.0 / elapsedMs;

            System.out.printf("Rapid create/destroy: %d cycles in %dms (%.1f cycles/sec)%n",
                cycles, elapsedMs, cyclesPerSec);

            assertTrue(cyclesPerSec > 100, "Should achieve >100 stream cycles/sec");
        }
    }

    // ==================== Concurrent Operations Tests ====================

    @Nested
    @DisplayName("Concurrent Operations")
    class ConcurrentOperationsTests {

        @Test
        @DisplayName("Concurrent memory transfers on multiple streams")
        void concurrentMemoryTransfers() throws Exception {
            int streamCount = 32;
            int transfersPerStream = 100;
            AtomicInteger completed = new AtomicInteger(0);

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-memcpy")) {
                for (int s = 0; s < streamCount; s++) {
                    scope.forkWithStream(lease -> {
                        for (int t = 0; t < transfersPerStream; t++) {
                            doSmallGpuWork(backend, lease);
                        }
                        completed.incrementAndGet();
                        return null;
                    });
                }
                scope.joinAll();
            }

            assertEquals(streamCount, completed.get());
            System.out.printf("Concurrent transfers: %d streams x %d transfers completed%n",
                streamCount, transfersPerStream);
        }

        @Test
        @DisplayName("Mixed operations under contention")
        void mixedOperationsUnderContention() throws Exception {
            int taskCount = 200;
            AtomicInteger allocOps = new AtomicInteger(0);
            AtomicInteger transferOps = new AtomicInteger(0);
            AtomicInteger syncOps = new AtomicInteger(0);

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "mixed-ops")) {
                for (int i = 0; i < taskCount; i++) {
                    final int opType = i % 3;
                    scope.forkWithStream(lease -> {
                        switch (opType) {
                            case 0 -> {
                                // Allocation operation
                                TensorSpec spec = TensorSpec.of(ScalarType.F32, 1024);
                                Tensor t = backend.allocateDevice(spec);
                                allocOps.incrementAndGet();
                            }
                            case 1 -> {
                                // Transfer operation
                                doSmallGpuWork(backend, lease);
                                transferOps.incrementAndGet();
                            }
                            case 2 -> {
                                // Sync operation
                                lease.synchronize();
                                syncOps.incrementAndGet();
                            }
                        }
                        return null;
                    });
                }
                scope.joinAll();
            }

            int total = allocOps.get() + transferOps.get() + syncOps.get();
            System.out.printf("Mixed ops: alloc=%d, transfer=%d, sync=%d, total=%d%n",
                allocOps.get(), transferOps.get(), syncOps.get(), total);
            assertEquals(taskCount, total);
        }
    }

    // ==================== Long Running Tests ====================

    @Nested
    @DisplayName("Long Running")
    class LongRunningTests {

        @Test
        @DisplayName("Sustained load for 30 seconds")
        void sustainedLoad30Seconds() throws Exception {
            long durationMs = 30_000;
            AtomicLong operationCount = new AtomicLong(0);
            AtomicInteger failures = new AtomicInteger(0);

            long startTime = System.currentTimeMillis();
            long endTime = startTime + durationMs;

            while (System.currentTimeMillis() < endTime) {
                try (GpuTaskScope scope = GpuTaskScope.open(backend, "sustained")) {
                    for (int i = 0; i < 100; i++) {
                        scope.forkWithStream(lease -> {
                            doSmallGpuWork(backend, lease);
                            operationCount.incrementAndGet();
                            return null;
                        });
                    }
                    scope.joinAll();
                } catch (Exception e) {
                    failures.incrementAndGet();
                }
            }

            long elapsed = System.currentTimeMillis() - startTime;
            double opsPerSec = operationCount.get() * 1000.0 / elapsed;

            System.out.printf("Sustained load: %d ops in %ds (%.1f ops/sec), failures=%d%n",
                operationCount.get(), elapsed / 1000, opsPerSec, failures.get());

            assertTrue(failures.get() < operationCount.get() * 0.01,
                "Less than 1% failure rate expected");
        }

        @Test
        @DisplayName("Bursty load pattern")
        void burstyLoadPattern() throws Exception {
            int bursts = 20;
            int tasksPerBurst = 500;
            AtomicInteger completed = new AtomicInteger(0);
            List<Long> burstTimings = new ArrayList<>();

            for (int burst = 0; burst < bursts; burst++) {
                long burstStart = System.nanoTime();

                try (GpuTaskScope scope = GpuTaskScope.open(backend, "burst-" + burst)) {
                    for (int t = 0; t < tasksPerBurst; t++) {
                        scope.forkWithStream(lease -> {
                            doSmallGpuWork(backend, lease);
                            completed.incrementAndGet();
                            return null;
                        });
                    }
                    scope.joinAll();
                }

                long burstMs = (System.nanoTime() - burstStart) / 1_000_000;
                burstTimings.add(burstMs);

                // Brief pause between bursts
                Thread.sleep(50);
            }

            LongSummaryStatistics stats = burstTimings.stream()
                .mapToLong(Long::longValue)
                .summaryStatistics();

            System.out.printf("Bursty load: %d bursts, avg=%dms, min=%dms, max=%dms%n",
                bursts, (long) stats.getAverage(), stats.getMin(), stats.getMax());

            assertEquals(bursts * tasksPerBurst, completed.get());
        }
    }

    // ==================== Latency Tests ====================

    @Nested
    @DisplayName("Latency")
    class LatencyTests {

        @Test
        @DisplayName("P99 latency under load")
        void p99LatencyUnderLoad() throws Exception {
            int sampleCount = 1000;
            ConcurrentLinkedQueue<Long> latencies = new ConcurrentLinkedQueue<>();

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "latency-test")) {
                for (int i = 0; i < sampleCount; i++) {
                    scope.forkWithStream(lease -> {
                        long start = System.nanoTime();
                        doSmallGpuWork(backend, lease);
                        long latencyNanos = System.nanoTime() - start;
                        latencies.add(latencyNanos);
                        return null;
                    });
                }
                scope.joinAll();
            }

            // Calculate percentiles
            List<Long> sortedLatencies = latencies.stream()
                .sorted()
                .toList();

            long p50 = sortedLatencies.get(sampleCount / 2);
            long p90 = sortedLatencies.get((int) (sampleCount * 0.9));
            long p99 = sortedLatencies.get((int) (sampleCount * 0.99));
            long max = sortedLatencies.get(sampleCount - 1);

            System.out.printf("Latency distribution (n=%d): p50=%dμs, p90=%dμs, p99=%dμs, max=%dμs%n",
                sampleCount, p50 / 1000, p90 / 1000, p99 / 1000, max / 1000);

            // P99 should be reasonable (within 10x of p50)
            assertTrue(p99 < p50 * 20, "P99 latency should be within 20x of median");
        }

        @Test
        @DisplayName("Latency stability over time")
        void latencyStabilityOverTime() throws Exception {
            int batches = 10;
            int samplesPerBatch = 100;
            List<Double> batchAverages = new ArrayList<>();

            for (int batch = 0; batch < batches; batch++) {
                AtomicLong totalLatency = new AtomicLong(0);

                try (GpuTaskScope scope = GpuTaskScope.open(backend, "stability-" + batch)) {
                    for (int i = 0; i < samplesPerBatch; i++) {
                        scope.forkWithStream(lease -> {
                            long start = System.nanoTime();
                            doSmallGpuWork(backend, lease);
                            totalLatency.addAndGet(System.nanoTime() - start);
                            return null;
                        });
                    }
                    scope.joinAll();
                }

                double avgLatencyUs = totalLatency.get() / 1000.0 / samplesPerBatch;
                batchAverages.add(avgLatencyUs);
            }

            double overallAvg = batchAverages.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double maxDeviation = batchAverages.stream()
                .mapToDouble(avg -> Math.abs(avg - overallAvg) / overallAvg)
                .max().orElse(0);

            System.out.printf("Latency stability: overall avg=%.1fμs, max deviation=%.1f%%%n",
                overallAvg, maxDeviation * 100);

            assertTrue(maxDeviation < 0.5, "Latency deviation should be <50% of average");
        }
    }

    // ==================== Helper Methods ====================

    /**
     * Performs a small GPU operation for stress testing.
     * Uses a tiny tensor to minimize individual operation time while
     * still exercising the full GPU operation path.
     */
    private void doSmallGpuWork(GpuBackend backend, GpuLease lease) {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, 256); // 1KB tensor

        try (Arena arena = Arena.ofConfined()) {
            Tensor hostTensor = Tensor.allocate(spec, arena);

            // Small H2D + D2H cycle
            Tensor deviceTensor = backend.copyToDeviceAsync(hostTensor, lease.streamHandle());
            backend.synchronizeStream(lease.streamHandle());
            Tensor resultTensor = backend.copyToHostAsync(deviceTensor, lease.streamHandle());
            backend.synchronizeStream(lease.streamHandle());
        }
    }
}
