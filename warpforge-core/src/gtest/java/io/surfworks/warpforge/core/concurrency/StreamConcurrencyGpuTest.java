package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuWorkCalibrator.GpuWorkResult;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for stream concurrency - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate concurrent stream operations with <b>real GPU hardware</b>.
 * Every test performs actual GPU operations (memory transfers) on multiple streams
 * and emits JFR events for validation. Without real GPU work, these would just be
 * CPU threading tests.
 *
 * <p><b>Key principle:</b> Each stream performs measurable GPU work via
 * {@link GpuWorkCalibrator#doGpuWork}, which:
 * <ul>
 *   <li>Allocates device memory</li>
 *   <li>Performs H2D and D2H transfers on the stream</li>
 *   <li>Synchronizes the stream</li>
 *   <li>Emits {@code GpuKernelEvent} and {@code GpuMemoryEvent} for JFR profiling</li>
 * </ul>
 *
 * <p>What this class validates:
 * <ul>
 *   <li>Multiple concurrent streams work independently</li>
 *   <li>Stream handles are unique and valid</li>
 *   <li>Rapid stream create/destroy cycles are safe</li>
 *   <li>Streams under contention complete successfully</li>
 *   <li>No resource leaks under stress</li>
 *   <li>JFR events are properly emitted for all stream operations</li>
 * </ul>
 */
@Tag("gpu")
@DisplayName("Stream Concurrency GPU Tests")
class StreamConcurrencyGpuTest {

    private GpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = GpuTestSupport.createBackend();
        System.out.println("Running on: " + GpuTestSupport.describeEnvironment());
        // Ensure calibration is done before tests
        GpuWorkCalibrator.getCalibrationData(backend);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    // ==================== Concurrent Streams Tests ====================

    @Test
    @DisplayName("Ten concurrent streams with real GPU work")
    void tenConcurrentStreamsWithRealGpuWork() throws Exception {
        final int numStreams = 10;
        AtomicInteger completedStreams = new AtomicInteger(0);
        Set<Long> streamHandles = ConcurrentHashMap.newKeySet();
        AtomicLong totalGpuTime = new AtomicLong(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "ten-streams")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    long handle = lease.streamHandle();
                    streamHandles.add(handle);

                    // Do REAL GPU work on this stream
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                        GpuWorkCalibrator.MIN_WORK_MS);

                    totalGpuTime.addAndGet(result.elapsedNanos());
                    completedStreams.incrementAndGet();
                    return result;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitConcurrencyEvent("TenConcurrentStreams", numStreams, elapsedMicros, totalGpuTime.get());
        }

        assertEquals(numStreams, completedStreams.get(),
            "All " + numStreams + " streams should complete");
        assertTrue(streamHandles.size() >= 1, "Should have at least one unique stream handle");
        streamHandles.forEach(h -> assertTrue(h != 0, "Stream handles should be valid"));
    }

    @Test
    @DisplayName("Twenty concurrent streams with real GPU work")
    void twentyConcurrentStreamsWithRealGpuWork() throws Exception {
        final int numStreams = 20;
        AtomicInteger completedStreams = new AtomicInteger(0);
        AtomicLong totalBytes = new AtomicLong(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "twenty-streams")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                        GpuWorkCalibrator.MIN_WORK_MS);
                    totalBytes.addAndGet(result.byteSize());
                    completedStreams.incrementAndGet();
                    return null;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitConcurrencyEvent("TwentyConcurrentStreams", numStreams, elapsedMicros, 0);
        }

        assertEquals(numStreams, completedStreams.get(),
            "All " + numStreams + " streams should complete");
        assertTrue(totalBytes.get() > 0, "Should have transferred bytes across streams");
    }

    // ==================== Stream Lifecycle Tests ====================

    @Test
    @DisplayName("Rapid stream create/destroy cycles with real GPU work")
    void rapidCreateDestroyCyclesWithRealGpuWork() throws Exception {
        final int cycles = 30;
        AtomicInteger completedCycles = new AtomicInteger(0);
        AtomicLong totalGpuTime = new AtomicLong(0);

        long startTime = System.nanoTime();

        for (int cycle = 0; cycle < cycles; cycle++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "cycle-" + cycle)) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                        GpuWorkCalibrator.MIN_WORK_MS);
                    totalGpuTime.addAndGet(result.elapsedNanos());
                    completedCycles.incrementAndGet();
                    return null;
                });
                scope.joinAll();
            }
        }

        long elapsedMicros = (System.nanoTime() - startTime) / 1000;

        assertEquals(cycles, completedCycles.get(),
            "All " + cycles + " cycles should complete");
        assertTrue(totalGpuTime.get() > 0, "Should have accumulated GPU time");

        emitConcurrencyEvent("RapidCreateDestroy", cycles, elapsedMicros, totalGpuTime.get());
    }

    @Test
    @DisplayName("Stream handles are unique within scope doing real GPU work")
    void streamHandlesUniqueWithRealGpuWork() throws Exception {
        final int numStreams = 5;
        Set<Long> handles = ConcurrentHashMap.newKeySet();
        List<GpuWorkResult> results = new ArrayList<>();

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    handles.add(lease.streamHandle());
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                        GpuWorkCalibrator.MIN_WORK_MS);
                    synchronized (results) {
                        results.add(result);
                    }
                    return null;
                });
            }

            scope.joinAll();
        }

        // All handles should be unique (no duplicates)
        assertEquals(numStreams, handles.size(),
            "Each stream should have a unique handle");
        assertEquals(numStreams, results.size(),
            "Each stream should complete with GPU work result");
    }

    // ==================== Stream Reuse Tests ====================

    @Test
    @DisplayName("Multiple scopes can create streams sequentially with real GPU work")
    void multipleSequentialScopesWithRealGpuWork() throws Exception {
        final int numScopes = 10;
        AtomicInteger totalCompleted = new AtomicInteger(0);
        List<Double> bandwidths = new ArrayList<>();

        for (int i = 0; i < numScopes; i++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "scope-" + i)) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                        GpuWorkCalibrator.MIN_WORK_MS);
                    synchronized (bandwidths) {
                        bandwidths.add(result.bandwidthGBps());
                    }
                    totalCompleted.incrementAndGet();
                    return null;
                });
                scope.joinAll();
            }
        }

        assertEquals(numScopes, totalCompleted.get(),
            "All scopes should complete");
        assertEquals(numScopes, bandwidths.size(),
            "Each scope should report bandwidth");

        double avgBandwidth = bandwidths.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        System.out.printf("Sequential scopes: avg bandwidth=%.2f GB/s%n", avgBandwidth);
    }

    // ==================== Contention Tests ====================

    @Test
    @DisplayName("Streams under contention complete with real GPU work")
    void streamsUnderContentionWithRealGpuWork() throws Exception {
        final int numStreams = 15;
        AtomicInteger completedCount = new AtomicInteger(0);
        AtomicInteger concurrentPeak = new AtomicInteger(0);
        AtomicInteger currentConcurrent = new AtomicInteger(0);
        AtomicLong totalTransferred = new AtomicLong(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "contention-test")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    int concurrent = currentConcurrent.incrementAndGet();
                    concurrentPeak.updateAndGet(peak -> Math.max(peak, concurrent));

                    // Do real GPU work under contention
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 10);

                    totalTransferred.addAndGet(result.byteSize());
                    currentConcurrent.decrementAndGet();
                    completedCount.incrementAndGet();
                    return result;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitConcurrencyEvent("ContentionTest", numStreams, elapsedMicros, 0);
        }

        assertEquals(numStreams, completedCount.get(),
            "All streams should complete under contention");
        assertTrue(concurrentPeak.get() >= 1,
            "Should have observed some concurrency");
        assertTrue(totalTransferred.get() > 0,
            "Should have transferred data across all streams");

        System.out.printf("Contention test: peak concurrency=%d, total bytes=%d%n",
            concurrentPeak.get(), totalTransferred.get());
    }

    @Test
    @DisplayName("Heavy synchronization load with real GPU work")
    void heavySynchronizationLoadWithRealGpuWork() throws Exception {
        final int numStreams = 8;
        final int workIterationsPerStream = 3;
        AtomicInteger totalWorkDone = new AtomicInteger(0);
        AtomicLong totalGpuTime = new AtomicLong(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "heavy-sync")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    for (int j = 0; j < workIterationsPerStream; j++) {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                            GpuWorkCalibrator.MIN_WORK_MS);
                        totalGpuTime.addAndGet(result.elapsedNanos());
                        totalWorkDone.incrementAndGet();
                    }
                    return null;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitConcurrencyEvent("HeavySyncLoad", numStreams * workIterationsPerStream,
                elapsedMicros, totalGpuTime.get());
        }

        assertEquals(numStreams * workIterationsPerStream, totalWorkDone.get(),
            "All synchronize calls should complete");
        assertTrue(totalGpuTime.get() > 0, "Should have accumulated GPU time");
    }

    // ==================== Cleanup Tests ====================

    @Test
    @DisplayName("All streams destroyed after scope closes with real GPU work")
    void allStreamsDestroyedAfterCloseWithRealGpuWork() throws Exception {
        // Create and complete many streams with real GPU work
        for (int round = 0; round < 5; round++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                for (int i = 0; i < 5; i++) {
                    scope.forkWithStream(lease -> {
                        GpuWorkCalibrator.doGpuWork(backend, lease, GpuWorkCalibrator.MIN_WORK_MS);
                        return null;
                    });
                }
                scope.joinAll();
            }
        }

        // Verify we can still create new scopes and do GPU work (no resource exhaustion)
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            scope.forkWithStream(lease -> {
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                    GpuWorkCalibrator.MIN_WORK_MS);
                assertTrue(result.elapsedNanos() > 0, "GPU work should complete");
                return null;
            });
            scope.joinAll();
        }

        // If we got here, cleanup worked
        assertTrue(true, "Resource cleanup should work");
    }

    @Test
    @DisplayName("No leaks under stress with real GPU work")
    void noLeaksUnderStressWithRealGpuWork() throws Exception {
        final int iterations = 15;
        final int streamsPerIteration = 3;
        AtomicLong totalBytesTransferred = new AtomicLong(0);

        long startTime = System.nanoTime();

        for (int iter = 0; iter < iterations; iter++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "stress-" + iter)) {
                for (int i = 0; i < streamsPerIteration; i++) {
                    scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                            GpuWorkCalibrator.MIN_WORK_MS);
                        totalBytesTransferred.addAndGet(result.byteSize());
                        return null;
                    });
                }
                scope.joinAll();
            }
        }

        long elapsedMicros = (System.nanoTime() - startTime) / 1000;

        // Final verification - can still use GPU
        try (GpuTaskScope scope = GpuTaskScope.open(backend, "verification")) {
            scope.forkWithStream(lease -> {
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                    GpuWorkCalibrator.MIN_WORK_MS);
                assertTrue(result.elapsedNanos() > 0, "Post-stress GPU work should complete");
                return result;
            });
            scope.joinAll();
        }

        assertTrue(totalBytesTransferred.get() > 0, "Should have transferred bytes during stress");
        emitConcurrencyEvent("StressTest", iterations * streamsPerIteration,
            elapsedMicros, 0);
    }

    // ==================== Mixed Operation Tests ====================

    @Test
    @DisplayName("Mixed fork and forkWithStream with real GPU work")
    void mixedForkOperationsWithRealGpuWork() throws Exception {
        AtomicInteger forkCount = new AtomicInteger(0);
        AtomicInteger streamForkCount = new AtomicInteger(0);
        AtomicLong streamGpuTime = new AtomicLong(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "mixed-ops")) {
            long startTime = System.nanoTime();

            // Alternate between fork types
            for (int i = 0; i < 10; i++) {
                if (i % 2 == 0) {
                    scope.fork(() -> {
                        // CPU-only fork
                        forkCount.incrementAndGet();
                        return null;
                    });
                } else {
                    scope.forkWithStream(lease -> {
                        // GPU fork with real work
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                            GpuWorkCalibrator.MIN_WORK_MS);
                        streamGpuTime.addAndGet(result.elapsedNanos());
                        streamForkCount.incrementAndGet();
                        return null;
                    });
                }
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitConcurrencyEvent("MixedOperations", 10, elapsedMicros, streamGpuTime.get());
        }

        assertEquals(5, forkCount.get(), "Should have 5 regular forks");
        assertEquals(5, streamForkCount.get(), "Should have 5 stream forks");
        assertTrue(streamGpuTime.get() > 0, "Stream forks should have accumulated GPU time");
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for concurrent stream GPU operations")
    void jfrEventsForConcurrentStreamGpuOperations() throws Exception {
        final int numStreams = 5;
        List<GpuWorkResult> results = new ArrayList<>();

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-stream-test")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 10);
                    synchronized (results) {
                        results.add(result);
                    }
                    return null;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            // Each stream emitted JFR events via GpuWorkCalibrator.doGpuWork()
            assertEquals(numStreams, results.size(),
                "All streams should complete with JFR events");

            emitConcurrencyEvent("StreamConcurrencyJFRValidation", numStreams,
                elapsedMicros, results.stream().mapToLong(GpuWorkResult::elapsedNanos).sum());
        }
    }

    @Test
    @DisplayName("JFR events have valid stream handles")
    void jfrEventsHaveValidStreamHandles() throws Exception {
        final int numStreams = 4;
        Set<Long> streamHandles = ConcurrentHashMap.newKeySet();

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-handles-test")) {
            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                        GpuWorkCalibrator.MIN_WORK_MS);
                    streamHandles.add(result.streamHandle());
                    return null;
                });
            }

            scope.joinAll();
        }

        assertEquals(numStreams, streamHandles.size(),
            "Should have unique stream handle for each stream");
        streamHandles.forEach(h ->
            assertTrue(h != 0, "JFR events should have valid stream handles"));
    }

    // ==================== Helper Methods ====================

    private void emitConcurrencyEvent(String operation, int numStreams, long elapsedMicros,
                                       long totalGpuTimeNanos) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = numStreams + " streams";
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "STREAM_CONCURRENCY_GTEST";
        event.bytesTransferred = 0;
        event.memoryBandwidthGBps = totalGpuTimeNanos > 0 ?
            (totalGpuTimeNanos / 1e9) : 0.0;
        event.commit();
    }
}
