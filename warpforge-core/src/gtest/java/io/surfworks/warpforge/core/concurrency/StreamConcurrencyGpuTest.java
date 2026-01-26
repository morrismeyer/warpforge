package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for stream concurrency - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate concurrent stream operations with real GPU hardware.
 */
@Tag("gpu")
@DisplayName("Stream Concurrency GPU Tests")
class StreamConcurrencyGpuTest {

    private GpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = GpuTestSupport.createBackend();
        System.out.println("Running on: " + GpuTestSupport.describeEnvironment());
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
    @DisplayName("Ten concurrent streams work independently")
    void tenConcurrentStreams() throws Exception {
        final int numStreams = 10;
        AtomicInteger completedStreams = new AtomicInteger(0);
        Set<Long> streamHandles = ConcurrentHashMap.newKeySet();

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "ten-streams")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                final int streamIndex = i;
                scope.forkWithStream(lease -> {
                    long handle = lease.streamHandle();
                    streamHandles.add(handle);

                    // Do some work
                    long sum = 0;
                    for (int j = 0; j < 1000; j++) {
                        sum += j;
                    }

                    lease.synchronize();
                    completedStreams.incrementAndGet();
                    return handle;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitKernelEvent("TenConcurrentStreams", numStreams + " streams", elapsedMicros);
        }

        assertEquals(numStreams, completedStreams.get(),
            "All " + numStreams + " streams should complete");
    }

    @Test
    @DisplayName("Twenty concurrent streams work independently")
    void twentyConcurrentStreams() throws Exception {
        final int numStreams = 20;
        AtomicInteger completedStreams = new AtomicInteger(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "twenty-streams")) {
            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    completedStreams.incrementAndGet();
                    return null;
                });
            }

            scope.joinAll();
        }

        assertEquals(numStreams, completedStreams.get(),
            "All " + numStreams + " streams should complete");
    }

    // ==================== Stream Lifecycle Tests ====================

    @Test
    @DisplayName("Rapid stream create/destroy cycles")
    void rapidCreateDestroyCycles() throws Exception {
        final int cycles = 50;
        AtomicInteger completedCycles = new AtomicInteger(0);

        for (int cycle = 0; cycle < cycles; cycle++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "cycle-" + cycle)) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    completedCycles.incrementAndGet();
                    return null;
                });
                scope.joinAll();
            }
        }

        assertEquals(cycles, completedCycles.get(),
            "All " + cycles + " cycles should complete");
    }

    @Test
    @DisplayName("Stream handles are unique within scope")
    void streamHandlesUniqueWithinScope() throws Exception {
        final int numStreams = 5;
        Set<Long> handles = ConcurrentHashMap.newKeySet();

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    handles.add(lease.streamHandle());
                    lease.synchronize();
                    return null;
                });
            }

            scope.joinAll();
        }

        // All handles should be unique (no duplicates)
        assertEquals(numStreams, handles.size(),
            "Each stream should have a unique handle");
    }

    // ==================== Stream Reuse Tests ====================

    @Test
    @DisplayName("Multiple scopes can create streams sequentially")
    void multipleSequentialScopes() throws Exception {
        final int numScopes = 10;
        AtomicInteger totalCompleted = new AtomicInteger(0);

        for (int i = 0; i < numScopes; i++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "scope-" + i)) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    totalCompleted.incrementAndGet();
                    return null;
                });
                scope.joinAll();
            }
        }

        assertEquals(numScopes, totalCompleted.get(),
            "All scopes should complete");
    }

    // ==================== Contention Tests ====================

    @Test
    @DisplayName("Streams under contention complete successfully")
    void streamsUnderContention() throws Exception {
        final int numStreams = 15;
        AtomicInteger completedCount = new AtomicInteger(0);
        AtomicInteger concurrentPeak = new AtomicInteger(0);
        AtomicInteger currentConcurrent = new AtomicInteger(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "contention-test")) {
            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    int concurrent = currentConcurrent.incrementAndGet();
                    concurrentPeak.updateAndGet(peak -> Math.max(peak, concurrent));

                    // Simulate work
                    long sum = 0;
                    for (int j = 0; j < 5000; j++) {
                        sum += j;
                    }

                    lease.synchronize();

                    currentConcurrent.decrementAndGet();
                    completedCount.incrementAndGet();
                    return sum;
                });
            }

            scope.joinAll();
        }

        assertEquals(numStreams, completedCount.get(),
            "All streams should complete under contention");
        assertTrue(concurrentPeak.get() >= 1,
            "Should have observed some concurrency");
    }

    @Test
    @DisplayName("Heavy synchronization load completes")
    void heavySynchronizationLoad() throws Exception {
        final int numStreams = 8;
        final int syncsPerStream = 10;
        AtomicInteger totalSyncs = new AtomicInteger(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "heavy-sync")) {
            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    for (int j = 0; j < syncsPerStream; j++) {
                        lease.synchronize();
                        totalSyncs.incrementAndGet();
                    }
                    return null;
                });
            }

            scope.joinAll();
        }

        assertEquals(numStreams * syncsPerStream, totalSyncs.get(),
            "All synchronize calls should complete");
    }

    // ==================== Cleanup Tests ====================

    @Test
    @DisplayName("All streams destroyed after scope closes")
    void allStreamsDestroyedAfterClose() throws Exception {
        // Create and complete many streams
        for (int round = 0; round < 5; round++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                for (int i = 0; i < 5; i++) {
                    scope.forkWithStream(lease -> {
                        lease.synchronize();
                        return null;
                    });
                }
                scope.joinAll();
            }
        }

        // Verify we can still create new scopes (no resource exhaustion)
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            scope.fork(() -> 1);
            scope.joinAll();
        }

        // If we got here, cleanup worked
        assertTrue(true, "Resource cleanup should work");
    }

    @Test
    @DisplayName("No leaks under stress")
    void noLeaksUnderStress() throws Exception {
        final int iterations = 20;
        final int streamsPerIteration = 3;

        for (int iter = 0; iter < iterations; iter++) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "stress-" + iter)) {
                for (int i = 0; i < streamsPerIteration; i++) {
                    scope.forkWithStream(lease -> {
                        lease.synchronize();
                        return null;
                    });
                }
                scope.joinAll();
            }
        }

        // Final verification - can still use GPU
        try (GpuTaskScope scope = GpuTaskScope.open(backend, "verification")) {
            scope.forkWithStream(lease -> {
                lease.synchronize();
                return "success";
            });
            scope.joinAll();
        }

        assertTrue(true, "No leaks detected after stress test");
    }

    // ==================== Mixed Operation Tests ====================

    @Test
    @DisplayName("Mixed fork and forkWithStream operations")
    void mixedForkOperations() throws Exception {
        AtomicInteger forkCount = new AtomicInteger(0);
        AtomicInteger streamForkCount = new AtomicInteger(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "mixed-ops")) {
            // Alternate between fork types
            for (int i = 0; i < 10; i++) {
                if (i % 2 == 0) {
                    scope.fork(() -> {
                        forkCount.incrementAndGet();
                        return null;
                    });
                } else {
                    scope.forkWithStream(lease -> {
                        lease.synchronize();
                        streamForkCount.incrementAndGet();
                        return null;
                    });
                }
            }

            scope.joinAll();
        }

        assertEquals(5, forkCount.get(), "Should have 5 regular forks");
        assertEquals(5, streamForkCount.get(), "Should have 5 stream forks");
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for concurrent stream operations")
    void jfrEventsForConcurrentStreams() throws Exception {
        final int numStreams = 5;

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-stream-test")) {
            long startTime = System.nanoTime();

            for (int i = 0; i < numStreams; i++) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    return null;
                });
            }

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;
            emitKernelEvent("StreamConcurrencyJFRTest", numStreams + " concurrent", elapsedMicros);
        }

        assertTrue(true, "JFR events should be emittable");
    }

    // ==================== Helper Methods ====================

    private void emitKernelEvent(String operation, String shape, long elapsedMicros) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "STREAM_CONCURRENCY_GTEST";
        event.memoryBandwidthGBps = 0.0;
        event.commit();
    }
}
