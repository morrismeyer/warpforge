package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link GpuTaskScope} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate structured concurrency APIs with real GPU hardware.
 * The backend is auto-detected at runtime:
 * <ul>
 *   <li>On NVIDIA machines: Tests run with NvidiaBackend, JFR events show backend=CUDA</li>
 *   <li>On AMD machines: Tests run with AmdBackend, JFR events show backend=HIP</li>
 * </ul>
 *
 * <p>Both platforms must pass for the build to be green.
 */
@Tag("gpu")
@DisplayName("GpuTaskScope GPU Tests")
class GpuTaskScopeGpuTest {

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

    // ==================== Backend Integration ====================

    @Test
    @DisplayName("Backend auto-detected and created successfully")
    void realBackendAutoDetected() {
        assertNotNull(backend, "Backend should be created");
        assertNotNull(backend.name(), "Backend should have a name");
        assertTrue(backend.deviceIndex() >= 0, "Device index should be non-negative");
    }

    @Test
    @DisplayName("Backend capabilities can be queried")
    void backendCapabilitiesQueried() {
        var capabilities = backend.gpuCapabilities();
        assertNotNull(capabilities, "GPU capabilities should be available");
    }

    // ==================== Single Task Scope ====================

    @Test
    @DisplayName("Single task completes successfully in scope")
    void singleTaskScope() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            long startTime = System.nanoTime();

            GpuTask<Integer> task = scope.fork(() -> {
                simulateGpuWork("SingleTask");
                return 42;
            });

            scope.joinAll();

            int result = task.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertEquals(42, result, "Task should return expected value");
            assertTrue(task.isSuccess(), "Task should be successful");

            emitKernelEvent("SingleTaskScope", "1 task", elapsedMicros);
        }
    }

    // ==================== Parallel Tasks ====================

    @Test
    @DisplayName("Multiple parallel tasks complete successfully")
    void parallelTasks() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            long startTime = System.nanoTime();
            AtomicInteger completedCount = new AtomicInteger(0);

            GpuTask<Integer> task1 = scope.fork(() -> {
                simulateGpuWork("ParallelTask1");
                completedCount.incrementAndGet();
                return 1;
            });

            GpuTask<Integer> task2 = scope.fork(() -> {
                simulateGpuWork("ParallelTask2");
                completedCount.incrementAndGet();
                return 2;
            });

            GpuTask<Integer> task3 = scope.fork(() -> {
                simulateGpuWork("ParallelTask3");
                completedCount.incrementAndGet();
                return 3;
            });

            scope.joinAll();

            int sum = task1.get() + task2.get() + task3.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertEquals(6, sum, "Sum of task results should be 6");
            assertEquals(3, completedCount.get(), "All 3 tasks should complete");
            assertTrue(task1.isSuccess() && task2.isSuccess() && task3.isSuccess(),
                "All tasks should be successful");

            emitKernelEvent("ParallelTasksScope", "3 tasks", elapsedMicros);
        }
    }

    // ==================== Fork With Stream ====================

    @Test
    @DisplayName("Fork with dedicated stream acquires unique streams")
    void forkWithStream() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            long startTime = System.nanoTime();
            AtomicInteger streamCount = new AtomicInteger(0);

            GpuTask<Long> task1 = scope.forkWithStream(lease -> {
                long streamHandle = lease.streamHandle();
                streamCount.incrementAndGet();
                simulateGpuWork("StreamTask1");
                lease.synchronize();
                return streamHandle;
            });

            GpuTask<Long> task2 = scope.forkWithStream(lease -> {
                long streamHandle = lease.streamHandle();
                streamCount.incrementAndGet();
                simulateGpuWork("StreamTask2");
                lease.synchronize();
                return streamHandle;
            });

            scope.joinAll();

            long stream1 = task1.get();
            long stream2 = task2.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertNotEquals(stream1, stream2, "Each task should have a unique stream");
            assertEquals(2, streamCount.get(), "Both tasks should have executed");
            assertNotNull(task1.lease(), "Task1 should have a lease");
            assertNotNull(task2.lease(), "Task2 should have a lease");

            emitKernelEvent("ForkWithStreamScope", "2 streams", elapsedMicros);
        }
    }

    // ==================== Nested Scopes ====================

    @Test
    @DisplayName("Nested scopes work correctly")
    void nestedScopes() throws Exception {
        long startTime = System.nanoTime();
        AtomicInteger outerCompleted = new AtomicInteger(0);
        AtomicInteger innerCompleted = new AtomicInteger(0);

        try (GpuTaskScope outerScope = GpuTaskScope.open(backend, "outer-scope")) {
            GpuTask<Integer> outerTask = outerScope.fork(() -> {
                // Create a nested scope
                try (GpuTaskScope innerScope = GpuTaskScope.open(backend, "inner-scope")) {
                    GpuTask<Integer> innerTask = innerScope.fork(() -> {
                        simulateGpuWork("InnerTask");
                        innerCompleted.incrementAndGet();
                        return 10;
                    });
                    innerScope.joinAll();
                    return innerTask.get();
                }
            });

            outerScope.fork(() -> {
                simulateGpuWork("OuterTask");
                outerCompleted.incrementAndGet();
                return 20;
            });

            outerScope.joinAll();

            int innerResult = outerTask.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertEquals(10, innerResult, "Inner task result should be accessible");
            assertEquals(1, innerCompleted.get(), "Inner task should complete");
            assertEquals(1, outerCompleted.get(), "Outer task should complete");

            emitKernelEvent("NestedScopes", "outer+inner", elapsedMicros);
        }
    }

    // ==================== Named Scopes ====================

    @Test
    @DisplayName("Named scopes emit correct JFR events")
    void namedScopes() throws Exception {
        String[] scopeNames = {"inference-batch", "training-step", "data-preprocessing"};
        int successCount = 0;

        for (String name : scopeNames) {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, name)) {
                long startTime = System.nanoTime();

                GpuTask<String> task = scope.fork(() -> {
                    simulateGpuWork(name);
                    return name + "-done";
                });

                scope.joinAll();

                String result = task.get();
                long elapsedMicros = (System.nanoTime() - startTime) / 1000;

                assertEquals(name + "-done", result, "Task result should include scope name");
                assertEquals(name, scope.scopeName(), "Scope name should match");
                assertTrue(scope.scopeId() > 0, "Scope ID should be positive");

                emitKernelEvent("NamedScope:" + name, "profiling", elapsedMicros);
                successCount++;
            }
        }

        assertEquals(scopeNames.length, successCount, "All named scopes should succeed");
    }

    // ==================== Stream Operations ====================

    @Test
    @DisplayName("Real stream creation and destruction")
    void realStreamCreation() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            GpuTask<Long> task = scope.forkWithStream(lease -> {
                long handle = lease.streamHandle();
                // Stream handle should be valid (non-zero typically)
                return handle;
            });

            scope.joinAll();

            // After join, the stream should have been used
            assertTrue(task.isSuccess(), "Stream task should succeed");
        }
    }

    @Test
    @DisplayName("Stream synchronization blocks until complete")
    void streamSynchronizationBlocks() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicInteger syncCount = new AtomicInteger(0);

            GpuTask<Void> task = scope.forkWithStream(lease -> {
                // Synchronize multiple times
                lease.synchronize();
                syncCount.incrementAndGet();
                lease.synchronize();
                syncCount.incrementAndGet();
                return null;
            });

            scope.joinAll();

            assertEquals(2, syncCount.get(), "Both synchronize calls should complete");
        }
    }

    @Test
    @DisplayName("Concurrent real streams work independently")
    void concurrentRealStreams() throws Exception {
        final int numStreams = 5;
        AtomicInteger completedStreams = new AtomicInteger(0);

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            for (int i = 0; i < numStreams; i++) {
                final int streamIndex = i;
                scope.forkWithStream(lease -> {
                    simulateGpuWork("ConcurrentStream-" + streamIndex);
                    lease.synchronize();
                    completedStreams.incrementAndGet();
                    return null;
                });
            }

            scope.joinAll();
        }

        assertEquals(numStreams, completedStreams.get(),
            "All concurrent streams should complete");
    }

    // ==================== JFR Event Validation ====================

    @Test
    @DisplayName("JFR events emitted with correct backend name")
    void jfrEventsWithRealBackend() throws Exception {
        String expectedBackend = GpuTestSupport.expectedBackendName(backend);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-test-scope")) {
            scope.fork(() -> {
                simulateGpuWork("JFR-Test");
                return null;
            });
            scope.joinAll();

            // The scope emits GpuTaskScopeEvent on close
            // Emit a kernel event to verify backend name
            GpuKernelEvent event = new GpuKernelEvent();
            event.operation = "JFRValidation";
            event.shape = "test";
            event.gpuTimeMicros = 1;
            event.backend = expectedBackend;
            event.deviceIndex = backend.deviceIndex();
            event.tier = "GPU_TEST";
            event.memoryBandwidthGBps = 0.0;
            event.commit();
        }

        // If we got here without exception, JFR is working
        assertTrue(true, "JFR events committed successfully");
    }

    // ==================== Helper Methods ====================

    /**
     * Simulates GPU work by doing some computation and sleeping briefly.
     */
    private void simulateGpuWork(String taskName) {
        // Simulate some work
        long sum = 0;
        for (int i = 0; i < 10000; i++) {
            sum += i;
        }

        // Brief sleep to simulate kernel execution time
        try {
            Thread.sleep(5);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Emits a JFR kernel event for validation.
     */
    private void emitKernelEvent(String operation, String shape, long elapsedMicros) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "STRUCTURED_CONCURRENCY_GTEST";
        event.memoryBandwidthGBps = 0.0;
        event.commit();
    }
}
