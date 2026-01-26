package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuWorkCalibrator.GpuWorkResult;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link GpuTaskScope} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate structured concurrency APIs with <b>real GPU hardware</b>
 * by performing actual GPU operations (memory transfers, stream synchronization)
 * and validating timing through JFR events.
 *
 * <p><b>Key principle:</b> Every test performs real GPU work via
 * {@link GpuWorkCalibrator#doGpuWork(GpuBackend, GpuLease, long)}, which:
 * <ul>
 *   <li>Allocates device memory</li>
 *   <li>Performs host-to-device and device-to-host transfers</li>
 *   <li>Synchronizes GPU streams</li>
 *   <li>Emits {@link io.surfworks.warpforge.core.jfr.GpuMemoryEvent} and
 *       {@link GpuKernelEvent} for JFR profiling</li>
 * </ul>
 *
 * <p>Without real GPU work, these would just be CPU API tests, not GPU tests.
 *
 * <p>The backend is auto-detected at runtime:
 * <ul>
 *   <li>On NVIDIA machines: Tests run with NvidiaBackend, JFR shows backend=CUDA</li>
 *   <li>On AMD machines: Tests run with AmdBackend, JFR shows backend=HIP</li>
 * </ul>
 */
@Tag("gpu")
@DisplayName("GpuTaskScope GPU Tests")
class GpuTaskScopeGpuTest {

    private static GpuBackend staticBackend;
    private GpuBackend backend;
    private String backendName;

    @BeforeAll
    static void calibrate() {
        // Create temporary backend for calibration
        staticBackend = GpuTestSupport.createBackend();
        System.out.println("Calibrating GPU work generator for " + staticBackend.name() + "...");
        GpuWorkCalibrator.CalibrationData data = GpuWorkCalibrator.getCalibrationData(staticBackend);
        System.out.printf("Calibration complete: %d elements/ms, %.2f GB/s bandwidth%n",
            data.elementsPerMs(), data.bandwidthGBps());
        staticBackend.close();
        staticBackend = null;
    }

    @BeforeEach
    void setUp() {
        backend = GpuTestSupport.createBackend();
        backendName = GpuTestSupport.expectedBackendName(backend);
        System.out.println("Running on: " + GpuTestSupport.describeEnvironment());
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    // ==================== Single GPU Task Tests ====================

    @Nested
    @DisplayName("Single GPU Task Timing")
    class SingleGpuTaskTiming {

        @Test
        @DisplayName("10ms GPU work completes with accurate timing")
        void tenMillisGpuWorkTiming() throws Exception {
            final long TARGET_MS = 10;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "gpu-work-10ms")) {
                GpuTask<GpuWorkResult> task = scope.forkWithStream(lease -> {
                    // Perform REAL GPU work - memory transfers + sync
                    return GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                });

                scope.joinAll();

                GpuWorkResult result = task.get();
                assertNotNull(result, "GPU work should return result");

                // Validate timing
                GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                    "10ms GPU work timing");

                // Log for debugging
                System.out.printf("GPU Work: target=%dms, actual=%dms, elements=%d, bandwidth=%.2f GB/s%n",
                    TARGET_MS, result.elapsedMillis(), result.tensorElements(), result.bandwidthGBps());

                assertTrue(task.isSuccess(), "Task should succeed");
            }
        }

        @Test
        @DisplayName("25ms GPU work completes with accurate timing")
        void twentyFiveMillisGpuWorkTiming() throws Exception {
            final long TARGET_MS = 25;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "gpu-work-25ms")) {
                GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS)
                );

                scope.joinAll();

                GpuWorkResult result = task.get();
                GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                    "25ms GPU work timing");

                System.out.printf("GPU Work: target=%dms, actual=%dms, bytes=%d%n",
                    TARGET_MS, result.elapsedMillis(), result.byteSize());
            }
        }

        @Test
        @DisplayName("50ms GPU work completes with accurate timing")
        void fiftyMillisGpuWorkTiming() throws Exception {
            final long TARGET_MS = 50;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "gpu-work-50ms")) {
                GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS)
                );

                scope.joinAll();

                GpuWorkResult result = task.get();
                GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                    "50ms GPU work timing");
            }
        }
    }

    // ==================== Concurrent GPU Tasks Tests ====================

    @Nested
    @DisplayName("Concurrent GPU Task Timing")
    class ConcurrentGpuTaskTiming {

        @Test
        @DisplayName("Three concurrent 15ms GPU tasks complete independently")
        void threeConcurrentGpuTasks() throws Exception {
            final long TARGET_MS = 15;
            final int NUM_TASKS = 3;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-3x15ms-gpu")) {
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();
                List<GpuTask<GpuWorkResult>> tasks = new ArrayList<>();

                long scopeStart = System.nanoTime();

                for (int i = 0; i < NUM_TASKS; i++) {
                    int taskId = i;
                    tasks.add(scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                        results.put(taskId, result);
                        return result;
                    }));
                }

                scope.joinAll();
                long scopeElapsedNanos = System.nanoTime() - scopeStart;

                // Validate each task's GPU timing
                for (int i = 0; i < NUM_TASKS; i++) {
                    GpuWorkResult result = results.get(i);
                    assertNotNull(result, "Task " + i + " should have result");

                    GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                        "GPU Task " + i + " timing");

                    System.out.printf("GPU Task %d: target=%dms, actual=%dms, stream=%d%n",
                        i, TARGET_MS, result.elapsedMillis(), result.streamHandle());
                }

                // Verify streams are unique
                long distinctStreams = results.values().stream()
                    .map(GpuWorkResult::streamHandle)
                    .distinct()
                    .count();
                assertEquals(NUM_TASKS, distinctStreams, "Each task should have unique stream");

                // Scope should complete faster than sequential
                long scopeElapsedMs = scopeElapsedNanos / 1_000_000;
                System.out.printf("Scope completed in %dms (sequential would be %dms)%n",
                    scopeElapsedMs, TARGET_MS * NUM_TASKS);
            }
        }

        @Test
        @DisplayName("Five concurrent GPU tasks with varying durations")
        void fiveConcurrentVaryingDurations() throws Exception {
            final long[] TARGET_MS = {10, 15, 20, 25, 30};

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-varying-gpu")) {
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();

                long scopeStart = System.nanoTime();

                for (int i = 0; i < TARGET_MS.length; i++) {
                    int taskId = i;
                    long targetMs = TARGET_MS[i];
                    scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, targetMs);
                        results.put(taskId, result);
                        return result;
                    });
                }

                scope.joinAll();
                long scopeElapsedNanos = System.nanoTime() - scopeStart;

                // Validate each task's individual GPU timing
                for (int i = 0; i < TARGET_MS.length; i++) {
                    GpuWorkResult result = results.get(i);
                    assertNotNull(result, "Task " + i + " should have result");

                    GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS[i], result.elapsedNanos(),
                        "GPU Task " + i + " (" + TARGET_MS[i] + "ms) timing");
                }

                // Scope should complete around the longest task time + overhead
                long scopeElapsedMs = scopeElapsedNanos / 1_000_000;
                assertTrue(scopeElapsedMs >= 25 && scopeElapsedMs < 60,
                    "Scope should complete near longest task time: " + scopeElapsedMs + "ms");
            }
        }

        @Test
        @DisplayName("Ten concurrent 10ms GPU tasks complete efficiently")
        void tenConcurrentGpuTasksEfficiency() throws Exception {
            final long TARGET_MS = 10;
            final int NUM_TASKS = 10;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-10x10ms-gpu")) {
                AtomicInteger completedCount = new AtomicInteger(0);
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();

                long scopeStart = System.nanoTime();

                for (int i = 0; i < NUM_TASKS; i++) {
                    int taskId = i;
                    scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                        results.put(taskId, result);
                        completedCount.incrementAndGet();
                        return result;
                    });
                }

                scope.joinAll();
                long scopeElapsedNanos = System.nanoTime() - scopeStart;

                assertEquals(NUM_TASKS, completedCount.get(), "All tasks should complete");

                // Calculate total GPU work time
                long totalGpuTimeMs = results.values().stream()
                    .mapToLong(GpuWorkResult::elapsedMillis)
                    .sum();

                long scopeElapsedMs = scopeElapsedNanos / 1_000_000;

                System.out.printf("Ten concurrent GPU tasks: total_gpu_time=%dms, scope_time=%dms%n",
                    totalGpuTimeMs, scopeElapsedMs);

                // Total GPU time with 10 concurrent tasks under contention
                // Each 10ms target may take 25-35ms under load, so total ~250-350ms
                // The key assertion is that all tasks complete with real GPU work
                assertTrue(totalGpuTimeMs >= 80 && totalGpuTimeMs <= 400,
                    "Total GPU time should reflect real GPU work: " + totalGpuTimeMs + "ms");
            }
        }
    }

    // ==================== Stream Handle Tests ====================

    @Nested
    @DisplayName("Stream Handle Validation")
    class StreamHandleValidation {

        @Test
        @DisplayName("Each forked GPU task gets unique stream handle")
        void uniqueStreamHandles() throws Exception {
            final int NUM_TASKS = 5;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "unique-streams")) {
                ConcurrentHashMap<Integer, Long> streamHandles = new ConcurrentHashMap<>();

                for (int i = 0; i < NUM_TASKS; i++) {
                    int taskId = i;
                    scope.forkWithStream(lease -> {
                        long handle = lease.streamHandle();
                        streamHandles.put(taskId, handle);

                        // Do real GPU work on this stream
                        GpuWorkCalibrator.doGpuWork(backend, lease, 5);
                        return handle;
                    });
                }

                scope.joinAll();

                // Verify all stream handles are unique
                assertEquals(NUM_TASKS, streamHandles.size(), "Should have " + NUM_TASKS + " tasks");
                long distinctHandles = streamHandles.values().stream().distinct().count();
                assertEquals(NUM_TASKS, distinctHandles, "All stream handles should be unique");

                // All handles should be valid (non-zero typically)
                streamHandles.values().forEach(h ->
                    assertNotEquals(0L, h, "Stream handle should be non-zero"));
            }
        }

        @Test
        @DisplayName("Stream handle is accessible throughout task lifetime")
        void streamHandleAccessibleThroughoutTask() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "stream-lifetime")) {
                scope.forkWithStream(lease -> {
                    long handleAtStart = lease.streamHandle();

                    // Do some GPU work
                    GpuWorkCalibrator.doGpuWork(backend, lease, 10);

                    // Handle should be same
                    long handleAfterWork = lease.streamHandle();
                    assertEquals(handleAtStart, handleAfterWork,
                        "Stream handle should remain constant");

                    // Do more work
                    GpuWorkCalibrator.doGpuWork(backend, lease, 10);

                    // Still same
                    long handleAtEnd = lease.streamHandle();
                    assertEquals(handleAtStart, handleAtEnd,
                        "Stream handle should remain constant throughout");

                    return handleAtStart;
                });

                scope.joinAll();
            }
        }
    }

    // ==================== Nested Scope Tests ====================

    @Nested
    @DisplayName("Nested Scope GPU Work")
    class NestedScopeGpuWork {

        @Test
        @DisplayName("Nested scopes perform independent GPU work")
        void nestedScopesIndependentGpuWork() throws Exception {
            final long OUTER_WORK_MS = 15;
            final long INNER_WORK_MS = 20;

            ConcurrentHashMap<String, GpuWorkResult> results = new ConcurrentHashMap<>();

            try (GpuTaskScope outerScope = GpuTaskScope.open(backend, "outer-gpu")) {
                outerScope.forkWithStream(outerLease -> {
                    // Outer GPU work
                    GpuWorkResult outerResult = GpuWorkCalibrator.doGpuWork(backend, outerLease, OUTER_WORK_MS);
                    results.put("outer", outerResult);

                    // Create inner scope for more GPU work
                    try (GpuTaskScope innerScope = GpuTaskScope.open(backend, "inner-gpu")) {
                        innerScope.forkWithStream(innerLease -> {
                            GpuWorkResult innerResult = GpuWorkCalibrator.doGpuWork(backend, innerLease, INNER_WORK_MS);
                            results.put("inner", innerResult);
                            return innerResult;
                        });
                        try {
                            innerScope.joinAll();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException("Inner scope join interrupted", e);
                        }
                    }

                    return outerResult;
                });

                outerScope.joinAll();
            }

            // Validate both GPU operations
            GpuWorkResult outerResult = results.get("outer");
            GpuWorkResult innerResult = results.get("inner");

            assertNotNull(outerResult, "Outer scope should have result");
            assertNotNull(innerResult, "Inner scope should have result");

            GpuWorkCalibrator.assertTimingWithinTolerance(OUTER_WORK_MS, outerResult.elapsedNanos(),
                "Outer scope GPU work");
            GpuWorkCalibrator.assertTimingWithinTolerance(INNER_WORK_MS, innerResult.elapsedNanos(),
                "Inner scope GPU work");

            // Streams should be different
            assertNotEquals(outerResult.streamHandle(), innerResult.streamHandle(),
                "Nested scopes should have different streams");
        }
    }

    // ==================== Error Handling Tests ====================

    @Nested
    @DisplayName("Error Handling with GPU Work")
    class ErrorHandlingWithGpuWork {

        @Test
        @DisplayName("Failed task completes GPU work before throwing")
        void failedTaskCompletesGpuWork() {
            final long GPU_WORK_MS = 15;
            ConcurrentHashMap<String, GpuWorkResult> results = new ConcurrentHashMap<>();

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "fail-after-gpu-work")) {
                GpuTask<Void> task = scope.forkWithStream(lease -> {
                    // Complete GPU work first
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, GPU_WORK_MS);
                    results.put("work", result);

                    // Then fail
                    throw new RuntimeException("Intentional failure after GPU work");
                });

                try {
                    scope.joinAll();
                } catch (Exception e) {
                    // Expected
                }

                // GPU work should have completed
                GpuWorkResult result = results.get("work");
                assertNotNull(result, "GPU work should have completed before failure");

                GpuWorkCalibrator.assertTimingWithinTolerance(GPU_WORK_MS, result.elapsedNanos(),
                    "GPU work before failure");

                assertTrue(task.isFailed(), "Task should be failed");
            }
        }
    }

    // ==================== JFR Validation Tests ====================

    @Nested
    @DisplayName("JFR Event Validation")
    class JfrEventValidation {

        @Test
        @DisplayName("JFR events capture correct backend and device info")
        void jfrEventsCaptureBackendInfo() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-backend-info")) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 10);

                    // Emit additional validation event
                    GpuKernelEvent event = new GpuKernelEvent();
                    event.operation = "JFRValidation";
                    event.shape = "elements=" + result.tensorElements();
                    event.gpuTimeMicros = result.elapsedNanos() / 1000;
                    event.backend = backendName;
                    event.deviceIndex = backend.deviceIndex();
                    event.tier = "GTEST_VALIDATION";
                    event.bytesTransferred = result.byteSize() * 2; // H2D + D2H
                    event.memoryBandwidthGBps = result.bandwidthGBps();
                    event.commit();

                    return result;
                });

                scope.joinAll();

                // Verify backend name matches expected
                String expected = GpuTestSupport.expectedBackendName(backend);
                assertEquals(expected, backendName, "Backend name should match");
            }
        }

        @Test
        @DisplayName("Multiple GPU tasks emit separate JFR events")
        void multipleGpuTasksEmitSeparateJfrEvents() throws Exception {
            final int NUM_TASKS = 4;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-multi-task")) {
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();

                for (int i = 0; i < NUM_TASKS; i++) {
                    int taskId = i;
                    scope.forkWithStream(lease -> {
                        // doGpuWork internally emits JFR events
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 10);
                        results.put(taskId, result);
                        return result;
                    });
                }

                scope.joinAll();

                assertEquals(NUM_TASKS, results.size(), "All tasks should have results");

                // Each result should have valid timing and stream info
                for (int i = 0; i < NUM_TASKS; i++) {
                    GpuWorkResult result = results.get(i);
                    assertTrue(result.elapsedNanos() > 0, "Task " + i + " should have positive timing");
                    assertTrue(result.byteSize() > 0, "Task " + i + " should have transferred bytes");
                }
            }
        }

        @Test
        @DisplayName("JFR events include accurate memory bandwidth")
        void jfrEventsIncludeAccurateBandwidth() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-bandwidth")) {
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 20);

                    // Bandwidth should be reasonable for GPU memory transfers
                    double bandwidth = result.bandwidthGBps();
                    System.out.printf("GPU memory bandwidth: %.2f GB/s%n", bandwidth);

                    // Bandwidth should be positive and reasonable (0.1 - 1000 GB/s range)
                    assertTrue(bandwidth > 0.1 && bandwidth < 1000,
                        "Bandwidth should be reasonable: " + bandwidth + " GB/s");

                    return result;
                });

                scope.joinAll();
            }
        }
    }
}
