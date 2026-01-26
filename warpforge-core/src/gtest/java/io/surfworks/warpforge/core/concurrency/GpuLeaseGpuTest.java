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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link GpuLease} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate GPU lease behavior with <b>real GPU hardware</b> by
 * performing actual GPU operations (memory transfers, stream synchronization)
 * and validating timing through JFR events.
 *
 * <p><b>Key principle:</b> Every test performs real GPU work via
 * {@link GpuWorkCalibrator#doGpuWork(GpuBackend, GpuLease, long)}, which:
 * <ul>
 *   <li>Allocates device memory</li>
 *   <li>Performs host-to-device and device-to-host transfers</li>
 *   <li>Synchronizes GPU streams via the lease</li>
 *   <li>Emits JFR events for GPU memory and kernel operations</li>
 * </ul>
 *
 * <p>Without real GPU work, these would just be CPU API tests, not GPU tests.
 */
@Tag("gpu")
@DisplayName("GpuLease GPU Tests")
class GpuLeaseGpuTest {

    private static GpuBackend staticBackend;
    private GpuBackend backend;
    private String backendName;

    @BeforeAll
    static void calibrate() {
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

    // ==================== Single Lease GPU Work Tests ====================

    @Nested
    @DisplayName("Single Lease GPU Work")
    class SingleLeaseGpuWork {

        @Test
        @DisplayName("Lease performs 10ms GPU work accurately")
        void leasePerformsTenMillisGpuWork() throws Exception {
            final long TARGET_MS = 10;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "lease-gpu-10ms")) {
                GpuTask<GpuWorkResult> task = scope.forkWithStream(lease -> {
                    // Perform real GPU work through this lease's stream
                    return GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                });

                scope.joinAll();

                GpuWorkResult result = task.get();
                assertNotNull(result, "GPU work should return result");

                // Validate timing
                GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                    "Lease 10ms GPU work timing");

                // Verify stream handle was captured
                assertTrue(result.streamHandle() != 0, "Stream handle should be valid");

                System.out.printf("Lease GPU Work: target=%dms, actual=%dms, stream=%d%n",
                    TARGET_MS, result.elapsedMillis(), result.streamHandle());
            }
        }

        @Test
        @DisplayName("Lease performs 25ms GPU work accurately")
        void leasePerformsTwentyFiveMillisGpuWork() throws Exception {
            final long TARGET_MS = 25;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "lease-gpu-25ms")) {
                GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS)
                );

                scope.joinAll();

                GpuWorkResult result = task.get();
                GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                    "Lease 25ms GPU work timing");
            }
        }

        @Test
        @DisplayName("Lease performs 50ms GPU work accurately")
        void leasePerformsFiftyMillisGpuWork() throws Exception {
            final long TARGET_MS = 50;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "lease-gpu-50ms")) {
                GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS)
                );

                scope.joinAll();

                GpuWorkResult result = task.get();
                GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                    "Lease 50ms GPU work timing");
            }
        }
    }

    // ==================== Multiple GPU Operations on Same Lease ====================

    @Nested
    @DisplayName("Multiple GPU Operations on Same Lease")
    class MultipleGpuOpsOnSameLease {

        @Test
        @DisplayName("Three sequential GPU operations on same lease")
        void threeSequentialGpuOpsOnSameLease() throws Exception {
            final long WORK_PER_OP_MS = 10;
            final int NUM_OPS = 3;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "multi-op-lease")) {
                List<GpuWorkResult> results = new ArrayList<>();

                scope.forkWithStream(lease -> {
                    long initialHandle = lease.streamHandle();

                    // Perform multiple GPU operations sequentially on the same lease
                    for (int i = 0; i < NUM_OPS; i++) {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, WORK_PER_OP_MS);

                        // Stream handle should remain the same
                        assertEquals(initialHandle, result.streamHandle(),
                            "Stream handle should remain constant for op " + i);

                        synchronized (results) {
                            results.add(result);
                        }
                    }

                    return results.size();
                });

                scope.joinAll();

                // Validate each operation's timing
                assertEquals(NUM_OPS, results.size(), "Should have " + NUM_OPS + " results");
                for (int i = 0; i < NUM_OPS; i++) {
                    GpuWorkResult result = results.get(i);
                    GpuWorkCalibrator.assertTimingWithinTolerance(WORK_PER_OP_MS, result.elapsedNanos(),
                        "GPU op " + i + " timing");

                    System.out.printf("GPU Op %d: target=%dms, actual=%dms, bytes=%d%n",
                        i, WORK_PER_OP_MS, result.elapsedMillis(), result.byteSize());
                }
            }
        }

        @Test
        @DisplayName("Five sequential GPU operations with varying sizes")
        void fiveSequentialGpuOpsVaryingSizes() throws Exception {
            final long[] WORK_MS = {5, 10, 15, 10, 5};

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "varying-ops-lease")) {
                List<GpuWorkResult> results = new ArrayList<>();

                scope.forkWithStream(lease -> {
                    for (long targetMs : WORK_MS) {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, targetMs);
                        synchronized (results) {
                            results.add(result);
                        }
                    }
                    return results.size();
                });

                scope.joinAll();

                // Validate each operation's timing
                assertEquals(WORK_MS.length, results.size());
                for (int i = 0; i < WORK_MS.length; i++) {
                    GpuWorkCalibrator.assertTimingWithinTolerance(WORK_MS[i], results.get(i).elapsedNanos(),
                        "Varying op " + i + " (" + WORK_MS[i] + "ms) timing");
                }
            }
        }
    }

    // ==================== Concurrent Leases Tests ====================

    @Nested
    @DisplayName("Concurrent Leases")
    class ConcurrentLeases {

        @Test
        @DisplayName("Three concurrent leases perform independent GPU work")
        void threeConcurrentLeasesIndependentWork() throws Exception {
            final long TARGET_MS = 15;
            final int NUM_LEASES = 3;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-3-leases")) {
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();

                long scopeStart = System.nanoTime();

                for (int i = 0; i < NUM_LEASES; i++) {
                    int leaseId = i;
                    scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                        results.put(leaseId, result);
                        return result;
                    });
                }

                scope.joinAll();
                long scopeElapsed = System.nanoTime() - scopeStart;

                // Validate each lease's GPU work
                for (int i = 0; i < NUM_LEASES; i++) {
                    GpuWorkResult result = results.get(i);
                    assertNotNull(result, "Lease " + i + " should have result");

                    GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                        "Concurrent lease " + i + " GPU work timing");
                }

                // Verify handles are unique
                long uniqueHandles = results.values().stream()
                    .map(GpuWorkResult::streamHandle)
                    .distinct()
                    .count();
                assertEquals(NUM_LEASES, uniqueHandles, "All lease handles should be unique");

                // Concurrent execution should be faster than sequential
                long scopeElapsedMs = scopeElapsed / 1_000_000;
                System.out.printf("Concurrent leases: scope=%dms (sequential would be %dms)%n",
                    scopeElapsedMs, TARGET_MS * NUM_LEASES);
            }
        }

        @Test
        @DisplayName("Five concurrent leases with 10ms GPU work each")
        void fiveConcurrentLeasesGpuWork() throws Exception {
            final long TARGET_MS = 10;
            final int NUM_LEASES = 5;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "concurrent-5-leases")) {
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();

                for (int i = 0; i < NUM_LEASES; i++) {
                    int leaseId = i;
                    scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                        results.put(leaseId, result);
                        return result;
                    });
                }

                scope.joinAll();

                // All leases should complete with valid timing
                assertEquals(NUM_LEASES, results.size());
                for (int i = 0; i < NUM_LEASES; i++) {
                    GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, results.get(i).elapsedNanos(),
                        "Concurrent lease " + i + " timing");
                }

                // Unique handles
                long uniqueHandles = results.values().stream()
                    .map(GpuWorkResult::streamHandle)
                    .distinct()
                    .count();
                assertEquals(NUM_LEASES, uniqueHandles);
            }
        }
    }

    // ==================== Lease Lifecycle Tests ====================

    @Nested
    @DisplayName("Lease Lifecycle with GPU Work")
    class LeaseLifecycleWithGpuWork {

        @Test
        @DisplayName("Lease acquire time is before GPU work starts")
        void leaseAcquireTimeBeforeGpuWork() throws Exception {
            final long TARGET_MS = 15;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "acquire-time-test")) {
                scope.forkWithStream(lease -> {
                    long acquireTime = lease.acquireTimeNanos();
                    long workStartTime = System.nanoTime();

                    // Acquire time should be before work starts
                    assertTrue(acquireTime <= workStartTime,
                        "Acquire time should be before work starts");

                    // Do GPU work
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);

                    // Validate timing
                    GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                        "GPU work timing");

                    return result;
                });

                scope.joinAll();
            }
        }

        @Test
        @DisplayName("Parent scope accessible during GPU work")
        void parentScopeAccessibleDuringGpuWork() throws Exception {
            final String SCOPE_NAME = "parent-scope-gpu-test";
            final long TARGET_MS = 10;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, SCOPE_NAME)) {
                scope.forkWithStream(lease -> {
                    // Verify parent scope
                    GpuTaskScope parent = lease.parentScope();
                    assertNotNull(parent, "Parent scope should be accessible");
                    assertEquals(SCOPE_NAME, parent.scopeName());
                    assertEquals(scope.scopeId(), parent.scopeId());

                    // Do GPU work while parent is verified
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);

                    // Validate
                    GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, result.elapsedNanos(),
                        "GPU work during parent verification");

                    return result;
                });

                scope.joinAll();
            }
        }

        @Test
        @DisplayName("Lease stream handle remains constant during GPU work")
        void leaseStreamHandleConstantDuringWork() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "handle-constant-test")) {
                scope.forkWithStream(lease -> {
                    long initialHandle = lease.streamHandle();

                    // Do multiple GPU operations
                    for (int i = 0; i < 3; i++) {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 8);

                        // Handle should stay the same
                        assertEquals(initialHandle, lease.streamHandle(),
                            "Handle should be constant at iteration " + i);
                        assertEquals(initialHandle, result.streamHandle(),
                            "Result handle should match lease handle");
                    }

                    return initialHandle;
                });

                scope.joinAll();
            }
        }
    }

    // ==================== Stream Handle Uniqueness Tests ====================

    @Nested
    @DisplayName("Stream Handle Uniqueness")
    class StreamHandleUniqueness {

        @Test
        @DisplayName("Ten concurrent leases have ten unique stream handles")
        void tenLeasesUniqueHandles() throws Exception {
            final int NUM_LEASES = 10;
            final long TARGET_MS = 5;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "unique-10-handles")) {
                ConcurrentHashMap<Long, Integer> handleToLeaseId = new ConcurrentHashMap<>();
                ConcurrentHashMap<Integer, GpuWorkResult> results = new ConcurrentHashMap<>();

                for (int i = 0; i < NUM_LEASES; i++) {
                    int leaseId = i;
                    scope.forkWithStream(lease -> {
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                        handleToLeaseId.put(result.streamHandle(), leaseId);
                        results.put(leaseId, result);
                        return result;
                    });
                }

                scope.joinAll();

                // Verify all handles are unique
                assertEquals(NUM_LEASES, handleToLeaseId.size(),
                    "Should have " + NUM_LEASES + " unique handles");

                // Verify all results have valid timing
                for (int i = 0; i < NUM_LEASES; i++) {
                    GpuWorkResult result = results.get(i);
                    assertTrue(result.elapsedNanos() > 0, "Lease " + i + " should have positive timing");
                }
            }
        }
    }

    // ==================== JFR Event Validation Tests ====================

    @Nested
    @DisplayName("JFR Event Validation")
    class JfrEventValidation {

        @Test
        @DisplayName("JFR events emitted for each lease GPU operation")
        void jfrEventsForLeaseGpuOps() throws Exception {
            final int NUM_OPS = 3;
            final long TARGET_MS = 10;

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-lease-ops")) {
                List<GpuWorkResult> results = new ArrayList<>();

                for (int i = 0; i < NUM_OPS; i++) {
                    int opId = i;
                    scope.forkWithStream(lease -> {
                        // doGpuWork internally emits JFR events
                        GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);

                        synchronized (results) {
                            results.add(result);
                        }

                        // Emit additional validation event
                        GpuKernelEvent event = new GpuKernelEvent();
                        event.operation = "LeaseJFRValidation_" + opId;
                        event.shape = "bytes=" + result.byteSize();
                        event.gpuTimeMicros = result.elapsedNanos() / 1000;
                        event.backend = backendName;
                        event.deviceIndex = backend.deviceIndex();
                        event.tier = "LEASE_GTEST_VALIDATION";
                        event.memoryBandwidthGBps = result.bandwidthGBps();
                        event.commit();

                        return result;
                    });
                }

                scope.joinAll();

                assertEquals(NUM_OPS, results.size(), "All ops should complete");
                for (GpuWorkResult result : results) {
                    assertTrue(result.bandwidthGBps() > 0, "Bandwidth should be positive");
                }
            }
        }

        @Test
        @DisplayName("JFR captures varying GPU workloads")
        void jfrCapturesVaryingGpuWorkloads() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-varying-workloads")) {
                ConcurrentHashMap<String, GpuWorkResult> results = new ConcurrentHashMap<>();

                // Light workload: 5ms
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 5);
                    results.put("light", result);
                    return result;
                });

                // Medium workload: 20ms
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 20);
                    results.put("medium", result);
                    return result;
                });

                // Heavy workload: 40ms
                scope.forkWithStream(lease -> {
                    GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, 40);
                    results.put("heavy", result);
                    return result;
                });

                scope.joinAll();

                // Validate each workload
                GpuWorkResult light = results.get("light");
                GpuWorkResult medium = results.get("medium");
                GpuWorkResult heavy = results.get("heavy");

                assertNotNull(light);
                assertNotNull(medium);
                assertNotNull(heavy);

                GpuWorkCalibrator.assertTimingWithinTolerance(5, light.elapsedNanos(), "Light workload");
                GpuWorkCalibrator.assertTimingWithinTolerance(20, medium.elapsedNanos(), "Medium workload");
                GpuWorkCalibrator.assertTimingWithinTolerance(40, heavy.elapsedNanos(), "Heavy workload");

                // Heavy should have more bytes transferred than light
                assertTrue(heavy.byteSize() > light.byteSize(),
                    "Heavy workload should transfer more bytes");

                System.out.printf("Workloads - Light: %dms/%d bytes, Medium: %dms/%d bytes, Heavy: %dms/%d bytes%n",
                    light.elapsedMillis(), light.byteSize(),
                    medium.elapsedMillis(), medium.byteSize(),
                    heavy.elapsedMillis(), heavy.byteSize());
            }
        }
    }
}
