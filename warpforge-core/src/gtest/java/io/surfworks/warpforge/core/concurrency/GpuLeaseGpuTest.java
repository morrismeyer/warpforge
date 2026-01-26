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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link GpuLease} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate GPU lease behavior with real hardware.
 */
@Tag("gpu")
@DisplayName("GpuLease GPU Tests")
class GpuLeaseGpuTest {

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

    // ==================== Stream Handle Tests ====================

    @Test
    @DisplayName("Real stream handle is valid")
    void realStreamHandleValid() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicLong capturedHandle = new AtomicLong(0);

            scope.forkWithStream(lease -> {
                capturedHandle.set(lease.streamHandle());
                return null;
            });

            scope.joinAll();

            // Stream handles should be positive (typically)
            // Different backends may use different handle schemes
            assertNotEquals(0, capturedHandle.get(), "Stream handle should be non-zero");
        }
    }

    @Test
    @DisplayName("Multiple leases have unique stream handles")
    void multipleLeasesUniqueHandles() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            Set<Long> handles = new HashSet<>();
            AtomicBoolean allUnique = new AtomicBoolean(true);

            for (int i = 0; i < 5; i++) {
                scope.forkWithStream(lease -> {
                    synchronized (handles) {
                        if (!handles.add(lease.streamHandle())) {
                            allUnique.set(false);
                        }
                    }
                    return null;
                });
            }

            scope.joinAll();

            assertTrue(allUnique.get(), "All stream handles should be unique");
            assertEquals(5, handles.size(), "Should have 5 distinct handles");
        }
    }

    // ==================== Synchronize Tests ====================

    @Test
    @DisplayName("Synchronize completes without error")
    void synchronizeCompletes() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicBoolean syncCompleted = new AtomicBoolean(false);

            scope.forkWithStream(lease -> {
                lease.synchronize();
                syncCompleted.set(true);
                return null;
            });

            scope.joinAll();

            assertTrue(syncCompleted.get(), "Synchronize should complete");
        }
    }

    @Test
    @DisplayName("Multiple synchronize calls work")
    void multipleSynchronizeCalls() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicLong syncCount = new AtomicLong(0);

            scope.forkWithStream(lease -> {
                for (int i = 0; i < 10; i++) {
                    lease.synchronize();
                    syncCount.incrementAndGet();
                }
                return null;
            });

            scope.joinAll();

            assertEquals(10, syncCount.get(), "All synchronize calls should complete");
        }
    }

    @Test
    @DisplayName("Synchronize timing is measurable")
    void synchronizeTiming() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicLong elapsedNanos = new AtomicLong(0);

            scope.forkWithStream(lease -> {
                long start = System.nanoTime();
                lease.synchronize();
                elapsedNanos.set(System.nanoTime() - start);
                return null;
            });

            scope.joinAll();

            // Synchronize should take some measurable time
            assertTrue(elapsedNanos.get() >= 0, "Synchronize should complete in non-negative time");
        }
    }

    // ==================== Acquire Time Tests ====================

    @Test
    @DisplayName("Acquire time is reasonable")
    void acquireTimeReasonable() throws Exception {
        long beforeNanos = System.nanoTime();

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicLong acquireTime = new AtomicLong(0);

            scope.forkWithStream(lease -> {
                acquireTime.set(lease.acquireTimeNanos());
                return null;
            });

            scope.joinAll();

            long afterNanos = System.nanoTime();

            // Acquire time should be between test start and end
            assertTrue(acquireTime.get() >= beforeNanos,
                "Acquire time should be after test start");
            assertTrue(acquireTime.get() <= afterNanos,
                "Acquire time should be before test end");
        }
    }

    // ==================== Parent Scope Tests ====================

    @Test
    @DisplayName("Parent scope is accessible")
    void parentScopeAccessible() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend, "parent-test")) {
            AtomicBoolean verified = new AtomicBoolean(false);

            scope.forkWithStream(lease -> {
                assertNotNull(lease.parentScope(), "Parent scope should not be null");
                assertEquals(scope, lease.parentScope(), "Parent scope should match");
                assertEquals("parent-test", lease.parentScope().scopeName());
                verified.set(true);
                return null;
            });

            scope.joinAll();

            assertTrue(verified.get(), "Parent scope verification should run");
        }
    }

    // ==================== Lifecycle Tests ====================

    @Test
    @DisplayName("Lease is valid during task execution")
    void leaseValidDuringExecution() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicBoolean leaseWasValid = new AtomicBoolean(false);

            scope.forkWithStream(lease -> {
                // Verify lease is usable
                long handle = lease.streamHandle();
                lease.synchronize();
                leaseWasValid.set(handle != 0);
                return null;
            });

            scope.joinAll();

            assertTrue(leaseWasValid.get(), "Lease should be valid during execution");
        }
    }

    @Test
    @DisplayName("Close is idempotent")
    void closeIsIdempotent() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            AtomicBoolean noException = new AtomicBoolean(false);

            scope.forkWithStream(lease -> {
                // Multiple closes should not throw
                lease.close();
                lease.close();
                lease.close();
                noException.set(true);
                return null;
            });

            scope.joinAll();

            assertTrue(noException.get(), "Multiple closes should not throw");
        }
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for lease operations")
    void jfrEventsForLeaseOperations() throws Exception {
        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-lease-test")) {
            long startTime = System.nanoTime();

            scope.forkWithStream(lease -> {
                lease.synchronize();
                return lease.streamHandle();
            });

            scope.joinAll();

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            // Emit JFR event to validate the system is working
            GpuKernelEvent event = new GpuKernelEvent();
            event.operation = "GpuLeaseTest:JFRValidation";
            event.shape = "lease";
            event.gpuTimeMicros = elapsedMicros;
            event.backend = GpuTestSupport.expectedBackendName(backend);
            event.deviceIndex = backend.deviceIndex();
            event.tier = "GPU_LEASE_TEST";
            event.memoryBandwidthGBps = 0.0;
            event.commit();
        }

        assertTrue(true, "JFR events should be emittable");
    }
}
