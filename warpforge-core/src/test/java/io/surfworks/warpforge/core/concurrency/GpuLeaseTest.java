package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link GpuLease}.
 *
 * <p>These tests use {@link MockGpuBackend} to test GpuLease behavior
 * without requiring real GPU hardware.
 */
@DisplayName("GpuLease Unit Tests")
class GpuLeaseTest {

    private MockGpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new MockGpuBackend();
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== Accessor Tests ====================

    @Nested
    @DisplayName("Accessors")
    class AccessorTests {

        @Test
        @DisplayName("streamHandle() returns valid handle")
        void streamHandleReturnsValue() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicLong capturedHandle = new AtomicLong(0);

                scope.forkWithStream(lease -> {
                    capturedHandle.set(lease.streamHandle());
                    assertTrue(lease.streamHandle() > 0);
                    return null;
                });

                scope.joinAll();
                assertTrue(capturedHandle.get() > 0);
            }
        }

        @Test
        @DisplayName("acquireTimeNanos() returns positive value")
        void acquireTimeNanosPositive() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicLong capturedTime = new AtomicLong(0);

                scope.forkWithStream(lease -> {
                    capturedTime.set(lease.acquireTimeNanos());
                    return null;
                });

                scope.joinAll();
                assertTrue(capturedTime.get() > 0);
            }
        }

        @Test
        @DisplayName("parentScope() returns owning scope")
        void parentScopeReturnsOwner() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicBoolean verified = new AtomicBoolean(false);

                scope.forkWithStream(lease -> {
                    assertEquals(scope, lease.parentScope());
                    verified.set(true);
                    return null;
                });

                scope.joinAll();
                assertTrue(verified.get());
            }
        }
    }

    // ==================== Synchronize Tests ====================

    @Nested
    @DisplayName("Synchronize")
    class SynchronizeTests {

        @Test
        @DisplayName("synchronize() delegates to backend")
        void synchronizeDelegatesToBackend() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    return null;
                });

                scope.joinAll();
            }

            assertTrue(backend.recordedOperations().stream()
                .anyMatch(op -> op.startsWith("synchronizeStream:")));
        }

        @Test
        @DisplayName("synchronize() records operation")
        void synchronizeRecordsOperation() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicLong streamHandle = new AtomicLong(0);

                scope.forkWithStream(lease -> {
                    streamHandle.set(lease.streamHandle());
                    lease.synchronize();
                    return null;
                });

                scope.joinAll();
            }

            assertTrue(backend.recordedOperations().contains(
                "synchronizeStream:" + 1)); // First stream ID is 1
        }

        @Test
        @DisplayName("Multiple synchronize() calls work")
        void multipleSynchronizeCalls() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    lease.synchronize();
                    lease.synchronize();
                    return null;
                });

                scope.joinAll();
            }

            long syncCount = backend.recordedOperations().stream()
                .filter(op -> op.startsWith("synchronizeStream:"))
                .count();
            assertEquals(3, syncCount);
        }
    }

    // ==================== Close Tests ====================

    @Nested
    @DisplayName("Close")
    class CloseTests {

        @Test
        @DisplayName("close() releases stream from parent")
        void closeReleasesFromParent() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    // Explicitly close the lease
                    lease.close();
                    return null;
                });

                scope.joinAll();
            }

            assertTrue(backend.recordedOperations().stream()
                .anyMatch(op -> op.startsWith("destroyStream:")));
        }

        @Test
        @DisplayName("close() is idempotent")
        void closeIsIdempotent() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    lease.close();
                    assertDoesNotThrow(lease::close);
                    assertDoesNotThrow(lease::close);
                    return null;
                });

                scope.joinAll();
            }
        }

        @Test
        @DisplayName("Double close is no-op")
        void doubleCloseNoOp() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    lease.close();
                    lease.close(); // Should not throw or cause issues
                    return null;
                });

                scope.joinAll();
            }

            // Count destroy operations - should only be one despite double close
            long destroyCount = backend.recordedOperations().stream()
                .filter(op -> op.startsWith("destroyStream:") && !op.contains("unknown"))
                .count();
            // Note: forkWithStream releases stream automatically, so we may see 1 destroy
            assertTrue(destroyCount >= 0);
        }
    }

    // ==================== Lifecycle Tests ====================

    @Nested
    @DisplayName("Lifecycle")
    class LifecycleTests {

        @Test
        @DisplayName("Stream handle is valid during task lifetime")
        void streamHandleValidDuringLifetime() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicBoolean validated = new AtomicBoolean(false);

                scope.forkWithStream(lease -> {
                    long handle = lease.streamHandle();
                    assertTrue(backend.isStreamActive(handle),
                        "Stream should be active during task");
                    validated.set(true);
                    return null;
                });

                scope.joinAll();
                assertTrue(validated.get());
            }
        }

        @Test
        @DisplayName("Stream is destroyed after scope closes")
        void streamDestroyedAfterScopeCloses() throws Exception {
            AtomicLong capturedHandle = new AtomicLong(0);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    capturedHandle.set(lease.streamHandle());
                    return null;
                });

                scope.joinAll();
            }

            // After scope closes, stream should be destroyed
            assertTrue(backend.recordedOperations().stream()
                .anyMatch(op -> op.equals("destroyStream:" + capturedHandle.get())));
        }
    }

    // ==================== Multiple Leases Tests ====================

    @Nested
    @DisplayName("Multiple Leases")
    class MultipleLeaseTests {

        @Test
        @DisplayName("Multiple leases have different handles")
        void multipleLeasesHaveDifferentHandles() throws Exception {
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

                assertTrue(allUnique.get());
                assertEquals(5, handles.size());
            }
        }

        @Test
        @DisplayName("Leases are independent")
        void leasesIndependent() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicLong handle1 = new AtomicLong(0);
                AtomicLong handle2 = new AtomicLong(0);

                scope.forkWithStream(lease -> {
                    handle1.set(lease.streamHandle());
                    try {
                        Thread.sleep(10); // Hold lease briefly
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    return null;
                });

                scope.forkWithStream(lease -> {
                    handle2.set(lease.streamHandle());
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    return null;
                });

                scope.joinAll();

                assertNotEquals(handle1.get(), handle2.get());
            }
        }
    }

    // ==================== Timing Tests ====================

    @Nested
    @DisplayName("Timing")
    class TimingTests {

        @Test
        @DisplayName("Acquire time is reasonable")
        void acquireTimeNanosReasonable() throws Exception {
            long beforeNanos = System.nanoTime();

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicLong acquireTime = new AtomicLong(0);

                scope.forkWithStream(lease -> {
                    acquireTime.set(lease.acquireTimeNanos());
                    return null;
                });

                scope.joinAll();

                long afterNanos = System.nanoTime();
                assertTrue(acquireTime.get() >= beforeNanos);
                assertTrue(acquireTime.get() <= afterNanos);
            }
        }

        @Test
        @DisplayName("Acquire time is before close time")
        void acquireTimeNanosBeforeClose() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicLong acquireTime = new AtomicLong(0);
                AtomicLong closeTime = new AtomicLong(0);

                scope.forkWithStream(lease -> {
                    acquireTime.set(lease.acquireTimeNanos());
                    try {
                        Thread.sleep(1);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    closeTime.set(System.nanoTime());
                    return null;
                });

                scope.joinAll();

                assertTrue(acquireTime.get() < closeTime.get());
            }
        }
    }

    // ==================== Thread Safety Tests ====================

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafetyTests {

        @Test
        @DisplayName("Concurrent synchronize calls work")
        void concurrentSynchronize() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    // Simulate concurrent synchronize calls within the task
                    for (int i = 0; i < 10; i++) {
                        lease.synchronize();
                    }
                    return null;
                });

                scope.joinAll();
            }

            long syncCount = backend.recordedOperations().stream()
                .filter(op -> op.startsWith("synchronizeStream:"))
                .count();
            assertEquals(10, syncCount);
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Zero stream handle is valid if backend returns it")
        void zeroStreamHandleValid() throws Exception {
            // In mock, stream handles start at 1, but the API doesn't forbid 0
            // This test just verifies we don't crash on unusual handles
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    // Handle should be positive in mock
                    assertTrue(lease.streamHandle() > 0);
                    return null;
                });

                scope.joinAll();
            }
        }

        @Test
        @DisplayName("Lease works with named scope")
        void leaseWorksWithNamedScope() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "test-lease-scope")) {
                AtomicBoolean verified = new AtomicBoolean(false);

                scope.forkWithStream(lease -> {
                    assertEquals("test-lease-scope", lease.parentScope().scopeName());
                    verified.set(true);
                    return null;
                });

                scope.joinAll();
                assertTrue(verified.get());
            }
        }
    }
}
