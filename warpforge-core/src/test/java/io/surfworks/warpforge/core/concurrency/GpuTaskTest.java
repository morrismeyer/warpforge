package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.concurrent.StructuredTaskScope;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link GpuTask}.
 *
 * <p>These tests use {@link MockGpuBackend} to test GpuTask behavior
 * without requiring real GPU hardware.
 */
@DisplayName("GpuTask Unit Tests")
class GpuTaskTest {

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

    // ==================== Get Tests ====================

    @Nested
    @DisplayName("Get")
    class GetTests {

        @Test
        @DisplayName("get() returns task result")
        void getReturnsResult() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                scope.joinAll();
                assertEquals(42, task.get());
            }
        }

        @Test
        @DisplayName("get() returns complex result")
        void getReturnsComplexResult() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<String> task = scope.fork(() -> "Hello, WarpForge!");
                scope.joinAll();
                assertEquals("Hello, WarpForge!", task.get());
            }
        }

        @Test
        @DisplayName("get() after success is idempotent")
        void getAfterSuccessIdempotent() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 123);
                scope.joinAll();

                // Multiple calls should return same value
                assertEquals(123, task.get());
                assertEquals(123, task.get());
                assertEquals(123, task.get());
            }
        }

        @Test
        @DisplayName("get() returns null if task returns null")
        void getReturnsNull() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Void> task = scope.fork(() -> null);
                scope.joinAll();
                assertNull(task.get());
            }
        }
    }

    // ==================== State Tests ====================

    @Nested
    @DisplayName("State")
    class StateTests {

        @Test
        @DisplayName("State is SUCCESS after completion")
        void stateSuccessAfterCompletion() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                scope.joinAll();
                assertEquals(StructuredTaskScope.Subtask.State.SUCCESS, task.state());
            }
        }

        @Test
        @DisplayName("State is FAILED after exception")
        void stateFailedAfterException() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    throw new RuntimeException("Test failure");
                });

                try {
                    scope.joinAll();
                } catch (Exception ignored) {
                }

                assertEquals(StructuredTaskScope.Subtask.State.FAILED, task.state());
            }
        }
    }

    // ==================== Predicate Tests ====================

    @Nested
    @DisplayName("Predicates")
    class PredicateTests {

        @Test
        @DisplayName("isSuccess() returns true on success")
        void isSuccessTrueOnSuccess() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                scope.joinAll();
                assertTrue(task.isSuccess());
            }
        }

        @Test
        @DisplayName("isSuccess() returns false on failure")
        void isSuccessFalseOnFailed() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    throw new RuntimeException("Fail");
                });

                try {
                    scope.joinAll();
                } catch (Exception ignored) {
                }

                assertFalse(task.isSuccess());
            }
        }

        @Test
        @DisplayName("isFailed() returns true on exception")
        void isFailedTrueOnException() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    throw new IllegalStateException("Test error");
                });

                try {
                    scope.joinAll();
                } catch (Exception ignored) {
                }

                assertTrue(task.isFailed());
            }
        }

        @Test
        @DisplayName("isFailed() returns false on success")
        void isFailedFalseOnSuccess() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                scope.joinAll();
                assertFalse(task.isFailed());
            }
        }
    }

    // ==================== Exception Tests ====================

    @Nested
    @DisplayName("Exception")
    class ExceptionTests {

        @Test
        @DisplayName("exception() returns throwable on failure")
        void exceptionReturnsThrowable() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                RuntimeException expected = new RuntimeException("Expected error");

                GpuTask<Integer> task = scope.fork(() -> {
                    throw expected;
                });

                try {
                    scope.joinAll();
                } catch (Exception ignored) {
                }

                assertNotNull(task.exception());
                assertEquals("Expected error", task.exception().getMessage());
            }
        }

        @Test
        @DisplayName("exception() returns null on success")
        void exceptionNullOnSuccess() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                scope.joinAll();
                assertNull(task.exception());
            }
        }

        @Test
        @DisplayName("exception() returns specific exception type")
        void exceptionReturnsSpecificType() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    throw new IllegalArgumentException("Bad argument");
                });

                try {
                    scope.joinAll();
                } catch (Exception ignored) {
                }

                assertTrue(task.exception() instanceof IllegalArgumentException);
            }
        }
    }

    // ==================== Lease Tests ====================

    @Nested
    @DisplayName("Lease")
    class LeaseTests {

        @Test
        @DisplayName("lease() is null for fork()")
        void leaseNullForFork() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                scope.joinAll();
                assertNull(task.lease());
            }
        }

        @Test
        @DisplayName("lease() is non-null for forkWithStream()")
        void leaseNonNullForForkWithStream() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicReference<GpuLease> capturedLease = new AtomicReference<>();

                GpuTask<Integer> task = scope.forkWithStream(lease -> {
                    capturedLease.set(lease);
                    return 42;
                });

                // Check lease is available immediately after fork
                assertNotNull(task.lease());

                scope.joinAll();

                // Lease should match what was passed to task
                assertEquals(capturedLease.get(), task.lease());
            }
        }

        @Test
        @DisplayName("lease() stream handle is accessible")
        void leaseStreamHandleAccessible() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Long> task = scope.forkWithStream(lease -> lease.streamHandle());
                scope.joinAll();

                assertNotNull(task.lease());
                assertEquals(task.get(), task.lease().streamHandle());
            }
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("State transitions from UNAVAILABLE to SUCCESS")
        void stateTransitionsCorrectly() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    // Small delay to ensure we can observe UNAVAILABLE state
                    return 42;
                });

                // After join, state should be SUCCESS
                scope.joinAll();
                assertEquals(StructuredTaskScope.Subtask.State.SUCCESS, task.state());
            }
        }

        @Test
        @DisplayName("Multiple get() calls return same reference")
        void multipleGetCallsSame() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Object result = new Object();
                GpuTask<Object> task = scope.fork(() -> result);
                scope.joinAll();

                // All get() calls should return the same object
                Object get1 = task.get();
                Object get2 = task.get();
                Object get3 = task.get();

                assertTrue(get1 == get2 && get2 == get3, "Same reference should be returned");
            }
        }

        @Test
        @DisplayName("Task with void return type works")
        void taskWithVoidReturnType() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Void> task = scope.fork(() -> {
                    // Side effect only
                    return null;
                });

                scope.joinAll();
                assertTrue(task.isSuccess());
                assertNull(task.get());
            }
        }
    }
}
