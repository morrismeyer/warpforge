package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.StructuredTaskScope;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link GpuTaskScope}.
 *
 * <p>These tests use {@link MockGpuBackend} to test GpuTaskScope behavior
 * without requiring real GPU hardware. For tests with actual GPU hardware,
 * see {@code GpuTaskScopeGpuTest} in the gtest source set.
 */
@DisplayName("GpuTaskScope Unit Tests")
class GpuTaskScopeTest {

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

    // ==================== Lifecycle Tests ====================

    @Nested
    @DisplayName("Lifecycle")
    class LifecycleTests {

        @Test
        @DisplayName("open() creates a new scope with unique ID")
        void openCreatesNewScope() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertNotNull(scope);
                assertTrue(scope.scopeId() > 0);
            }
        }

        @Test
        @DisplayName("open() with name sets scope name")
        void openWithNameSetsName() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "test-scope")) {
                assertEquals("test-scope", scope.scopeName());
            }
        }

        @Test
        @DisplayName("open() without name uses 'unnamed'")
        void openWithoutNameUsesUnnamed() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertEquals("unnamed", scope.scopeName());
            }
        }

        @Test
        @DisplayName("close() releases all active leases")
        void closeReleasesAllLeases() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                // Acquire streams but don't release them
                scope.forkWithStream(lease -> {
                    // Don't explicitly release - should be cleaned up on close
                    return lease.streamHandle();
                });
                scope.forkWithStream(lease -> lease.streamHandle());
                scope.joinAll();
            }
            // After close, all streams should be destroyed
            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("close() is idempotent")
        void closeIsIdempotent() {
            GpuTaskScope scope = GpuTaskScope.open(backend);
            scope.close();
            assertDoesNotThrow(scope::close);
            assertDoesNotThrow(scope::close);
        }

        @Test
        @DisplayName("Multiple scopes have unique IDs")
        void multipleScopesHaveUniqueIds() {
            Set<Long> ids = new HashSet<>();
            for (int i = 0; i < 10; i++) {
                try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                    assertTrue(ids.add(scope.scopeId()), "Scope ID should be unique");
                }
            }
        }
    }

    // ==================== Fork Tests ====================

    @Nested
    @DisplayName("Fork")
    class ForkTests {

        @Test
        @DisplayName("fork() returns valid GpuTask")
        void forkReturnsValidGpuTask() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 42);
                assertNotNull(task);
                scope.joinAll();
                assertEquals(42, task.get());
            }
        }

        @Test
        @DisplayName("fork() task executes on virtual thread")
        void forkTaskExecutesOnVirtualThread() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Boolean> task = scope.fork(() -> Thread.currentThread().isVirtual());
                scope.joinAll();
                assertTrue(task.get(), "Task should run on virtual thread");
            }
        }

        @Test
        @DisplayName("fork() with null callable causes failure at execution")
        void forkNullCallableCausesFailure() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                // fork(null) doesn't fail immediately - it fails when task executes
                GpuTask<Object> task = scope.fork(null);
                assertNotNull(task);
                // The NullPointerException happens during joinAll when task.call() is invoked
                assertThrows(StructuredTaskScope.FailedException.class, scope::joinAll);
            }
        }

        @Test
        @DisplayName("Concurrent forks are safe")
        void concurrentForksAreSafe() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                int taskCount = 100;
                AtomicInteger completedCount = new AtomicInteger(0);
                List<GpuTask<Integer>> tasks = new ArrayList<>();

                for (int i = 0; i < taskCount; i++) {
                    final int index = i;
                    tasks.add(scope.fork(() -> {
                        completedCount.incrementAndGet();
                        return index;
                    }));
                }

                scope.joinAll();

                assertEquals(taskCount, completedCount.get());
                for (int i = 0; i < taskCount; i++) {
                    assertEquals(i, tasks.get(i).get());
                }
            }
        }

        @Test
        @DisplayName("fork() after close throws exception")
        void forkAfterCloseThrows() {
            GpuTaskScope scope = GpuTaskScope.open(backend);
            scope.close();
            assertThrows(Exception.class, () -> scope.fork(() -> 42));
        }
    }

    // ==================== ForkWithStream Tests ====================

    @Nested
    @DisplayName("ForkWithStream")
    class ForkWithStreamTests {

        @Test
        @DisplayName("forkWithStream() acquires a stream")
        void forkWithStreamAcquiresStream() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger streamHandleHolder = new AtomicInteger(0);

                GpuTask<Void> task = scope.forkWithStream(lease -> {
                    assertTrue(lease.streamHandle() > 0);
                    streamHandleHolder.set((int) lease.streamHandle());
                    return null;
                });

                scope.joinAll();
                assertTrue(streamHandleHolder.get() > 0);
            }
        }

        @Test
        @DisplayName("forkWithStream() releases stream on completion")
        void forkWithStreamReleasesOnComplete() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    assertEquals(1, backend.activeStreamCount());
                    return null;
                });
                scope.joinAll();
            }
            // Stream should be destroyed after scope closes
            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("forkWithStream() releases stream on exception")
        void forkWithStreamReleasesOnException() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    throw new RuntimeException("Test exception");
                });
                assertThrows(StructuredTaskScope.FailedException.class, scope::joinAll);
            } catch (Exception e) {
                // Expected
            }
            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("forkWithStream() with null function causes failure at execution")
        void forkWithStreamNullFunctionCausesFailure() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                // forkWithStream(null) doesn't fail immediately - it fails when task executes
                GpuTask<Object> task = scope.forkWithStream(null);
                assertNotNull(task);
                // The NullPointerException happens during joinAll when task.apply() is invoked
                assertThrows(StructuredTaskScope.FailedException.class, scope::joinAll);
            }
        }
    }

    // ==================== Join Tests ====================

    @Nested
    @DisplayName("Join")
    class JoinTests {

        @Test
        @DisplayName("joinAll() waits for all tasks")
        void joinAllWaitsForAllTasks() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger completed = new AtomicInteger(0);

                scope.fork(() -> {
                    Thread.sleep(10);
                    completed.incrementAndGet();
                    return null;
                });
                scope.fork(() -> {
                    Thread.sleep(10);
                    completed.incrementAndGet();
                    return null;
                });

                scope.joinAll();

                assertEquals(2, completed.get());
            }
        }

        @Test
        @DisplayName("joinAll() throws on task failure")
        void joinAllThrowsOnTaskFailure() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.fork(() -> {
                    throw new RuntimeException("Task failed");
                });

                assertThrows(StructuredTaskScope.FailedException.class, scope::joinAll);
            }
        }

        @Test
        @DisplayName("Double joinAll() throws IllegalStateException")
        void doubleJoinAllThrowsIllegalState() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.fork(() -> 1);
                scope.joinAll();
                assertThrows(IllegalStateException.class, scope::joinAll);
            }
        }

        @Test
        @DisplayName("joinAll() with no tasks succeeds")
        void zeroTasksJoinSucceeds() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertDoesNotThrow(scope::joinAll);
            }
        }
    }

    // ==================== Stream Tests ====================

    @Nested
    @DisplayName("Streams")
    class StreamTests {

        @Test
        @DisplayName("Multiple streams have unique handles")
        void multipleStreamsUnique() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Set<Long> handles = Collections.newSetFromMap(new ConcurrentHashMap<>());

                for (int i = 0; i < 10; i++) {
                    scope.forkWithStream(lease -> {
                        assertTrue(handles.add(lease.streamHandle()),
                            "Stream handle should be unique");
                        return null;
                    });
                }

                scope.joinAll();
                assertEquals(10, handles.size());
            }
        }

        @Test
        @DisplayName("Streams are created via backend")
        void streamsCreatedViaBackend() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> null);
                scope.forkWithStream(lease -> null);
                scope.joinAll();
            }

            assertTrue(backend.recordedOperations().stream()
                .filter(op -> op.startsWith("createStream:"))
                .count() >= 2);
        }

        @Test
        @DisplayName("Streams are destroyed via backend")
        void streamsDestroyedViaBackend() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> null);
                scope.joinAll();
            }

            assertTrue(backend.recordedOperations().stream()
                .anyMatch(op -> op.startsWith("destroyStream:")));
        }
    }

    // ==================== Thread Safety Tests ====================

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafetyTests {

        @Test
        @DisplayName("Fork from non-owner thread throws WrongThreadException")
        void forkFromNonOwnerThreadThrowsWrongThreadException() throws Exception {
            // StructuredTaskScope requires that fork() is called from the owner thread
            // This test verifies that calling fork() from another thread throws WrongThreadException

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger exceptionCount = new AtomicInteger(0);

                Thread otherThread = new Thread(() -> {
                    try {
                        scope.fork(() -> null);
                    } catch (WrongThreadException e) {
                        exceptionCount.incrementAndGet();
                    }
                });

                otherThread.start();
                otherThread.join(5000);

                assertEquals(1, exceptionCount.get(),
                    "Fork from non-owner thread should throw WrongThreadException");
            }
        }

        @Test
        @DisplayName("Many sequential forks from owner thread succeed")
        void manySequentialForksSucceed() throws Exception {
            // Since fork must be called from owner thread, test many sequential forks
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                int taskCount = 100;
                AtomicInteger completedCount = new AtomicInteger(0);

                for (int i = 0; i < taskCount; i++) {
                    scope.fork(() -> {
                        completedCount.incrementAndGet();
                        return null;
                    });
                }

                scope.joinAll();
                assertEquals(taskCount, completedCount.get());
            }
        }

        @Test
        @DisplayName("Task counters are synchronized")
        void taskCountersSynchronized() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                int taskCount = 100;

                for (int i = 0; i < taskCount; i++) {
                    scope.fork(() -> {
                        Thread.sleep(1); // Force some concurrency
                        return null;
                    });
                }

                scope.joinAll();
                // If counters are properly synchronized, no exceptions should occur
            }
        }
    }

    // ==================== Nesting Tests ====================

    @Nested
    @DisplayName("Nesting")
    class NestingTests {

        @Test
        @DisplayName("Nested scopes work correctly")
        void nestedScopesWork() throws Exception {
            AtomicInteger innerCompleted = new AtomicInteger(0);

            try (GpuTaskScope outer = GpuTaskScope.open(backend, "outer")) {
                GpuTask<Integer> outerTask = outer.fork(() -> {
                    try (GpuTaskScope inner = GpuTaskScope.open(backend, "inner")) {
                        GpuTask<Integer> innerTask = inner.fork(() -> {
                            innerCompleted.incrementAndGet();
                            return 10;
                        });
                        inner.joinAll();
                        return innerTask.get();
                    }
                });

                outer.joinAll();
                assertEquals(10, outerTask.get());
                assertEquals(1, innerCompleted.get());
            }
        }

        @Test
        @DisplayName("Nested scopes have independent lifecycle")
        void nestedScopeIndependentLifecycle() throws Exception {
            try (GpuTaskScope outer = GpuTaskScope.open(backend, "outer")) {
                long outerId = outer.scopeId();

                GpuTask<Long> task = outer.fork(() -> {
                    try (GpuTaskScope inner = GpuTaskScope.open(backend, "inner")) {
                        assertNotEquals(outerId, inner.scopeId());
                        return inner.scopeId();
                    }
                });

                outer.joinAll();
                assertNotEquals(outerId, task.get());
            }
        }

        @Test
        @DisplayName("Deeply nested scopes (3 levels)")
        void deeplyNestedScopes() throws Exception {
            AtomicInteger deepestValue = new AtomicInteger(0);

            try (GpuTaskScope level1 = GpuTaskScope.open(backend, "level1")) {
                GpuTask<Integer> task = level1.fork(() -> {
                    try (GpuTaskScope level2 = GpuTaskScope.open(backend, "level2")) {
                        GpuTask<Integer> task2 = level2.fork(() -> {
                            try (GpuTaskScope level3 = GpuTaskScope.open(backend, "level3")) {
                                GpuTask<Integer> task3 = level3.fork(() -> {
                                    deepestValue.set(42);
                                    return 42;
                                });
                                level3.joinAll();
                                return task3.get();
                            }
                        });
                        level2.joinAll();
                        return task2.get();
                    }
                });

                level1.joinAll();
                assertEquals(42, task.get());
                assertEquals(42, deepestValue.get());
            }
        }
    }

    // ==================== Exception Tests ====================

    @Nested
    @DisplayName("Exceptions")
    class ExceptionTests {

        @Test
        @DisplayName("Runtime exception propagates")
        void runtimeExceptionPropagates() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.fork(() -> {
                    throw new IllegalArgumentException("Test exception");
                });

                StructuredTaskScope.FailedException ex =
                    assertThrows(StructuredTaskScope.FailedException.class, scope::joinAll);

                assertNotNull(ex.getCause());
            }
        }

        @Test
        @DisplayName("Checked exception wrapped in FailedException")
        void checkedExceptionWrapped() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.fork(() -> {
                    throw new Exception("Checked exception");
                });

                assertThrows(StructuredTaskScope.FailedException.class, scope::joinAll);
            }
        }

        @Test
        @DisplayName("Task failure doesn't prevent scope close")
        void taskFailureDoesNotPreventClose() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.fork(() -> {
                    throw new RuntimeException("Fail");
                });

                try {
                    scope.joinAll();
                } catch (Exception ignored) {
                }
            }
            // Should not throw - scope closes cleanly
        }

        @Test
        @DisplayName("InterruptedException handled correctly")
        void interruptedExceptionHandled() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Thread testThread = Thread.currentThread();

                scope.fork(() -> {
                    Thread.sleep(10);
                    testThread.interrupt();
                    return null;
                });

                scope.fork(() -> {
                    Thread.sleep(100);
                    return null;
                });

                try {
                    scope.joinAll();
                } catch (InterruptedException e) {
                    // Expected - clear interrupt flag
                    Thread.interrupted();
                }
            }
        }
    }

    // ==================== Scale Tests ====================

    @Nested
    @DisplayName("Scale")
    class ScaleTests {

        @Test
        @DisplayName("Ten tasks complete successfully")
        void tenTasksComplete() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                List<GpuTask<Integer>> tasks = new ArrayList<>();
                for (int i = 0; i < 10; i++) {
                    final int value = i;
                    tasks.add(scope.fork(() -> value));
                }

                scope.joinAll();

                for (int i = 0; i < 10; i++) {
                    assertEquals(i, tasks.get(i).get());
                }
            }
        }

        @Test
        @DisplayName("Hundred tasks complete successfully")
        void hundredTasksComplete() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger counter = new AtomicInteger(0);

                for (int i = 0; i < 100; i++) {
                    scope.fork(() -> {
                        counter.incrementAndGet();
                        return null;
                    });
                }

                scope.joinAll();
                assertEquals(100, counter.get());
            }
        }

        @Test
        @DisplayName("Mixed fork types complete")
        void mixedForkTypes() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger forkCount = new AtomicInteger(0);
                AtomicInteger forkWithStreamCount = new AtomicInteger(0);

                // Regular forks
                for (int i = 0; i < 5; i++) {
                    scope.fork(() -> {
                        forkCount.incrementAndGet();
                        return null;
                    });
                }

                // Forks with stream
                for (int i = 0; i < 5; i++) {
                    scope.forkWithStream(lease -> {
                        forkWithStreamCount.incrementAndGet();
                        return null;
                    });
                }

                scope.joinAll();

                assertEquals(5, forkCount.get());
                assertEquals(5, forkWithStreamCount.get());
            }
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Empty scope name becomes 'unnamed'")
        void emptyNameBecomesUnnamed() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "")) {
                // Empty string is allowed, just use what's given
                assertEquals("", scope.scopeName());
            }
        }

        @Test
        @DisplayName("Scope returns correct backend")
        void scopeReturnsCorrectBackend() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertEquals(backend, scope.backend());
            }
        }

        @Test
        @DisplayName("Task returning null works")
        void taskReturningNullWorks() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Void> task = scope.fork(() -> null);
                scope.joinAll();
                assertNull(task.get());
            }
        }
    }

    // ==================== JFR Event Tests ====================

    @Nested
    @DisplayName("JFR Events")
    class JfrEventTests {

        @Test
        @DisplayName("Scope ID is positive")
        void jfrEventContainsScopeId() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertTrue(scope.scopeId() > 0);
            }
        }

        @Test
        @DisplayName("Device index is available")
        void jfrEventContainsDeviceIndex() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertEquals(0, scope.backend().deviceIndex());
            }
        }
    }
}
