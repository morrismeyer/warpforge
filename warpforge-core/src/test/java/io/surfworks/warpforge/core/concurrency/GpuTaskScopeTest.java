package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link GpuTaskScope}.
 */
@DisplayName("GpuTaskScope")
class GpuTaskScopeTest {

    private MockGpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new MockGpuBackend();
    }

    @Nested
    @DisplayName("Scope Lifecycle")
    class ScopeLifecycle {

        @Test
        @DisplayName("opens and closes cleanly")
        void opensAndClosesCleanly() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertNotNull(scope);
                assertTrue(scope.scopeId() > 0);
            }
        }

        @Test
        @DisplayName("has unique scope IDs")
        void hasUniqueScopeIds() {
            long id1, id2;
            try (GpuTaskScope scope1 = GpuTaskScope.open(backend)) {
                id1 = scope1.scopeId();
            }
            try (GpuTaskScope scope2 = GpuTaskScope.open(backend)) {
                id2 = scope2.scopeId();
            }
            assertTrue(id2 > id1, "Scope IDs should increase");
        }

        @Test
        @DisplayName("returns backend and scopeName")
        void returnsBackendAndScopeName() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend, "test-scope")) {
                assertEquals(backend, scope.backend());
                assertEquals("test-scope", scope.scopeName());
            }
        }
    }

    @Nested
    @DisplayName("Stream Cleanup")
    class StreamCleanup {

        @Test
        @DisplayName("releases streams on normal exit")
        void releasesStreamsOnNormalExit() throws InterruptedException {
            assertEquals(0, backend.activeStreamCount(), "No streams before scope");

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.forkWithStream(lease -> {
                    assertTrue(backend.isStreamActive(lease.streamHandle()));
                    return 42;
                });
                scope.joinAll();
                assertEquals(42, task.get());
            }

            assertEquals(0, backend.activeStreamCount(), "All streams released after scope");
        }

        @Test
        @DisplayName("releases streams on exception")
        void releasesStreamsOnException() {
            assertEquals(0, backend.activeStreamCount());

            try {
                try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                    scope.forkWithStream(lease -> {
                        assertTrue(backend.isStreamActive(lease.streamHandle()));
                        throw new RuntimeException("Simulated failure");
                    });
                    scope.joinAll();
                }
            } catch (Exception e) {
                // Expected
            }

            assertEquals(0, backend.activeStreamCount(), "Streams released despite exception");
        }

        @Test
        @DisplayName("releases multiple streams from concurrent tasks")
        void releasesMultipleStreams() throws InterruptedException {
            assertEquals(0, backend.activeStreamCount());

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger maxConcurrent = new AtomicInteger(0);

                for (int i = 0; i < 5; i++) {
                    scope.forkWithStream(lease -> {
                        int current = backend.activeStreamCount();
                        maxConcurrent.updateAndGet(max -> Math.max(max, current));
                        Thread.sleep(10);
                        return current;
                    });
                }
                scope.joinAll();

                assertTrue(maxConcurrent.get() >= 1, "Should have had concurrent streams");
            }

            assertEquals(0, backend.activeStreamCount(), "All streams released");
        }
    }

    @Nested
    @DisplayName("Task Forking")
    class TaskForking {

        @Test
        @DisplayName("fork executes callable and returns result")
        void forkExecutesAndReturns() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<String> task = scope.fork(() -> "hello");
                scope.joinAll();
                assertTrue(task.isSuccess());
                assertEquals("hello", task.get());
            }
        }

        @Test
        @DisplayName("forkWithStream provides valid stream handle")
        void forkWithStreamProvidesHandle() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Long> task = scope.forkWithStream(lease -> {
                    assertNotNull(lease);
                    assertTrue(lease.streamHandle() > 0);
                    return lease.streamHandle();
                });
                scope.joinAll();
                assertTrue(task.get() > 0);
            }
        }

        @Test
        @DisplayName("multiple forks execute in parallel")
        void multipleForks() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                AtomicInteger counter = new AtomicInteger(0);

                GpuTask<Integer> t1 = scope.fork(() -> {
                    Thread.sleep(10);
                    return counter.incrementAndGet();
                });
                GpuTask<Integer> t2 = scope.fork(() -> {
                    Thread.sleep(10);
                    return counter.incrementAndGet();
                });
                GpuTask<Integer> t3 = scope.fork(() -> {
                    Thread.sleep(10);
                    return counter.incrementAndGet();
                });

                scope.joinAll();

                assertEquals(3, counter.get());
                assertTrue(t1.isSuccess());
                assertTrue(t2.isSuccess());
                assertTrue(t3.isSuccess());
            }
        }

        @Test
        @DisplayName("cannot join twice")
        void cannotJoinTwice() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.fork(() -> 1);
                scope.joinAll();
                assertThrows(IllegalStateException.class, scope::joinAll);
            }
        }
    }

    @Nested
    @DisplayName("GpuTask")
    class GpuTaskTests {

        @Test
        @DisplayName("tracks state correctly")
        void tracksState() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    Thread.sleep(20);
                    return 42;
                });

                assertFalse(task.isSuccess());

                scope.joinAll();

                assertTrue(task.isSuccess());
                assertFalse(task.isFailed());
                assertFalse(task.isUnavailable());
                assertEquals(42, task.get());
            }
        }

        @Test
        @DisplayName("captures exception for failed tasks")
        void capturesException() {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    throw new IllegalArgumentException("test error");
                });

                try {
                    scope.joinAll();
                } catch (Exception e) {
                    // Expected
                }

                assertTrue(task.isFailed());
                assertNotNull(task.exception());
                assertTrue(task.exception() instanceof IllegalArgumentException);
            }
        }

        @Test
        @DisplayName("lease is null for fork without stream")
        void leaseNullForFork() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 1);
                scope.joinAll();
                assertEquals(null, task.lease());
            }
        }

        @Test
        @DisplayName("lease is present for forkWithStream")
        void leasePresentForForkWithStream() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.forkWithStream(lease -> 1);
                scope.joinAll();
                assertNotNull(task.lease());
            }
        }
    }

    @Nested
    @DisplayName("GpuLease")
    class GpuLeaseTests {

        @Test
        @DisplayName("provides stream handle and parent scope")
        void providesStreamAndParent() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Boolean> task = scope.forkWithStream(lease -> {
                    assertTrue(lease.streamHandle() > 0);
                    assertEquals(scope, lease.parentScope());
                    assertTrue(lease.acquireTimeNanos() > 0);
                    return true;
                });
                scope.joinAll();
                assertTrue(task.get());
            }
        }

        @Test
        @DisplayName("synchronize waits for stream")
        void synchronizeWaits() throws InterruptedException {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Long> task = scope.forkWithStream(lease -> {
                    long before = System.nanoTime();
                    lease.synchronize();
                    long after = System.nanoTime();
                    return after - before;
                });
                scope.joinAll();
                assertTrue(task.get() > 0, "Synchronize should take some time");
            }
        }
    }
}
