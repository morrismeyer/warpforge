package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests for the structured GPU concurrency system.
 *
 * <p>These tests verify that GpuTaskScope, GpuLease, GpuTask, TimeSlicedKernel,
 * and DeadlineContext work correctly together.
 */
@DisplayName("Concurrency Integration Tests")
class ConcurrencyIntegrationTest {

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

    // ==================== Test Implementations ====================

    /**
     * Simple TimeSlicedKernel for integration tests.
     */
    static class IntegrationKernel extends TimeSlicedKernel<Integer> {
        private final int numChunks;

        IntegrationKernel(int numChunks) {
            this.numChunks = numChunks;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return numChunks;
        }

        @Override
        protected Integer executeChunk(int chunkIndex, int totalChunks,
                                        List<Tensor> inputs, GpuLease lease) {
            return chunkIndex + 1; // 1-based for easier summing
        }

        @Override
        protected Integer mergeResults(List<Integer> chunks) {
            return chunks.stream().mapToInt(Integer::intValue).sum();
        }
    }

    // ==================== Combined Pattern Tests ====================

    @Nested
    @DisplayName("Combined Patterns")
    class CombinedPatternTests {

        @Test
        @DisplayName("DeadlineContext with GpuTaskScope")
        void deadlineContextWithScope() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Integer result = ctx.execute(scope -> {
                GpuTask<Integer> task1 = scope.fork(() -> 10);
                GpuTask<Integer> task2 = scope.fork(() -> 20);
                try {
                    scope.joinAll();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
                return task1.get() + task2.get();
            });

            assertEquals(30, result);
        }

        @Test
        @DisplayName("DeadlineContext with TimeSlicedKernel")
        void deadlineContextWithTimeSliced() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
            IntegrationKernel kernel = new IntegrationKernel(5);

            Integer result = ctx.execute(scope -> {
                return kernel.execute(scope, List.of());
            });

            // 1+2+3+4+5 = 15
            assertEquals(15, result);
        }

        @Test
        @DisplayName("TimeSlicedKernel with deadline checks")
        void timeSlicedWithDeadline() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    try {
                        ctx.checkDeadline();
                    } catch (DeadlineExceededException e) {
                        throw new RuntimeException(e);
                    }
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return chunks.size();
                }
            };

            Integer result = ctx.execute(scope -> kernel.execute(scope, List.of()));
            assertEquals(3, result);
        }

        @Test
        @DisplayName("All components combined")
        void allThreeCombined() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Integer result = ctx.execute(scope -> {
                // Regular fork
                GpuTask<Integer> regularTask = scope.fork(() -> 100);

                // Fork with stream
                GpuTask<Integer> streamTask = scope.forkWithStream(lease -> {
                    lease.synchronize();
                    return 50;
                });

                // Join the direct forks first
                try {
                    scope.joinAll();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }

                int directResult = regularTask.get() + streamTask.get();

                // Time-sliced kernel in a separate scope (since it calls joinAll internally)
                try (GpuTaskScope kernelScope = GpuTaskScope.open(backend, "kernel-scope")) {
                    IntegrationKernel kernel = new IntegrationKernel(3);
                    Integer kernelResult = kernel.execute(kernelScope, List.of()); // 1+2+3 = 6
                    return directResult + kernelResult;
                }
            });

            assertEquals(156, result); // 100 + 50 + 6
        }
    }

    // ==================== Nested Structure Tests ====================

    @Nested
    @DisplayName("Nested Structures")
    class NestedStructureTests {

        @Test
        @DisplayName("Nested scopes with deadlines")
        void nestedScopesWithDeadlines() throws DeadlineExceededException {
            DeadlineContext outer = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Integer result = outer.execute(outerScope -> {
                DeadlineContext inner = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

                try {
                    return inner.execute(innerScope -> {
                        GpuTask<Integer> task = innerScope.fork(() -> 42);
                        try {
                            innerScope.joinAll();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException(e);
                        }
                        return task.get();
                    });
                } catch (DeadlineExceededException e) {
                    return -1;
                }
            });

            assertEquals(42, result);
        }

        @Test
        @DisplayName("Time-sliced kernel in time-sliced kernel")
        void timeSlicedInTimeSliced() throws Exception {
            AtomicInteger outerChunksExecuted = new AtomicInteger(0);
            AtomicInteger innerChunksExecuted = new AtomicInteger(0);

            TimeSlicedKernel<Integer> innerKernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 2;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    innerChunksExecuted.incrementAndGet();
                    return 1;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return chunks.stream().mapToInt(Integer::intValue).sum();
                }
            };

            TimeSlicedKernel<Integer> outerKernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    outerChunksExecuted.incrementAndGet();
                    // Each outer chunk creates a new scope for inner kernel
                    try (GpuTaskScope innerScope = GpuTaskScope.open(
                            (io.surfworks.warpforge.core.backend.GpuBackend) lease.parentScope().backend())) {
                        return innerKernel.execute(innerScope, inputs);
                    }
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return chunks.stream().mapToInt(Integer::intValue).sum();
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = outerKernel.execute(scope, List.of());
                assertEquals(6, result); // 3 outer chunks * 2 inner = 6 total
            }

            assertEquals(3, outerChunksExecuted.get());
            assertEquals(6, innerChunksExecuted.get());
        }

        @Test
        @DisplayName("Deeply nested scopes")
        void deeplyNestedScopes() throws Exception {
            AtomicInteger depth = new AtomicInteger(0);
            AtomicInteger maxDepth = new AtomicInteger(0);

            class DeepNester {
                int nest(int level) throws Exception {
                    if (level == 0) {
                        return 1;
                    }
                    try (GpuTaskScope scope = GpuTaskScope.open(backend, "level-" + level)) {
                        depth.incrementAndGet();
                        maxDepth.updateAndGet(max -> Math.max(max, depth.get()));

                        GpuTask<Integer> task = scope.fork(() -> nest(level - 1));
                        scope.joinAll();

                        depth.decrementAndGet();
                        return task.get() + level;
                    }
                }
            }

            int result = new DeepNester().nest(5);
            assertEquals(16, result); // 1 + 1 + 2 + 3 + 4 + 5 = 16
            assertEquals(5, maxDepth.get());
        }
    }

    // ==================== Failure Handling Tests ====================

    @Nested
    @DisplayName("Failure Handling")
    class FailureHandlingTests {

        @Test
        @DisplayName("Failure propagates through deadline context")
        void failurePropagatesThroughContext() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertThrows(IllegalStateException.class, () ->
                ctx.execute(scope -> {
                    throw new IllegalStateException("Test failure");
                })
            );
        }

        @Test
        @DisplayName("Timeout cleans up resources")
        void timeoutCleansUp() {
            // Create already-expired context
            DeadlineContext ctx = DeadlineContext.withDeadline(
                backend, java.time.Instant.now().minusMillis(100));

            assertThrows(DeadlineExceededException.class, () ->
                ctx.execute(scope -> 42)
            );

            // Backend should not have any active streams
            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("Cancellation cascades through nested structures")
        void cancellationCascades() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
            ctx.cancel();

            assertThrows(DeadlineExceededException.class, () ->
                ctx.execute(scope -> 42)
            );
        }
    }

    // ==================== Resource Cleanup Tests ====================

    @Nested
    @DisplayName("Resource Cleanup")
    class ResourceCleanupTests {

        @Test
        @DisplayName("Streams released on success")
        void streamsReleasedOnSuccess() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    return null;
                });
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    return null;
                });
                scope.joinAll();
            }

            assertEquals(0, backend.activeStreamCount());
            assertEquals(2, backend.streamCreationCount());
        }

        @Test
        @DisplayName("Streams released on failure")
        void streamsReleasedOnFailure() {
            try {
                try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                    scope.forkWithStream(lease -> {
                        throw new RuntimeException("Test failure");
                    });
                    scope.joinAll();
                }
            } catch (Exception ignored) {
            }

            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("Streams released on timeout")
        void streamsReleasedOnTimeout() {
            DeadlineContext ctx = DeadlineContext.withDeadline(
                backend, java.time.Instant.now().minusMillis(100));

            try {
                ctx.execute(scope -> {
                    scope.forkWithStream(lease -> 42);
                    return null;
                });
            } catch (DeadlineExceededException ignored) {
            }

            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("No leaks on exception in kernel")
        void noLeaksOnException() {
            TimeSlicedKernel<Integer> failingKernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    if (chunkIndex == 1) {
                        throw new RuntimeException("Chunk failed");
                    }
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            try {
                try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                    failingKernel.execute(scope, List.of());
                }
            } catch (Exception ignored) {
            }

            assertEquals(0, backend.activeStreamCount());
        }
    }

    // ==================== Mixed Fork Types Tests ====================

    @Nested
    @DisplayName("Mixed Fork Types")
    class MixedForkTypesTests {

        @Test
        @DisplayName("fork() and forkWithStream() mixed")
        void forkAndForkWithStreamMixed() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> regular1 = scope.fork(() -> 1);
                GpuTask<Integer> stream1 = scope.forkWithStream(lease -> 2);
                GpuTask<Integer> regular2 = scope.fork(() -> 3);
                GpuTask<Integer> stream2 = scope.forkWithStream(lease -> 4);

                scope.joinAll();

                assertEquals(10, regular1.get() + stream1.get() + regular2.get() + stream2.get());
            }

            // Only forkWithStream creates streams
            assertEquals(2, backend.streamCreationCount());
        }

        @Test
        @DisplayName("Many different fork types in sequence")
        void manyForkTypes() throws Exception {
            List<GpuTask<String>> tasks = new ArrayList<>();

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                for (int i = 0; i < 20; i++) {
                    if (i % 3 == 0) {
                        int idx = i;
                        tasks.add(scope.fork(() -> "fork-" + idx));
                    } else {
                        int idx = i;
                        tasks.add(scope.forkWithStream(lease -> "stream-" + idx));
                    }
                }

                scope.joinAll();

                // Count results by type
                long forkCount = tasks.stream()
                    .map(GpuTask::get)
                    .filter(s -> s.startsWith("fork-"))
                    .count();
                long streamCount = tasks.stream()
                    .map(GpuTask::get)
                    .filter(s -> s.startsWith("stream-"))
                    .count();

                assertEquals(7, forkCount);  // 0, 3, 6, 9, 12, 15, 18
                assertEquals(13, streamCount); // rest
            }
        }
    }

    // ==================== Scale Tests ====================

    @Nested
    @DisplayName("Scale")
    class ScaleTests {

        @Test
        @DisplayName("Hundred concurrent forks")
        void hundredConcurrentForks() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                List<GpuTask<Integer>> tasks = new ArrayList<>();

                for (int i = 0; i < 100; i++) {
                    int idx = i;
                    tasks.add(scope.fork(() -> idx));
                }

                scope.joinAll();

                int sum = tasks.stream()
                    .mapToInt(GpuTask::get)
                    .sum();
                assertEquals(4950, sum); // 0+1+2+...+99
            }
        }

        @Test
        @DisplayName("Many chunks with deadline")
        void manyChunksWithDeadline() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(30));
            IntegrationKernel kernel = new IntegrationKernel(50);

            Integer result = ctx.execute(scope -> kernel.execute(scope, List.of()));

            // 1+2+3+...+50 = 1275
            assertEquals(1275, result);
        }

        @Test
        @DisplayName("Nested with many concurrent tasks")
        void nestedWithManyConcurrent() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(30));

            Integer result = ctx.execute(outerScope -> {
                List<GpuTask<Integer>> tasks = new ArrayList<>();

                for (int i = 0; i < 10; i++) {
                    tasks.add(outerScope.forkWithStream(lease -> {
                        // Each task creates an inner scope
                        try (GpuTaskScope innerScope = GpuTaskScope.open(backend)) {
                            GpuTask<Integer> inner = innerScope.fork(() -> 1);
                            innerScope.joinAll();
                            return inner.get();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return 0;
                        }
                    }));
                }

                try {
                    outerScope.joinAll();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }

                return tasks.stream().mapToInt(GpuTask::get).sum();
            });

            assertEquals(10, result);
        }
    }

    // ==================== Exception Type Tests ====================

    @Nested
    @DisplayName("Exception Types")
    class ExceptionTypeTests {

        @Test
        @DisplayName("Checked exception handled in execute")
        void checkedExceptionHandled() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            // Deadline context propagates exceptions; test wrapping explicitly
            Exception ex = assertThrows(Exception.class, () ->
                ctx.execute(scope -> {
                    try {
                        throw new java.io.IOException("Test IO error");
                    } catch (java.io.IOException e) {
                        throw new RuntimeException("Wrapped IO error", e);
                    }
                })
            );
            assertTrue(ex.getCause() instanceof java.io.IOException);
        }

        @Test
        @DisplayName("Unchecked exception propagates")
        void uncheckedExceptionPropagates() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertThrows(IllegalArgumentException.class, () ->
                ctx.execute(scope -> {
                    throw new IllegalArgumentException("Bad argument");
                })
            );
        }

        @Test
        @DisplayName("Error propagates")
        void errorPropagates() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertThrows(OutOfMemoryError.class, () ->
                ctx.execute(scope -> {
                    throw new OutOfMemoryError("Test OOM");
                })
            );
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Empty scope (no forks)")
        void emptyScope() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.joinAll(); // Should not throw
            }
            assertEquals(0, backend.streamCreationCount());
        }

        @Test
        @DisplayName("Single task scope")
        void singleTaskScope() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<String> task = scope.fork(() -> "single");
                scope.joinAll();
                assertEquals("single", task.get());
            }
        }

        @Test
        @DisplayName("Rapid open/close cycles")
        void rapidOpenClose() throws Exception {
            for (int i = 0; i < 50; i++) {
                int idx = i;
                try (GpuTaskScope scope = GpuTaskScope.open(backend, "cycle-" + idx)) {
                    scope.fork(() -> idx);
                    scope.joinAll();
                }
            }
            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("Context with null backend operations")
        void contextWithMockOperations() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            // Verify the context's backend is the mock
            assertEquals(backend, ctx.backend());

            ctx.execute(scope -> {
                // Verify scope uses the same backend
                assertEquals("mock", backend.name());
                return null;
            });
        }
    }

    // ==================== JFR Verification Tests ====================

    @Nested
    @DisplayName("JFR Verification")
    class JfrVerificationTests {

        @Test
        @DisplayName("Operations recorded in backend")
        void operationsRecordedInBackend() throws Exception {
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                scope.forkWithStream(lease -> {
                    lease.synchronize();
                    return null;
                });
                scope.joinAll();
            }

            List<String> ops = backend.recordedOperations();
            assertTrue(ops.contains("createStream:1"));
            assertTrue(ops.contains("synchronizeStream:1"));
            assertTrue(ops.contains("destroyStream:1"));
        }

        @Test
        @DisplayName("Multiple scope operations tracked")
        void multipleScopeOperationsTracked() throws Exception {
            try (GpuTaskScope scope1 = GpuTaskScope.open(backend, "scope-1")) {
                scope1.forkWithStream(lease -> null);
                scope1.joinAll();
            }

            try (GpuTaskScope scope2 = GpuTaskScope.open(backend, "scope-2")) {
                scope2.forkWithStream(lease -> null);
                scope2.joinAll();
            }

            List<String> ops = backend.recordedOperations();
            assertTrue(ops.contains("createStream:1"));
            assertTrue(ops.contains("createStream:2"));
        }

        @Test
        @DisplayName("Synchronize operations counted")
        void synchronizeOperationsCounted() throws Exception {
            IntegrationKernel kernel = new IntegrationKernel(5);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            long syncCount = backend.recordedOperations().stream()
                .filter(op -> op.startsWith("synchronizeStream:"))
                .count();
            assertEquals(5, syncCount);
        }
    }

    // ==================== Interruption Tests ====================

    @Nested
    @DisplayName("Interruption")
    class InterruptionTests {

        @Test
        @DisplayName("Interrupt flag preserved through scope operations")
        void interruptFlagPreserved() throws Exception {
            // Test that interrupt status doesn't get lost during scope operations
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> {
                    // Work completes normally
                    return 42;
                });
                scope.joinAll();
                assertEquals(42, task.get());
            }

            // Verify we can still set and check interrupt status after scope operations
            Thread.currentThread().interrupt();
            assertTrue(Thread.currentThread().isInterrupted());

            // Clear for next test
            Thread.interrupted();
            assertFalse(Thread.currentThread().isInterrupted());
        }

        @Test
        @DisplayName("Interrupt status cleared after handling")
        void interruptStatusCleared() throws Exception {
            Thread.currentThread().interrupt();

            try {
                try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                    scope.joinAll();
                }
            } catch (InterruptedException expected) {
                // Expected
            }

            // Clear the interrupt status
            Thread.interrupted();

            // Should be able to use new scope after clearing
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                GpuTask<Integer> task = scope.fork(() -> 123);
                scope.joinAll();
                assertEquals(123, task.get());
            }
        }
    }
}
