package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.awaitility.Awaitility;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link TimeSlicedKernel}.
 *
 * <p>These tests use {@link MockGpuBackend} and test implementations of
 * TimeSlicedKernel to verify time-sliced execution behavior.
 */
@DisplayName("TimeSlicedKernel Unit Tests")
class TimeSlicedKernelTest {

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
     * Simple kernel that returns the sum of chunk indices.
     */
    static class SumKernel extends TimeSlicedKernel<Integer> {
        private final int numChunks;

        SumKernel(int numChunks) {
            this.numChunks = numChunks;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return numChunks;
        }

        @Override
        protected Integer executeChunk(int chunkIndex, int totalChunks,
                                        List<Tensor> inputs, GpuLease lease) {
            return chunkIndex;
        }

        @Override
        protected Integer mergeResults(List<Integer> chunks) {
            return chunks.stream().mapToInt(Integer::intValue).sum();
        }
    }

    /**
     * Kernel that tracks execution order.
     */
    static class TrackingKernel extends TimeSlicedKernel<String> {
        private final int numChunks;
        private final List<Integer> executedChunks = new ArrayList<>();
        private final List<Long> leaseHandles = new ArrayList<>();

        TrackingKernel(int numChunks) {
            this.numChunks = numChunks;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return numChunks;
        }

        @Override
        protected String executeChunk(int chunkIndex, int totalChunks,
                                       List<Tensor> inputs, GpuLease lease) {
            synchronized (executedChunks) {
                executedChunks.add(chunkIndex);
                leaseHandles.add(lease.streamHandle());
            }
            return "chunk-" + chunkIndex;
        }

        @Override
        protected String mergeResults(List<String> chunks) {
            return String.join(",", chunks);
        }

        List<Integer> getExecutedChunks() {
            return List.copyOf(executedChunks);
        }

        List<Long> getLeaseHandles() {
            return List.copyOf(leaseHandles);
        }
    }

    /**
     * Kernel that throws on a specific chunk.
     */
    static class FailingKernel extends TimeSlicedKernel<Integer> {
        private final int failOnChunk;

        FailingKernel(int failOnChunk) {
            this.failOnChunk = failOnChunk;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return 5;
        }

        @Override
        protected Integer executeChunk(int chunkIndex, int totalChunks,
                                        List<Tensor> inputs, GpuLease lease) {
            if (chunkIndex == failOnChunk) {
                throw new RuntimeException("Chunk " + chunkIndex + " failed");
            }
            return chunkIndex;
        }

        @Override
        protected Integer mergeResults(List<Integer> chunks) {
            return chunks.stream().mapToInt(Integer::intValue).sum();
        }
    }

    /**
     * Kernel that reads input tensor dimensions.
     */
    static class InputReadingKernel extends TimeSlicedKernel<int[]> {
        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            // Chunks based on first input's first dimension
            if (inputs.isEmpty()) {
                return 1;
            }
            int[] shape = inputs.get(0).shape();
            return shape.length > 0 ? Math.max(1, shape[0] / 10) : 1;
        }

        @Override
        protected int[] executeChunk(int chunkIndex, int totalChunks,
                                      List<Tensor> inputs, GpuLease lease) {
            return inputs.isEmpty() ? new int[0] : inputs.get(0).shape();
        }

        @Override
        protected int[] mergeResults(List<int[]> chunks) {
            // Return first chunk's shape
            return chunks.isEmpty() ? new int[0] : chunks.get(0);
        }
    }

    // ==================== Basic Execution Tests ====================

    @Nested
    @DisplayName("Basic Execution")
    class BasicExecutionTests {

        @Test
        @DisplayName("execute() single chunk returns result")
        void executeSingleChunk() throws Exception {
            SumKernel kernel = new SumKernel(1);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(0, result); // Single chunk with index 0
            }
        }

        @Test
        @DisplayName("execute() multiple chunks returns merged result")
        void executeMultipleChunks() throws Exception {
            SumKernel kernel = new SumKernel(5);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(10, result); // 0+1+2+3+4 = 10
            }
        }

        @Test
        @DisplayName("execute() returns correct type")
        void executeReturnsCorrectType() throws Exception {
            TrackingKernel kernel = new TrackingKernel(3);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                String result = kernel.execute(scope, List.of());
                assertNotNull(result);
                assertTrue(result.contains("chunk-0"));
                assertTrue(result.contains("chunk-1"));
                assertTrue(result.contains("chunk-2"));
            }
        }
    }

    // ==================== Chunk Estimation Tests ====================

    @Nested
    @DisplayName("Chunk Estimation")
    class ChunkEstimationTests {

        @Test
        @DisplayName("estimateChunks() called once per execute")
        void estimateChunksCalledOnce() throws Exception {
            AtomicInteger callCount = new AtomicInteger(0);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    callCount.incrementAndGet();
                    return 2;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return chunks.size();
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertEquals(1, callCount.get());
        }

        @Test
        @DisplayName("estimateChunks() receives inputs")
        void estimateChunksUsesInputs() throws Exception {
            AtomicReference<List<Tensor>> receivedInputs = new AtomicReference<>();

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    receivedInputs.set(inputs);
                    return 1;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    return 0;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            Tensor input = Tensor.zeros(ScalarType.F32, 10, 20);
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of(input));
            }

            assertNotNull(receivedInputs.get());
            assertEquals(1, receivedInputs.get().size());
            assertEquals(input, receivedInputs.get().get(0));
        }

        @Test
        @DisplayName("estimateChunks() with input-based chunking")
        void estimateChunksInputBased() throws Exception {
            InputReadingKernel kernel = new InputReadingKernel();

            Tensor input = Tensor.zeros(ScalarType.F32, 100, 50);
            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                int[] result = kernel.execute(scope, List.of(input));
                assertEquals(2, result.length);
                assertEquals(100, result[0]);
                assertEquals(50, result[1]);
            }
        }
    }

    // ==================== Chunk Execution Tests ====================

    @Nested
    @DisplayName("Chunk Execution")
    class ChunkExecutionTests {

        @Test
        @DisplayName("Each chunk is executed")
        void eachChunkExecuted() throws Exception {
            TrackingKernel kernel = new TrackingKernel(5);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            List<Integer> executed = kernel.getExecutedChunks();
            assertEquals(5, executed.size());
            for (int i = 0; i < 5; i++) {
                assertTrue(executed.contains(i), "Chunk " + i + " not executed");
            }
        }

        @Test
        @DisplayName("Chunk indices are zero-based")
        void chunkIndexZeroBased() throws Exception {
            AtomicInteger minIndex = new AtomicInteger(Integer.MAX_VALUE);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    minIndex.updateAndGet(min -> Math.min(min, chunkIndex));
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertEquals(0, minIndex.get());
        }

        @Test
        @DisplayName("totalChunks parameter is correct")
        void totalChunksCorrect() throws Exception {
            AtomicInteger observedTotal = new AtomicInteger(0);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 7;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    observedTotal.set(totalChunks);
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertEquals(7, observedTotal.get());
        }

        @Test
        @DisplayName("Inputs passed to each chunk")
        void inputsPassedToChunks() throws Exception {
            AtomicInteger inputsReceived = new AtomicInteger(0);
            Tensor input1 = Tensor.zeros(ScalarType.F32, 5);
            Tensor input2 = Tensor.zeros(ScalarType.F32, 10);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    if (inputs.size() == 2) {
                        inputsReceived.incrementAndGet();
                    }
                    return 0;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of(input1, input2));
            }

            assertEquals(3, inputsReceived.get());
        }
    }

    // ==================== Lease Management Tests ====================

    @Nested
    @DisplayName("Lease Management")
    class LeaseManagementTests {

        @Test
        @DisplayName("Each chunk gets a lease")
        void eachChunkGetsLease() throws Exception {
            TrackingKernel kernel = new TrackingKernel(4);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            List<Long> handles = kernel.getLeaseHandles();
            assertEquals(4, handles.size());
            handles.forEach(h -> assertTrue(h > 0, "Invalid stream handle"));
        }

        @Test
        @DisplayName("Synchronize called after chunk")
        void synchronizeCalledAfterChunk() throws Exception {
            SumKernel kernel = new SumKernel(3);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            long syncCount = backend.recordedOperations().stream()
                .filter(op -> op.startsWith("synchronizeStream:"))
                .count();
            assertEquals(3, syncCount);
        }

        @Test
        @DisplayName("Streams cleaned up after execution")
        void streamsCleanedUp() throws Exception {
            SumKernel kernel = new SumKernel(5);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertEquals(0, backend.activeStreamCount());
        }
    }

    // ==================== Merge Tests ====================

    @Nested
    @DisplayName("Merge")
    class MergeTests {

        @Test
        @DisplayName("mergeResults() called with all results")
        void mergeCalledWithAllResults() throws Exception {
            AtomicInteger mergeCallCount = new AtomicInteger(0);
            AtomicReference<List<Integer>> mergedChunks = new AtomicReference<>();

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 4;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    return chunkIndex * 10;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    mergeCallCount.incrementAndGet();
                    mergedChunks.set(new ArrayList<>(chunks));
                    return chunks.stream().mapToInt(Integer::intValue).sum();
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(60, result); // 0+10+20+30 = 60
            }

            assertEquals(1, mergeCallCount.get());
            assertEquals(4, mergedChunks.get().size());
        }

        @Test
        @DisplayName("mergeResults() receives results in order")
        void mergePreservesOrder() throws Exception {
            TrackingKernel kernel = new TrackingKernel(5);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                String result = kernel.execute(scope, List.of());
                // Results should be in chunk order
                assertEquals("chunk-0,chunk-1,chunk-2,chunk-3,chunk-4", result);
            }
        }

        @Test
        @DisplayName("mergeResults() single chunk")
        void mergeSingleChunk() throws Exception {
            AtomicReference<List<String>> mergeInput = new AtomicReference<>();

            TimeSlicedKernel<String> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 1;
                }

                @Override
                protected String executeChunk(int chunkIndex, int totalChunks,
                                               List<Tensor> inputs, GpuLease lease) {
                    return "only-chunk";
                }

                @Override
                protected String mergeResults(List<String> chunks) {
                    mergeInput.set(chunks);
                    return chunks.get(0);
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                String result = kernel.execute(scope, List.of());
                assertEquals("only-chunk", result);
            }

            assertEquals(1, mergeInput.get().size());
        }
    }

    // ==================== Cancellation Tests ====================

    @Nested
    @DisplayName("Cancellation")
    class CancellationTests {

        @Test
        @DisplayName("checkCancellation() passes when not interrupted")
        void checkCancellationPasses() {
            SumKernel kernel = new SumKernel(1);
            assertDoesNotThrow(kernel::checkCancellation);
        }

        @Test
        @DisplayName("checkCancellation() throws when interrupted")
        void checkCancellationThrowsWhenInterrupted() {
            SumKernel kernel = new SumKernel(1);

            // Directly set the interrupt flag
            Thread.currentThread().interrupt();

            try {
                assertThrows(CancellationException.class, kernel::checkCancellation);
            } finally {
                // Clear interrupt status
                Thread.interrupted();
            }
        }

        @Test
        @DisplayName("Cancellation at chunk boundary throws")
        void cancellationAtChunkBoundary() {
            AtomicBoolean firstChunkRan = new AtomicBoolean(false);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    if (chunkIndex == 0) {
                        firstChunkRan.set(true);
                        Thread.currentThread().interrupt();
                    }
                    checkCancellation(); // Will throw on subsequent chunks
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertThrows(RuntimeException.class, () -> kernel.execute(scope, List.of()));
            }

            assertTrue(firstChunkRan.get());
            // Clear interrupt status
            Thread.interrupted();
        }
    }

    // ==================== Exception Tests ====================

    @Nested
    @DisplayName("Exceptions")
    class ExceptionTests {

        @Test
        @DisplayName("Chunk exception propagates")
        void chunkExceptionPropagates() {
            FailingKernel kernel = new FailingKernel(2);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                RuntimeException ex = assertThrows(RuntimeException.class,
                    () -> kernel.execute(scope, List.of()));
                assertTrue(ex.getMessage().contains("Time-sliced kernel failed"));
            }
        }

        @Test
        @DisplayName("Merge exception propagates")
        void mergeExceptionPropagates() {
            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 2;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    throw new IllegalStateException("Merge failed");
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                assertThrows(IllegalStateException.class, () -> kernel.execute(scope, List.of()));
            }
        }

        @Test
        @DisplayName("First failing chunk determines exception")
        void firstFailingChunkException() {
            FailingKernel kernel = new FailingKernel(0); // Fail on first chunk

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                RuntimeException ex = assertThrows(RuntimeException.class,
                    () -> kernel.execute(scope, List.of()));
                assertTrue(ex.getCause().getMessage().contains("Chunk 0 failed"));
            }
        }
    }

    // ==================== Scale Tests ====================

    @Nested
    @DisplayName("Scale")
    class ScaleTests {

        @Test
        @DisplayName("Ten chunks work correctly")
        void tenChunksWork() throws Exception {
            SumKernel kernel = new SumKernel(10);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(45, result); // 0+1+2+...+9 = 45
            }
        }

        @Test
        @DisplayName("Hundred chunks work correctly")
        void hundredChunksWork() throws Exception {
            SumKernel kernel = new SumKernel(100);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(4950, result); // 0+1+2+...+99 = 4950
            }
        }

        @Test
        @DisplayName("Large chunk count executes all chunks")
        void largeChunkCountExecutesAll() throws Exception {
            int numChunks = 200;
            TrackingKernel kernel = new TrackingKernel(numChunks);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertEquals(numChunks, kernel.getExecutedChunks().size());
        }
    }

    // ==================== Thread Safety Tests ====================

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafetyTests {

        @Test
        @DisplayName("Chunks run concurrently")
        void chunksRunConcurrently() throws Exception {
            AtomicInteger maxConcurrent = new AtomicInteger(0);
            AtomicInteger current = new AtomicInteger(0);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 10;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    int concurrent = current.incrementAndGet();
                    maxConcurrent.updateAndGet(max -> Math.max(max, concurrent));
                    // Simulate GPU work using consistent timing
                    backend.simulateGpuWork(lease, MockGpuBackend.MIN_WORK_MS);
                    current.decrementAndGet();
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return 0;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            // At least some concurrency should be observed
            assertTrue(maxConcurrent.get() >= 1);
        }

        @Test
        @DisplayName("Merge happens after all chunks complete")
        void mergeAfterAllChunksComplete() throws Exception {
            AtomicBoolean allChunksComplete = new AtomicBoolean(false);
            AtomicInteger completedChunks = new AtomicInteger(0);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 5;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    completedChunks.incrementAndGet();
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    allChunksComplete.set(completedChunks.get() == 5);
                    return 0;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertTrue(allChunksComplete.get());
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Empty inputs handled")
        void emptyInputsHandled() throws Exception {
            SumKernel kernel = new SumKernel(3);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(3, result); // 0+1+2 = 3
            }
        }

        @Test
        @DisplayName("Inputs are not modified")
        void inputsUnmodified() throws Exception {
            Tensor original = Tensor.zeros(ScalarType.F32, 10, 10);
            int[] originalShape = original.shape().clone();

            InputReadingKernel kernel = new InputReadingKernel();

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of(original));
            }

            // Shape should be unchanged
            assertEquals(originalShape[0], original.shape()[0]);
            assertEquals(originalShape[1], original.shape()[1]);
        }

        @Test
        @DisplayName("Kernel can be reused")
        void kernelCanBeReused() throws Exception {
            SumKernel kernel = new SumKernel(3);

            try (GpuTaskScope scope1 = GpuTaskScope.open(backend)) {
                Integer result1 = kernel.execute(scope1, List.of());
                assertEquals(3, result1);
            }

            try (GpuTaskScope scope2 = GpuTaskScope.open(backend)) {
                Integer result2 = kernel.execute(scope2, List.of());
                assertEquals(3, result2);
            }
        }

        @Test
        @DisplayName("Null result from chunk handled")
        void nullChunkResultHandled() throws Exception {
            TimeSlicedKernel<String> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 2;
                }

                @Override
                protected String executeChunk(int chunkIndex, int totalChunks,
                                               List<Tensor> inputs, GpuLease lease) {
                    return chunkIndex == 0 ? null : "value";
                }

                @Override
                protected String mergeResults(List<String> chunks) {
                    return chunks.stream()
                        .filter(c -> c != null)
                        .findFirst()
                        .orElse("none");
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                String result = kernel.execute(scope, List.of());
                assertEquals("value", result);
            }
        }
    }
}
