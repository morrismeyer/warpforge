package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link TimeSlicedKernel}.
 */
@DisplayName("TimeSlicedKernel")
class TimeSlicedKernelTest {

    private MockGpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new MockGpuBackend();
    }

    /**
     * Simple kernel that sums integers from chunks.
     */
    static class SummingKernel extends TimeSlicedKernel<Integer> {
        private final int chunksToCreate;

        SummingKernel(int chunksToCreate) {
            this.chunksToCreate = chunksToCreate;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return chunksToCreate;
        }

        @Override
        protected Integer executeChunk(int chunkIndex, int totalChunks,
                                        List<Tensor> inputs, GpuLease lease) {
            return chunkIndex + 1;  // Return 1, 2, 3, ...
        }

        @Override
        protected Integer mergeResults(List<Integer> chunks) {
            return chunks.stream().mapToInt(i -> i).sum();
        }
    }

    /**
     * Kernel that tracks execution order.
     */
    static class TrackingKernel extends TimeSlicedKernel<Integer> {
        private final int chunks;
        private final List<Integer> executionOrder;

        TrackingKernel(int chunks, List<Integer> executionOrder) {
            this.chunks = chunks;
            this.executionOrder = executionOrder;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return chunks;
        }

        @Override
        protected Integer executeChunk(int chunkIndex, int totalChunks,
                                        List<Tensor> inputs, GpuLease lease) {
            synchronized (executionOrder) {
                executionOrder.add(chunkIndex);
            }
            return chunkIndex;
        }

        @Override
        protected Integer mergeResults(List<Integer> chunks) {
            return chunks.size();
        }
    }

    @Nested
    @DisplayName("Chunk Execution")
    class ChunkExecution {

        @Test
        @DisplayName("executes all chunks and merges results")
        void executesAllChunks() {
            SummingKernel kernel = new SummingKernel(5);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                // Sum of 1 + 2 + 3 + 4 + 5 = 15
                assertEquals(15, result);
            }
        }

        @Test
        @DisplayName("single chunk works correctly")
        void singleChunk() {
            SummingKernel kernel = new SummingKernel(1);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of());
                assertEquals(1, result);
            }
        }

        @Test
        @DisplayName("executes all chunk indices")
        void executesAllIndices() {
            List<Integer> executed = new ArrayList<>();
            TrackingKernel kernel = new TrackingKernel(4, executed);

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                kernel.execute(scope, List.of());
            }

            assertEquals(4, executed.size());
            assertTrue(executed.contains(0));
            assertTrue(executed.contains(1));
            assertTrue(executed.contains(2));
            assertTrue(executed.contains(3));
        }
    }

    @Nested
    @DisplayName("Cancellation")
    class Cancellation {

        @Test
        @DisplayName("checkCancellation throws when interrupted")
        void checkCancellationThrows() {
            TimeSlicedKernel<Void> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 1;
                }

                @Override
                protected Void executeChunk(int chunkIndex, int totalChunks,
                                             List<Tensor> inputs, GpuLease lease) {
                    Thread.currentThread().interrupt();
                    checkCancellation();
                    return null;
                }

                @Override
                protected Void mergeResults(List<Void> chunks) {
                    return null;
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                RuntimeException ex = assertThrows(RuntimeException.class, () -> kernel.execute(scope, List.of()));
                // Verify that the root cause is a CancellationException
                Throwable cause = ex.getCause();
                while (cause != null && !(cause instanceof CancellationException)) {
                    cause = cause.getCause();
                }
                assertTrue(cause instanceof CancellationException, "Root cause should be CancellationException");
            }

            // Clear interrupt flag
            Thread.interrupted();
        }

        @Test
        @DisplayName("cancellation stops further chunks")
        void cancellationStopsChunks() {
            AtomicInteger executedChunks = new AtomicInteger(0);
            AtomicBoolean shouldCancel = new AtomicBoolean(false);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 10;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    if (shouldCancel.get()) {
                        Thread.currentThread().interrupt();
                    }
                    checkCancellation();
                    executedChunks.incrementAndGet();
                    if (chunkIndex == 2) {
                        shouldCancel.set(true);
                    }
                    return chunkIndex;
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return chunks.size();
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                try {
                    kernel.execute(scope, List.of());
                } catch (RuntimeException e) {
                    // Expected - CancellationException is wrapped
                }
            }

            // Should have executed some chunks but not all
            assertTrue(executedChunks.get() < 10);

            // Clear interrupt flag
            Thread.interrupted();
        }
    }

    @Nested
    @DisplayName("Stream Management")
    class StreamManagement {

        @Test
        @DisplayName("each chunk gets a stream")
        void eachChunkGetsStream() {
            AtomicInteger streamsUsed = new AtomicInteger(0);

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    if (lease.streamHandle() > 0) {
                        streamsUsed.incrementAndGet();
                    }
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

            assertEquals(3, streamsUsed.get());
            assertEquals(0, backend.activeStreamCount(), "All streams released");
        }

        @Test
        @DisplayName("streams are released after execution")
        void streamsReleasedAfterExecution() {
            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
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

            // Verification is implicit - if streams are released correctly,
            // synchronization happened
            assertEquals(0, backend.activeStreamCount());
        }
    }

    @Nested
    @DisplayName("Input Handling")
    class InputHandling {

        @Test
        @DisplayName("inputs are passed to each chunk")
        void inputsPassedToChunks() {
            Tensor inputTensor = Tensor.zeros(ScalarType.F32, 10);
            List<Tensor> capturedInputs = new ArrayList<>();

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 3;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    synchronized (capturedInputs) {
                        if (capturedInputs.isEmpty()) {
                            capturedInputs.addAll(inputs);
                        }
                    }
                    return inputs.size();
                }

                @Override
                protected Integer mergeResults(List<Integer> chunks) {
                    return chunks.get(0);
                }
            };

            try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
                Integer result = kernel.execute(scope, List.of(inputTensor));
                assertEquals(1, result);
            }

            assertEquals(1, capturedInputs.size());
            assertEquals(inputTensor, capturedInputs.get(0));
        }

        @Test
        @DisplayName("chunk receives correct indices")
        void chunkReceivesCorrectIndices() {
            List<int[]> capturedIndices = new ArrayList<>();

            TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
                @Override
                protected int estimateChunks(List<Tensor> inputs) {
                    return 4;
                }

                @Override
                protected Integer executeChunk(int chunkIndex, int totalChunks,
                                                List<Tensor> inputs, GpuLease lease) {
                    synchronized (capturedIndices) {
                        capturedIndices.add(new int[]{chunkIndex, totalChunks});
                    }
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

            assertEquals(4, capturedIndices.size());
            for (int[] indices : capturedIndices) {
                assertEquals(4, indices[1], "totalChunks should be 4");
                assertTrue(indices[0] >= 0 && indices[0] < 4, "chunkIndex should be 0-3");
            }
        }
    }
}
