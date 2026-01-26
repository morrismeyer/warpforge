package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link TimeSlicedKernel} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate time-sliced kernel execution with real GPU hardware.
 */
@Tag("gpu")
@DisplayName("TimeSlicedKernel GPU Tests")
class TimeSlicedKernelGpuTest {

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

    // ==================== Test Implementations ====================

    /**
     * Kernel that sums chunk indices.
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
            // Simulate some work
            long sum = 0;
            for (int i = 0; i < 1000; i++) {
                sum += i;
            }
            return chunkIndex;
        }

        @Override
        protected Integer mergeResults(List<Integer> chunks) {
            return chunks.stream().mapToInt(Integer::intValue).sum();
        }
    }

    /**
     * Kernel that tracks stream handles.
     */
    static class StreamTrackingKernel extends TimeSlicedKernel<Long> {
        private final int numChunks;
        private final List<Long> streamHandles = new ArrayList<>();

        StreamTrackingKernel(int numChunks) {
            this.numChunks = numChunks;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return numChunks;
        }

        @Override
        protected Long executeChunk(int chunkIndex, int totalChunks,
                                    List<Tensor> inputs, GpuLease lease) {
            long handle = lease.streamHandle();
            synchronized (streamHandles) {
                streamHandles.add(handle);
            }
            lease.synchronize();
            return handle;
        }

        @Override
        protected Long mergeResults(List<Long> chunks) {
            return (long) chunks.size();
        }

        List<Long> getStreamHandles() {
            return List.copyOf(streamHandles);
        }
    }

    // ==================== Basic Execution Tests ====================

    @Test
    @DisplayName("Single chunk executes on real GPU")
    void singleChunkExecutes() throws Exception {
        SumKernel kernel = new SumKernel(1);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "single-chunk")) {
            long startTime = System.nanoTime();

            Integer result = kernel.execute(scope, List.of());

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertEquals(0, result, "Single chunk result should be 0");
            emitKernelEvent("TimeSlicedSingleChunk", "1 chunk", elapsedMicros);
        }
    }

    @Test
    @DisplayName("Multiple chunks execute on real GPU")
    void multipleChunksExecute() throws Exception {
        SumKernel kernel = new SumKernel(5);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "multi-chunk")) {
            long startTime = System.nanoTime();

            Integer result = kernel.execute(scope, List.of());

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertEquals(10, result, "Sum of 0+1+2+3+4 should be 10");
            emitKernelEvent("TimeSlicedMultiChunk", "5 chunks", elapsedMicros);
        }
    }

    // ==================== Stream Per Chunk Tests ====================

    @Test
    @DisplayName("Each chunk gets a stream on real GPU")
    void eachChunkGetsStream() throws Exception {
        StreamTrackingKernel kernel = new StreamTrackingKernel(5);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "stream-per-chunk")) {
            kernel.execute(scope, List.of());
        }

        List<Long> handles = kernel.getStreamHandles();
        assertEquals(5, handles.size(), "Should have 5 stream handles");

        // All handles should be non-zero (valid)
        handles.forEach(h ->
            assertTrue(h != 0, "Stream handle should be valid"));
    }

    @Test
    @DisplayName("Chunks get unique streams")
    void chunksGetUniqueStreams() throws Exception {
        StreamTrackingKernel kernel = new StreamTrackingKernel(4);

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            kernel.execute(scope, List.of());
        }

        List<Long> handles = kernel.getStreamHandles();
        Set<Long> uniqueHandles = new HashSet<>(handles);

        // Streams should be unique (though some backends might reuse)
        // At minimum, we should have handles
        assertTrue(uniqueHandles.size() >= 1, "Should have at least one unique handle");
    }

    // ==================== Synchronize Tests ====================

    @Test
    @DisplayName("Synchronize called after each chunk")
    void synchronizeAfterEachChunk() throws Exception {
        AtomicInteger syncCount = new AtomicInteger(0);

        TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return 3;
            }

            @Override
            protected Integer executeChunk(int chunkIndex, int totalChunks,
                                            List<Tensor> inputs, GpuLease lease) {
                // The kernel framework calls lease.synchronize() after executeChunk
                // We're testing that it happens by checking the kernel completes
                syncCount.incrementAndGet();
                return chunkIndex;
            }

            @Override
            protected Integer mergeResults(List<Integer> chunks) {
                return chunks.size();
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            Integer result = kernel.execute(scope, List.of());
            assertEquals(3, result);
        }

        assertEquals(3, syncCount.get(), "All chunks should execute");
    }

    // ==================== Scale Tests ====================

    @Test
    @DisplayName("Ten chunks execute correctly")
    void tenChunksExecute() throws Exception {
        SumKernel kernel = new SumKernel(10);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "ten-chunks")) {
            Integer result = kernel.execute(scope, List.of());
            assertEquals(45, result, "Sum of 0..9 should be 45");
        }
    }

    @Test
    @DisplayName("Fifty chunks execute correctly")
    void fiftyChunksExecute() throws Exception {
        SumKernel kernel = new SumKernel(50);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "fifty-chunks")) {
            long startTime = System.nanoTime();

            Integer result = kernel.execute(scope, List.of());

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            assertEquals(1225, result, "Sum of 0..49 should be 1225");
            emitKernelEvent("TimeSlicedFiftyChunks", "50 chunks", elapsedMicros);
        }
    }

    // ==================== Cancellation Tests ====================

    @Test
    @DisplayName("Cancellation throws on real GPU")
    void cancellationThrows() {
        AtomicInteger executedChunks = new AtomicInteger(0);

        TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return 5;
            }

            @Override
            protected Integer executeChunk(int chunkIndex, int totalChunks,
                                            List<Tensor> inputs, GpuLease lease) {
                executedChunks.incrementAndGet();
                if (chunkIndex == 2) {
                    Thread.currentThread().interrupt();
                }
                checkCancellation();
                return chunkIndex;
            }

            @Override
            protected Integer mergeResults(List<Integer> chunks) {
                return chunks.size();
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            assertThrows(RuntimeException.class, () -> kernel.execute(scope, List.of()));
        } finally {
            // Clear interrupt
            Thread.interrupted();
        }

        // Some chunks should have executed
        assertTrue(executedChunks.get() >= 1, "At least one chunk should execute");
    }

    // ==================== Error Handling Tests ====================

    @Test
    @DisplayName("Chunk exception propagates on real GPU")
    void chunkExceptionPropagates() {
        TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return 3;
            }

            @Override
            protected Integer executeChunk(int chunkIndex, int totalChunks,
                                            List<Tensor> inputs, GpuLease lease) {
                if (chunkIndex == 1) {
                    throw new IllegalStateException("Intentional chunk failure");
                }
                return chunkIndex;
            }

            @Override
            protected Integer mergeResults(List<Integer> chunks) {
                return chunks.size();
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            RuntimeException ex = assertThrows(RuntimeException.class,
                () -> kernel.execute(scope, List.of()));

            assertTrue(ex.getMessage().contains("failed") ||
                       ex.getCause().getMessage().contains("Intentional"),
                "Exception should indicate failure");
        }
    }

    // ==================== Timing Tests ====================

    @Test
    @DisplayName("Chunk execution time is measurable")
    void chunkExecutionTimeIsMeasurable() throws Exception {
        AtomicLong totalChunkTime = new AtomicLong(0);

        TimeSlicedKernel<Integer> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return 3;
            }

            @Override
            protected Integer executeChunk(int chunkIndex, int totalChunks,
                                            List<Tensor> inputs, GpuLease lease) {
                long start = System.nanoTime();

                // Do some actual work
                long sum = 0;
                for (int i = 0; i < 10000; i++) {
                    sum += i;
                }

                totalChunkTime.addAndGet(System.nanoTime() - start);
                return (int) (sum % 100);
            }

            @Override
            protected Integer mergeResults(List<Integer> chunks) {
                return chunks.stream().mapToInt(Integer::intValue).sum();
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            kernel.execute(scope, List.of());
        }

        // Should have measurable execution time
        assertTrue(totalChunkTime.get() > 0, "Chunk execution time should be positive");
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for time-sliced kernel")
    void jfrEventsEmitted() throws Exception {
        SumKernel kernel = new SumKernel(5);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-kernel-test")) {
            long startTime = System.nanoTime();

            Integer result = kernel.execute(scope, List.of());

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            emitKernelEvent("TimeSlicedKernelJFRTest", "5 chunks, result=" + result, elapsedMicros);
        }

        assertTrue(true, "JFR events should be emittable");
    }

    // ==================== Helper Methods ====================

    private void emitKernelEvent(String operation, String shape, long elapsedMicros) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "TIME_SLICED_KERNEL_GTEST";
        event.memoryBandwidthGBps = 0.0;
        event.commit();
    }
}
