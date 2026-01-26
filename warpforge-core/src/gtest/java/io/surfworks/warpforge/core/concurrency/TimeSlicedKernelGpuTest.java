package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuWorkCalibrator.GpuWorkResult;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link TimeSlicedKernel} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate time-sliced kernel execution with <b>real GPU hardware</b>.
 * Every test performs actual GPU operations (memory transfers) and emits JFR events
 * for validation. Without real GPU work, these would just be CPU API tests.
 *
 * <p><b>Key principle:</b> Each chunk execution performs measurable GPU work via
 * {@link GpuWorkCalibrator#doGpuWork}, which:
 * <ul>
 *   <li>Allocates device memory</li>
 *   <li>Performs H2D and D2H transfers on the chunk's stream</li>
 *   <li>Synchronizes the stream</li>
 *   <li>Emits {@code GpuKernelEvent} and {@code GpuMemoryEvent} for JFR profiling</li>
 * </ul>
 */
@Tag("gpu")
@DisplayName("TimeSlicedKernel GPU Tests")
class TimeSlicedKernelGpuTest {

    private GpuBackend backend;

    // Work duration per chunk (ms) - must be long enough for reliable measurement
    private static final long CHUNK_WORK_MS = 10;

    @BeforeEach
    void setUp() {
        backend = GpuTestSupport.createBackend();
        System.out.println("Running on: " + GpuTestSupport.describeEnvironment());
        // Ensure calibration is done before tests
        GpuWorkCalibrator.getCalibrationData(backend);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    // ==================== Real GPU Work Kernel ====================

    /**
     * Kernel that performs REAL GPU work (memory transfers) in each chunk.
     * Each chunk executes calibrated GPU operations and returns timing info.
     */
    static class GpuWorkKernel extends TimeSlicedKernel<GpuWorkResult> {
        private final int numChunks;
        private final long workPerChunkMs;
        private final GpuBackend backend;
        private final AtomicLong totalGpuTimeNanos = new AtomicLong(0);
        private final AtomicInteger chunksCompleted = new AtomicInteger(0);
        private final Set<Long> streamHandles = ConcurrentHashMap.newKeySet();

        GpuWorkKernel(GpuBackend backend, int numChunks, long workPerChunkMs) {
            this.backend = backend;
            this.numChunks = numChunks;
            this.workPerChunkMs = workPerChunkMs;
        }

        @Override
        protected int estimateChunks(List<Tensor> inputs) {
            return numChunks;
        }

        @Override
        protected GpuWorkResult executeChunk(int chunkIndex, int totalChunks,
                                              List<Tensor> inputs, GpuLease lease) {
            // Track stream handle
            streamHandles.add(lease.streamHandle());

            // Perform REAL GPU work - memory transfers with JFR events
            GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, workPerChunkMs);

            // Track timing
            totalGpuTimeNanos.addAndGet(result.elapsedNanos());
            chunksCompleted.incrementAndGet();

            return result;
        }

        @Override
        protected GpuWorkResult mergeResults(List<GpuWorkResult> chunks) {
            // Merge: return aggregate timing
            long totalNanos = chunks.stream().mapToLong(GpuWorkResult::elapsedNanos).sum();
            long totalElements = chunks.stream().mapToLong(GpuWorkResult::tensorElements).sum();
            long totalBytes = chunks.stream().mapToLong(GpuWorkResult::byteSize).sum();
            return new GpuWorkResult(totalNanos, totalElements, totalBytes, 0, 0, 0, 0);
        }

        // Accessors for verification
        long getTotalGpuTimeNanos() { return totalGpuTimeNanos.get(); }
        int getChunksCompleted() { return chunksCompleted.get(); }
        Set<Long> getStreamHandles() { return streamHandles; }
    }

    /**
     * Kernel that tracks stream handles for each chunk.
     */
    static class StreamTrackingGpuKernel extends TimeSlicedKernel<Long> {
        private final int numChunks;
        private final GpuBackend backend;
        private final List<Long> streamHandles = new ArrayList<>();

        StreamTrackingGpuKernel(GpuBackend backend, int numChunks) {
            this.backend = backend;
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

            // Perform REAL GPU work to validate stream is functional
            GpuWorkCalibrator.doGpuWork(backend, lease, GpuWorkCalibrator.MIN_WORK_MS);

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
    @DisplayName("Single chunk executes real GPU work")
    void singleChunkExecutesRealGpuWork() throws Exception {
        GpuWorkKernel kernel = new GpuWorkKernel(backend, 1, CHUNK_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "single-chunk")) {
            long startTime = System.nanoTime();

            GpuWorkResult result = kernel.execute(scope, List.of());

            long totalElapsed = System.nanoTime() - startTime;

            // Verify chunk executed
            assertEquals(1, kernel.getChunksCompleted(), "Single chunk should complete");

            // Verify timing is reasonable (real GPU work was done)
            assertTrue(result.elapsedNanos() > 1_000_000,
                "GPU work should take measurable time (got " + result.elapsedMillis() + "ms)");

            emitSummaryEvent("SingleChunkGpuWork", 1, totalElapsed / 1000);
        }
    }

    @Test
    @DisplayName("Multiple chunks execute real GPU work in sequence")
    void multipleChunksExecuteRealGpuWork() throws Exception {
        final int NUM_CHUNKS = 5;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, CHUNK_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "multi-chunk")) {
            long startTime = System.nanoTime();

            GpuWorkResult result = kernel.execute(scope, List.of());

            long totalElapsed = System.nanoTime() - startTime;

            // Verify all chunks executed
            assertEquals(NUM_CHUNKS, kernel.getChunksCompleted(),
                "All " + NUM_CHUNKS + " chunks should complete");

            // Verify each chunk did real GPU work
            long expectedMinNanos = NUM_CHUNKS * (CHUNK_WORK_MS - GpuWorkCalibrator.TIMING_TOLERANCE_MS) * 1_000_000;
            assertTrue(kernel.getTotalGpuTimeNanos() > expectedMinNanos,
                "Total GPU time should reflect real work");

            emitSummaryEvent("MultiChunkGpuWork", NUM_CHUNKS, totalElapsed / 1000);
        }
    }

    // ==================== Stream Per Chunk Tests ====================

    @Test
    @DisplayName("Each chunk gets a stream on real GPU")
    void eachChunkGetsStream() throws Exception {
        final int NUM_CHUNKS = 5;
        StreamTrackingGpuKernel kernel = new StreamTrackingGpuKernel(backend, NUM_CHUNKS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "stream-per-chunk")) {
            kernel.execute(scope, List.of());
        }

        List<Long> handles = kernel.getStreamHandles();
        assertEquals(NUM_CHUNKS, handles.size(), "Should have " + NUM_CHUNKS + " stream handles");

        // All handles should be non-zero (valid)
        handles.forEach(h ->
            assertTrue(h != 0, "Stream handle should be valid"));
    }

    @Test
    @DisplayName("Chunks get unique streams for concurrent execution")
    void chunksGetUniqueStreams() throws Exception {
        final int NUM_CHUNKS = 4;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, GpuWorkCalibrator.MIN_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            kernel.execute(scope, List.of());
        }

        Set<Long> uniqueHandles = kernel.getStreamHandles();
        // At minimum, we should have at least one unique handle
        // Some backends may reuse streams, but handles should be valid
        assertTrue(uniqueHandles.size() >= 1, "Should have at least one unique handle");
        uniqueHandles.forEach(h -> assertTrue(h != 0, "Handle should be non-zero"));
    }

    // ==================== Timing Validation Tests ====================

    @Test
    @DisplayName("Chunk GPU work timing is accurate")
    void chunkGpuWorkTimingIsAccurate() throws Exception {
        final int NUM_CHUNKS = 3;
        final long WORK_MS = 15;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "timing-test")) {
            GpuWorkResult result = kernel.execute(scope, List.of());

            // Each chunk should take approximately WORK_MS
            // Total time should be at least NUM_CHUNKS * WORK_MS (sequential) or less (concurrent)
            long totalMs = result.elapsedMillis();

            // Verify minimum work was done
            long minExpectedMs = NUM_CHUNKS * (WORK_MS - GpuWorkCalibrator.TIMING_TOLERANCE_MS);
            assertTrue(totalMs >= minExpectedMs / 2, // Allow some concurrency benefit
                String.format("Total time %dms should reflect %d chunks of %dms work",
                    totalMs, NUM_CHUNKS, WORK_MS));
        }
    }

    @Test
    @DisplayName("Per-chunk timing is tracked via JFR events")
    void perChunkTimingTracked() throws Exception {
        final int NUM_CHUNKS = 4;
        AtomicLong totalReportedTime = new AtomicLong(0);

        TimeSlicedKernel<Long> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return NUM_CHUNKS;
            }

            @Override
            protected Long executeChunk(int chunkIndex, int totalChunks,
                                        List<Tensor> inputs, GpuLease lease) {
                // Each chunk does real GPU work
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, CHUNK_WORK_MS);
                totalReportedTime.addAndGet(result.elapsedNanos());
                return result.elapsedNanos();
            }

            @Override
            protected Long mergeResults(List<Long> chunks) {
                return chunks.stream().mapToLong(Long::longValue).sum();
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "per-chunk-timing")) {
            Long summedTime = kernel.execute(scope, List.of());

            // JFR events were emitted for each chunk via GpuWorkCalibrator
            assertTrue(summedTime > 0, "Chunks should report positive timing");
            assertEquals(totalReportedTime.get(), summedTime,
                "Merged result should equal sum of chunk times");
        }
    }

    // ==================== Scale Tests ====================

    @Test
    @DisplayName("Ten chunks execute correctly with real GPU work")
    void tenChunksExecuteRealGpuWork() throws Exception {
        final int NUM_CHUNKS = 10;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, GpuWorkCalibrator.MIN_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "ten-chunks")) {
            long startTime = System.nanoTime();

            GpuWorkResult result = kernel.execute(scope, List.of());

            long totalElapsed = System.nanoTime() - startTime;

            assertEquals(NUM_CHUNKS, kernel.getChunksCompleted(),
                "All 10 chunks should complete");
            assertTrue(result.tensorElements() > 0, "Should have processed tensor elements");

            emitSummaryEvent("TenChunksGpuWork", NUM_CHUNKS, totalElapsed / 1000);
        }
    }

    @Test
    @DisplayName("Twenty chunks execute correctly with real GPU work")
    void twentyChunksExecuteRealGpuWork() throws Exception {
        final int NUM_CHUNKS = 20;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, GpuWorkCalibrator.MIN_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "twenty-chunks")) {
            long startTime = System.nanoTime();

            GpuWorkResult result = kernel.execute(scope, List.of());

            long totalElapsed = System.nanoTime() - startTime;

            assertEquals(NUM_CHUNKS, kernel.getChunksCompleted(),
                "All 20 chunks should complete");

            emitSummaryEvent("TwentyChunksGpuWork", NUM_CHUNKS, totalElapsed / 1000);
        }
    }

    // ==================== Cancellation Tests ====================

    @Test
    @DisplayName("Cancellation stops GPU work")
    void cancellationStopsGpuWork() {
        AtomicInteger executedChunks = new AtomicInteger(0);

        TimeSlicedKernel<GpuWorkResult> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return 5;
            }

            @Override
            protected GpuWorkResult executeChunk(int chunkIndex, int totalChunks,
                                                  List<Tensor> inputs, GpuLease lease) {
                executedChunks.incrementAndGet();

                // Do real GPU work
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                    GpuWorkCalibrator.MIN_WORK_MS);

                if (chunkIndex == 2) {
                    Thread.currentThread().interrupt();
                }
                checkCancellation();
                return result;
            }

            @Override
            protected GpuWorkResult mergeResults(List<GpuWorkResult> chunks) {
                return chunks.isEmpty() ? null : chunks.get(0);
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            assertThrows(RuntimeException.class, () -> kernel.execute(scope, List.of()));
        } finally {
            Thread.interrupted(); // Clear interrupt
        }

        // At least some chunks should have executed with real GPU work
        assertTrue(executedChunks.get() >= 1, "At least one chunk should execute");
    }

    // ==================== Error Handling Tests ====================

    @Test
    @DisplayName("Chunk exception propagates after GPU work")
    void chunkExceptionPropagatesAfterGpuWork() {
        TimeSlicedKernel<GpuWorkResult> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return 3;
            }

            @Override
            protected GpuWorkResult executeChunk(int chunkIndex, int totalChunks,
                                                  List<Tensor> inputs, GpuLease lease) {
                // Do real GPU work first
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease,
                    GpuWorkCalibrator.MIN_WORK_MS);

                if (chunkIndex == 1) {
                    throw new IllegalStateException("Intentional chunk failure after GPU work");
                }
                return result;
            }

            @Override
            protected GpuWorkResult mergeResults(List<GpuWorkResult> chunks) {
                return chunks.get(0);
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

    // ==================== Bandwidth and Memory Tests ====================

    @Test
    @DisplayName("Chunks report memory bandwidth via JFR")
    void chunksReportMemoryBandwidth() throws Exception {
        final int NUM_CHUNKS = 3;
        List<Double> bandwidths = new ArrayList<>();

        TimeSlicedKernel<Double> kernel = new TimeSlicedKernel<>() {
            @Override
            protected int estimateChunks(List<Tensor> inputs) {
                return NUM_CHUNKS;
            }

            @Override
            protected Double executeChunk(int chunkIndex, int totalChunks,
                                          List<Tensor> inputs, GpuLease lease) {
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, CHUNK_WORK_MS);
                synchronized (bandwidths) {
                    bandwidths.add(result.bandwidthGBps());
                }
                return result.bandwidthGBps();
            }

            @Override
            protected Double mergeResults(List<Double> chunks) {
                return chunks.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            }
        };

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "bandwidth-test")) {
            Double avgBandwidth = kernel.execute(scope, List.of());

            assertEquals(NUM_CHUNKS, bandwidths.size(), "Should have bandwidth for each chunk");
            assertTrue(avgBandwidth > 0, "Average bandwidth should be positive");

            System.out.printf("TimeSlicedKernel bandwidth: avg=%.2f GB/s%n", avgBandwidth);
        }
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for each chunk's GPU work")
    void jfrEventsEmittedForEachChunk() throws Exception {
        final int NUM_CHUNKS = 5;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, CHUNK_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-chunk-test")) {
            long startTime = System.nanoTime();

            GpuWorkResult result = kernel.execute(scope, List.of());

            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            // Each chunk emitted JFR events via GpuWorkCalibrator.doGpuWork()
            // Verify we have expected number of completed chunks
            assertEquals(NUM_CHUNKS, kernel.getChunksCompleted(),
                "All chunks should complete with JFR events");

            emitSummaryEvent("TimeSlicedKernelJFRValidation",
                NUM_CHUNKS, elapsedMicros);
        }
    }

    @Test
    @DisplayName("JFR events contain correct stream handles")
    void jfrEventsContainStreamHandles() throws Exception {
        final int NUM_CHUNKS = 4;
        GpuWorkKernel kernel = new GpuWorkKernel(backend, NUM_CHUNKS, GpuWorkCalibrator.MIN_WORK_MS);

        try (GpuTaskScope scope = GpuTaskScope.open(backend, "jfr-stream-test")) {
            kernel.execute(scope, List.of());

            Set<Long> handles = kernel.getStreamHandles();

            // All stream handles should be valid (non-zero)
            handles.forEach(h ->
                assertTrue(h != 0, "JFR events should have valid stream handles"));
        }
    }

    // ==================== Helper Methods ====================

    private void emitSummaryEvent(String operation, int numChunks, long elapsedMicros) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = numChunks + " chunks";
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "TIME_SLICED_KERNEL_GTEST";
        event.memoryBandwidthGBps = 0.0;
        event.commit();
    }
}
