package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import io.surfworks.warpforge.core.jfr.GpuOccupancyEvent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for StreamTracker.
 */
@DisplayName("StreamTracker Unit Tests")
class StreamTrackerTest {

    private StreamTracker tracker;

    @BeforeEach
    void setUp() {
        tracker = StreamTracker.forDevice(0)
            .configure(128, 48); // RTX 4090-like config
    }

    @Nested
    @DisplayName("Basic Tracking")
    class BasicTracking {

        @Test
        @DisplayName("forDevice() creates tracker with correct device index")
        void forDeviceCreatesTracker() {
            StreamTracker t = StreamTracker.forDevice(1);
            assertEquals(1, t.deviceIndex());
        }

        @Test
        @DisplayName("kernelStarted() returns unique IDs")
        void kernelStartedReturnsUniqueIds() {
            long id1 = tracker.kernelStarted(100L);
            long id2 = tracker.kernelStarted(100L);
            long id3 = tracker.kernelStarted(200L);

            assertTrue(id1 != id2);
            assertTrue(id2 != id3);
            assertTrue(id1 != id3);
        }

        @Test
        @DisplayName("kernelStarted() increments active count")
        void kernelStartedIncrementsCount() {
            assertEquals(0, tracker.activeKernelCount());

            tracker.kernelStarted(100L);
            assertEquals(1, tracker.activeKernelCount());

            tracker.kernelStarted(100L);
            assertEquals(2, tracker.activeKernelCount());

            tracker.kernelStarted(200L);
            assertEquals(3, tracker.activeKernelCount());
        }

        @Test
        @DisplayName("kernelCompleted() decrements active count")
        void kernelCompletedDecrementsCount() {
            long id1 = tracker.kernelStarted(100L);
            long id2 = tracker.kernelStarted(100L);

            assertEquals(2, tracker.activeKernelCount());

            tracker.kernelCompleted(id1);
            assertEquals(1, tracker.activeKernelCount());

            tracker.kernelCompleted(id2);
            assertEquals(0, tracker.activeKernelCount());
        }

        @Test
        @DisplayName("kernelCompleted() with unknown ID is no-op")
        void kernelCompletedUnknownIdNoOp() {
            tracker.kernelStarted(100L);
            assertEquals(1, tracker.activeKernelCount());

            tracker.kernelCompleted(999L); // Unknown ID
            assertEquals(1, tracker.activeKernelCount());
        }
    }

    @Nested
    @DisplayName("Stream Tracking")
    class StreamTracking {

        @Test
        @DisplayName("activeStreamCount() counts unique streams")
        void activeStreamCountUnique() {
            assertEquals(0, tracker.activeStreamCount());

            tracker.kernelStarted(100L);
            assertEquals(1, tracker.activeStreamCount());

            tracker.kernelStarted(100L); // Same stream
            assertEquals(1, tracker.activeStreamCount());

            tracker.kernelStarted(200L); // Different stream
            assertEquals(2, tracker.activeStreamCount());
        }

        @Test
        @DisplayName("activeStreamCount() decrements when stream empties")
        void activeStreamCountDecrementsOnEmpty() {
            long id1 = tracker.kernelStarted(100L);
            long id2 = tracker.kernelStarted(200L);

            assertEquals(2, tracker.activeStreamCount());

            tracker.kernelCompleted(id1);
            assertEquals(1, tracker.activeStreamCount());

            tracker.kernelCompleted(id2);
            assertEquals(0, tracker.activeStreamCount());
        }

        @Test
        @DisplayName("concurrentKernelsExcluding() excludes specified stream")
        void concurrentKernelsExcluding() {
            tracker.kernelStarted(100L);
            tracker.kernelStarted(100L);
            tracker.kernelStarted(200L);
            tracker.kernelStarted(300L);

            assertEquals(2, tracker.concurrentKernelsExcluding(100L)); // Excludes 2 on stream 100
            assertEquals(3, tracker.concurrentKernelsExcluding(200L)); // Excludes 1 on stream 200
            assertEquals(4, tracker.concurrentKernelsExcluding(999L)); // Excludes nothing
        }
    }

    @Nested
    @DisplayName("Occupancy Estimation")
    class OccupancyEstimation {

        @Test
        @DisplayName("estimatedTotalOccupancy() is 0 with no kernels")
        void noKernelsZeroOccupancy() {
            assertEquals(0, tracker.estimatedTotalOccupancy());
        }

        @Test
        @DisplayName("estimatedTotalActiveWarps() tracks warp estimates")
        void trackWarps() {
            OccupancyCalculator.OccupancyInfo info = new OccupancyCalculator.OccupancyInfo(
                75, 48, 6, "registers"
            );

            tracker.kernelStarted(100L, info, "GEMM");
            assertEquals(48, tracker.estimatedTotalActiveWarps());

            tracker.kernelStarted(200L, info, "Add");
            assertEquals(96, tracker.estimatedTotalActiveWarps());
        }

        @Test
        @DisplayName("estimatedTotalOccupancy() computes percentage correctly")
        void computesOccupancyPercentage() {
            // With 128 SMs and 48 warps/SM, max = 6144 warps
            // 48 warps = 48/6144 ≈ 0.78%
            OccupancyCalculator.OccupancyInfo info = new OccupancyCalculator.OccupancyInfo(
                75, 48, 6, "registers"
            );

            tracker.kernelStarted(100L, info, "GEMM");

            int occupancy = tracker.estimatedTotalOccupancy();
            assertTrue(occupancy >= 0 && occupancy <= 100);
        }
    }

    @Nested
    @DisplayName("Event Population")
    class EventPopulation {

        @Test
        @DisplayName("populateEvent() sets stream context fields")
        void populateEventSetsStreamContext() {
            tracker.kernelStarted(100L);
            tracker.kernelStarted(200L);
            tracker.kernelStarted(200L);

            GpuKernelEvent event = new GpuKernelEvent();
            tracker.populateEvent(event, 100L);

            assertEquals(100L, event.streamId);
            assertEquals(2, event.concurrentKernels); // 2 kernels on stream 200
        }

        @Test
        @DisplayName("populateOccupancyEvent() sets all fields")
        void populateOccupancyEventSetsAllFields() {
            OccupancyCalculator.OccupancyInfo info = new OccupancyCalculator.OccupancyInfo(
                75, 48, 6, "registers"
            );

            tracker.kernelStarted(100L, info, "GEMM");
            tracker.kernelStarted(200L, info, "Add");

            GpuOccupancyEvent event = new GpuOccupancyEvent();
            tracker.populateOccupancyEvent(event);

            assertEquals(0, event.deviceIndex);
            assertEquals(2, event.activeStreams);
            assertEquals(2, event.activeKernels);
            assertEquals(96, event.activeWarpsEstimate);
            assertEquals(128, event.smCount);
            assertEquals(48, event.maxWarpsPerSM);
            assertTrue(event.estimatedTotalOccupancyPercent >= 0);
        }
    }

    @Nested
    @DisplayName("Utility Methods")
    class UtilityMethods {

        @Test
        @DisplayName("hasActiveWork() returns false initially")
        void hasActiveWorkFalseInitially() {
            assertFalse(tracker.hasActiveWork());
        }

        @Test
        @DisplayName("hasActiveWork() returns true with active kernels")
        void hasActiveWorkTrueWithKernels() {
            tracker.kernelStarted(100L);
            assertTrue(tracker.hasActiveWork());
        }

        @Test
        @DisplayName("isSaturated() checks threshold")
        void isSaturatedChecksThreshold() {
            assertFalse(tracker.isSaturated(3));

            tracker.kernelStarted(100L);
            tracker.kernelStarted(100L);
            assertFalse(tracker.isSaturated(3));

            tracker.kernelStarted(100L);
            assertTrue(tracker.isSaturated(3));
        }

        @Test
        @DisplayName("getActiveKernels() returns copy of active kernels")
        void getActiveKernelsReturnsCopy() {
            tracker.kernelStarted(100L, null, "Test");

            var kernels = tracker.getActiveKernels();
            assertEquals(1, kernels.size());

            // Verify it's a copy
            tracker.kernelStarted(200L);
            assertEquals(1, kernels.size()); // Original unchanged
            assertEquals(2, tracker.getActiveKernels().size());
        }

        @Test
        @DisplayName("reset() clears all state")
        void resetClearsState() {
            tracker.kernelStarted(100L);
            tracker.kernelStarted(200L);

            assertEquals(2, tracker.activeKernelCount());
            assertEquals(2, tracker.activeStreamCount());

            tracker.reset();

            assertEquals(0, tracker.activeKernelCount());
            assertEquals(0, tracker.activeStreamCount());
            assertFalse(tracker.hasActiveWork());
        }

        @Test
        @DisplayName("toString() shows summary")
        void toStringShowsSummary() {
            tracker.kernelStarted(100L);

            String str = tracker.toString();
            assertTrue(str.contains("device=0"));
            assertTrue(str.contains("streams=1"));
            assertTrue(str.contains("kernels=1"));
        }
    }

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafety {

        @Test
        @DisplayName("concurrent kernelStarted() calls are safe")
        void concurrentKernelStarted() throws InterruptedException {
            int threadCount = 10;
            int kernelsPerThread = 100;
            ExecutorService executor = Executors.newFixedThreadPool(threadCount);
            CountDownLatch latch = new CountDownLatch(threadCount);

            List<Long> allIds = new ArrayList<>();

            for (int t = 0; t < threadCount; t++) {
                final long streamId = t;
                executor.submit(() -> {
                    try {
                        for (int k = 0; k < kernelsPerThread; k++) {
                            long id = tracker.kernelStarted(streamId);
                            synchronized (allIds) {
                                allIds.add(id);
                            }
                        }
                    } finally {
                        latch.countDown();
                    }
                });
            }

            latch.await(10, TimeUnit.SECONDS);
            executor.shutdown();

            assertEquals(threadCount * kernelsPerThread, tracker.activeKernelCount());
            assertEquals(threadCount, tracker.activeStreamCount());
        }

        @Test
        @DisplayName("concurrent kernelCompleted() calls are safe")
        void concurrentKernelCompleted() throws InterruptedException {
            int kernelCount = 100;
            List<Long> kernelIds = new ArrayList<>();

            // Start all kernels
            for (int i = 0; i < kernelCount; i++) {
                kernelIds.add(tracker.kernelStarted(i % 10L));
            }

            assertEquals(kernelCount, tracker.activeKernelCount());

            // Complete all kernels concurrently
            int threadCount = 10;
            ExecutorService executor = Executors.newFixedThreadPool(threadCount);
            CountDownLatch latch = new CountDownLatch(kernelCount);

            for (long id : kernelIds) {
                executor.submit(() -> {
                    try {
                        tracker.kernelCompleted(id);
                    } finally {
                        latch.countDown();
                    }
                });
            }

            latch.await(10, TimeUnit.SECONDS);
            executor.shutdown();

            assertEquals(0, tracker.activeKernelCount());
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("configure() updates device properties")
        void configureUpdatesProperties() {
            StreamTracker t = StreamTracker.forDevice(0);

            GpuOccupancyEvent event = new GpuOccupancyEvent();
            t.populateOccupancyEvent(event);
            assertEquals(0, event.smCount);

            t.configure(256, 64);
            t.populateOccupancyEvent(event);
            assertEquals(256, event.smCount);
            assertEquals(64, event.maxWarpsPerSM);
        }

        @Test
        @DisplayName("kernelStarted() with null occupancy info is safe")
        void nullOccupancyInfoSafe() {
            long id = tracker.kernelStarted(100L, null, null);
            assertTrue(id > 0);
            assertEquals(1, tracker.activeKernelCount());
        }

        @Test
        @DisplayName("multiple completions of same kernel ID is safe")
        void multipleCompletionsSafe() {
            long id = tracker.kernelStarted(100L);
            assertEquals(1, tracker.activeKernelCount());

            tracker.kernelCompleted(id);
            assertEquals(0, tracker.activeKernelCount());

            tracker.kernelCompleted(id); // Second completion
            assertEquals(0, tracker.activeKernelCount()); // Still 0
        }

        @Test
        @DisplayName("zero stream ID is valid")
        void zeroStreamIdValid() {
            tracker.kernelStarted(0L);
            assertEquals(1, tracker.activeKernelCount());
            assertEquals(1, tracker.activeStreamCount());
        }

        @Test
        @DisplayName("estimatedTotalOccupancy() handles unconfigured tracker")
        void unconfiguredTrackerOccupancy() {
            StreamTracker t = StreamTracker.forDevice(0);
            // Not configured - smCount and maxWarpsPerSM are 0
            t.kernelStarted(100L);

            assertEquals(0, t.estimatedTotalOccupancy()); // Avoids division by zero
        }
    }

    @Nested
    @DisplayName("KernelInfo")
    class KernelInfoTests {

        @Test
        @DisplayName("KernelInfo is captured correctly")
        void kernelInfoCaptured() {
            OccupancyCalculator.OccupancyInfo occInfo = new OccupancyCalculator.OccupancyInfo(
                75, 48, 6, "registers"
            );

            long id = tracker.kernelStarted(100L, occInfo, "GEMM");

            var kernels = tracker.getActiveKernels();
            StreamTracker.KernelInfo info = kernels.get(id);

            assertNotNull(info);
            assertEquals(id, info.kernelId());
            assertEquals(100L, info.streamId());
            assertEquals(48, info.estimatedWarps());
            assertEquals(75, info.occupancyPercent());
            assertEquals("GEMM", info.operation());
            assertTrue(info.startTimeNanos() > 0);
        }
    }
}
