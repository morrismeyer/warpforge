package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryProfilerTest {

    @Nested
    class BasicProfileTests {

        @Test
        void testStartAndStop() {
            MemoryProfiler profiler = new MemoryProfiler();
            profiler.start();
            MemoryProfiler.MemorySnapshot snapshot = profiler.stop();

            assertNotNull(snapshot);
            assertTrue(snapshot.sampleCount() >= 0);
        }

        @Test
        void testCannotStartTwice() {
            MemoryProfiler profiler = new MemoryProfiler();
            profiler.start();

            assertThrows(IllegalStateException.class, profiler::start);

            profiler.stop();
        }

        @Test
        void testCannotStopWithoutStart() {
            MemoryProfiler profiler = new MemoryProfiler();
            assertThrows(IllegalStateException.class, profiler::stop);
        }
    }

    @Nested
    class MeasurementTests {

        @Test
        void testMeasureAllocations() {
            MemoryProfiler.MemorySnapshot snapshot = MemoryProfiler.measure(() -> {
                // Allocate some memory
                List<byte[]> allocations = new ArrayList<>();
                for (int i = 0; i < 10; i++) {
                    allocations.add(new byte[1024 * 1024]); // 1MB each
                }
                // Keep reference to prevent GC
                assertTrue(allocations.size() == 10);
            });

            assertNotNull(snapshot);
            assertTrue(snapshot.peakHeapUsed() > 0);
        }

        @Test
        void testQuickSnapshot() {
            MemoryProfiler.MemorySnapshot snapshot = MemoryProfiler.quickSnapshot();

            assertNotNull(snapshot);
            assertTrue(snapshot.peakHeapUsed() > 0);
            assertEquals(1, snapshot.sampleCount());
        }
    }

    @Nested
    class StaticMethodTests {

        @Test
        void testCurrentHeapUsed() {
            long used = MemoryProfiler.currentHeapUsed();
            assertTrue(used > 0);
        }

        @Test
        void testMaxHeap() {
            long max = MemoryProfiler.maxHeap();
            assertTrue(max > 0);
        }

        @Test
        void testCurrentNonHeapUsed() {
            long used = MemoryProfiler.currentNonHeapUsed();
            assertTrue(used >= 0);
        }
    }

    @Nested
    class SnapshotTests {

        @Test
        void testTotalPeakUsed() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1000, 100, 10, 1000000
            );

            assertEquals(3300, snapshot.totalPeakUsed());
        }

        @Test
        void testTotalAllocated() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1500, 200, 10, 1000000
            );

            assertEquals(1700, snapshot.totalAllocated());
        }

        @Test
        void testHeapUtilization() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 5000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1000, 100, 10, 1000000
            );

            assertEquals(50.0, snapshot.heapUtilization(), 0.01);
        }

        @Test
        void testAllocationRate() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1_000_000_000L, 100, 10, 1_000_000_000L  // 1GB in 1 second
            );

            assertEquals(1_000_000_000.0, snapshot.allocationRateBytesPerSec(), 1.0);
        }

        @Test
        void testSummary() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1000, 100, 10, 1000000
            );

            String summary = snapshot.summary();
            assertNotNull(summary);
            assertTrue(summary.contains("Memory Profile"));
            assertTrue(summary.contains("Heap"));
        }

        @Test
        void testToMap() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1000, 100, 10, 1000000
            );

            var map = snapshot.toMap();
            assertNotNull(map);
            assertEquals(3000L, map.get("peak_heap_bytes"));
            assertEquals(2500L, map.get("avg_heap_bytes"));
        }

        @Test
        void testToMemoryStats() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 500, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1000, 100, 10, 1000000
            );

            BenchmarkResult.MemoryStats stats = snapshot.toMemoryStats();
            assertEquals(3000, stats.peakUsageBytes());
            assertEquals(1000, stats.allocatedBytes());
        }
    }

    @Nested
    class SampleTests {

        @Test
        void testManualSample() {
            MemoryProfiler profiler = new MemoryProfiler();
            profiler.start();

            MemoryProfiler.MemorySample sample = profiler.sample();

            assertNotNull(sample);
            assertTrue(sample.heapUsed() > 0);
            assertTrue(sample.timestampNanos() >= 0);

            profiler.stop();
        }
    }
}
