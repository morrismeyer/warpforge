package io.surfworks.warpforge.data.golden;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GoldenComparisonTest {

    @Nested
    class BasicComparisonTests {

        @Test
        void testExactMatch() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{2, 3},
                        new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
                TensorView t2 = createFloatTensor(arena, new long[]{2, 3},
                        new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                assertTrue(result.matches());
                assertEquals(6, result.totalElements());
                assertEquals(0, result.mismatchCount());
                assertEquals(0.0, result.maxDifference(), 1e-10);
            }
        }

        @Test
        void testWithinTolerance() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView expected = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, 2.0f, 3.0f});
                TensorView actual = createFloatTensor(arena, new long[]{3},
                        new float[]{1.00001f, 2.00001f, 3.00001f});

                ComparisonResult result = GoldenComparison.compare(expected, actual, 1e-4);

                assertTrue(result.matches());
                assertEquals(3, result.totalElements());
                assertTrue(result.maxDifference() < 1e-4);
            }
        }

        @Test
        void testExceedsTolerance() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView expected = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, 2.0f, 3.0f});
                TensorView actual = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, 2.1f, 3.0f}); // 0.1 difference at index 1

                ComparisonResult result = GoldenComparison.compare(expected, actual, 1e-4);

                assertFalse(result.matches());
                assertEquals(3, result.totalElements());
                assertEquals(1, result.mismatchCount());
                assertTrue(result.maxDifference() >= 0.09);

                // Check mismatch details
                assertEquals(1, result.firstMismatches().size());
                assertEquals(1, result.firstMismatches().get(0).indices()[0]);
            }
        }

        @Test
        void testMultipleMismatches() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView expected = createFloatTensor(arena, new long[]{5},
                        new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
                TensorView actual = createFloatTensor(arena, new long[]{5},
                        new float[]{1.5f, 2.0f, 3.5f, 4.0f, 5.5f}); // 3 mismatches

                ComparisonResult result = GoldenComparison.compare(expected, actual, 0.1);

                assertFalse(result.matches());
                assertEquals(3, result.mismatchCount());
                assertEquals(3, result.firstMismatches().size());
            }
        }
    }

    @Nested
    class ShapeAndDTypeTests {

        @Test
        void testShapeMismatch() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{2, 3},
                        new float[]{1, 2, 3, 4, 5, 6});
                TensorView t2 = createFloatTensor(arena, new long[]{3, 2},
                        new float[]{1, 2, 3, 4, 5, 6});

                ComparisonResult result = GoldenComparison.compare(t1, t2);

                assertFalse(result.matches());
                assertTrue(result.shapeMismatch());
                assertTrue(result.summary().contains("SHAPE MISMATCH"));
            }
        }

        @Test
        void testDifferentRankShapes() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{6},
                        new float[]{1, 2, 3, 4, 5, 6});
                TensorView t2 = createFloatTensor(arena, new long[]{2, 3},
                        new float[]{1, 2, 3, 4, 5, 6});

                ComparisonResult result = GoldenComparison.compare(t1, t2);

                assertFalse(result.matches());
                assertTrue(result.shapeMismatch());
            }
        }
    }

    @Nested
    class RelativeToleranceTests {

        @Test
        void testRelativeToleranceSmallValues() {
            try (Arena arena = Arena.ofConfined()) {
                // For small values, relative tolerance is more lenient
                TensorView expected = createFloatTensor(arena, new long[]{2},
                        new float[]{0.001f, 0.002f});
                TensorView actual = createFloatTensor(arena, new long[]{2},
                        new float[]{0.00101f, 0.00201f}); // 1% relative error

                // Absolute tolerance of 1e-5 would fail, but relative tolerance of 1% passes
                ComparisonResult result = GoldenComparison.compareWithRelative(expected, actual, 1e-6, 0.01);

                assertTrue(result.matches());
            }
        }

        @Test
        void testRelativeToleranceLargeValues() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView expected = createFloatTensor(arena, new long[]{2},
                        new float[]{1000.0f, 2000.0f});
                TensorView actual = createFloatTensor(arena, new long[]{2},
                        new float[]{1001.0f, 2002.0f}); // ~0.1% relative error

                ComparisonResult result = GoldenComparison.compareWithRelative(expected, actual, 0.1, 0.001);

                assertTrue(result.matches());
            }
        }
    }

    @Nested
    class StatisticsTests {

        @Test
        void testTensorStats() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView tensor = createFloatTensor(arena, new long[]{5},
                        new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

                GoldenComparison.TensorStats stats = GoldenComparison.stats(tensor);

                assertEquals(5, stats.count());
                assertEquals(1.0, stats.min(), 1e-6);
                assertEquals(5.0, stats.max(), 1e-6);
                assertEquals(3.0, stats.mean(), 1e-6);

                // std of [1,2,3,4,5] = sqrt(2) ≈ 1.414
                assertEquals(Math.sqrt(2), stats.std(), 0.001);
            }
        }

        @Test
        void testEmptyTensorStats() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment empty = arena.allocate(0);
                TensorInfo info = new TensorInfo("empty", DType.F32, new long[]{0}, 0, 0);
                TensorView tensor = new TensorView(empty, info);

                GoldenComparison.TensorStats stats = GoldenComparison.stats(tensor);

                assertEquals(0, stats.count());
            }
        }
    }

    @Nested
    class NaNHandlingTests {

        @Test
        void testNaNMatchesNaN() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, Float.NaN, 3.0f});
                TensorView t2 = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, Float.NaN, 3.0f});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                assertTrue(result.matches());
            }
        }

        @Test
        void testNaNDoesNotMatchNumber() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, Float.NaN, 3.0f});
                TensorView t2 = createFloatTensor(arena, new long[]{3},
                        new float[]{1.0f, 2.0f, 3.0f});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                assertFalse(result.matches());
                assertEquals(1, result.mismatchCount());
            }
        }
    }

    @Nested
    class SummaryTests {

        @Test
        void testMatchSummary() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{3}, new float[]{1, 2, 3});
                TensorView t2 = createFloatTensor(arena, new long[]{3}, new float[]{1, 2, 3});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                String summary = result.summary();
                assertTrue(summary.startsWith("MATCH"));
                assertTrue(summary.contains("3 elements"));
            }
        }

        @Test
        void testMismatchSummary() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{3}, new float[]{1, 2, 3});
                TensorView t2 = createFloatTensor(arena, new long[]{3}, new float[]{1, 5, 3});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                String summary = result.summary();
                assertTrue(summary.startsWith("MISMATCH"));
                assertTrue(summary.contains("1/3"));
            }
        }

        @Test
        void testDetailedReport() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{3}, new float[]{1, 2, 3});
                TensorView t2 = createFloatTensor(arena, new long[]{3}, new float[]{1, 5, 3});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                String report = result.detailedReport();
                assertTrue(report.contains("MISMATCH"));
                assertTrue(report.contains("expected="));
                assertTrue(report.contains("actual="));
            }
        }

        @Test
        void testMatchPercentage() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{4}, new float[]{1, 2, 3, 4});
                TensorView t2 = createFloatTensor(arena, new long[]{4}, new float[]{1, 5, 3, 5});

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                assertEquals(50.0, result.matchPercentage(), 1e-6);
            }
        }
    }

    @Nested
    class MultiDimensionalTests {

        @Test
        void test2DTensor() {
            try (Arena arena = Arena.ofConfined()) {
                TensorView t1 = createFloatTensor(arena, new long[]{2, 3},
                        new float[]{1, 2, 3, 4, 5, 6});
                TensorView t2 = createFloatTensor(arena, new long[]{2, 3},
                        new float[]{1, 2, 3, 4, 5, 10}); // Mismatch at [1,2]

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                assertFalse(result.matches());
                assertEquals(1, result.mismatchCount());

                // Check mismatch location
                long[] indices = result.firstMismatches().get(0).indices();
                assertEquals(2, indices.length);
                assertEquals(1, indices[0]);
                assertEquals(2, indices[1]);
            }
        }

        @Test
        void test3DTensor() {
            try (Arena arena = Arena.ofConfined()) {
                float[] data = new float[24];
                for (int i = 0; i < 24; i++) data[i] = i;

                TensorView t1 = createFloatTensor(arena, new long[]{2, 3, 4}, data);

                float[] data2 = data.clone();
                data2[23] = 100; // Mismatch at [1,2,3]
                TensorView t2 = createFloatTensor(arena, new long[]{2, 3, 4}, data2);

                ComparisonResult result = GoldenComparison.compare(t1, t2, 1e-6);

                assertFalse(result.matches());
                assertEquals(1, result.mismatchCount());

                long[] indices = result.firstMismatches().get(0).indices();
                assertEquals(3, indices.length);
                assertEquals(1, indices[0]);
                assertEquals(2, indices[1]);
                assertEquals(3, indices[2]);
            }
        }
    }

    // Helper method to create test tensors
    private TensorView createFloatTensor(Arena arena, long[] shape, float[] data) {
        MemorySegment segment = arena.allocate(data.length * 4L);
        for (int i = 0; i < data.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
        }
        TensorInfo info = new TensorInfo("test", DType.F32, shape, 0, segment.byteSize());
        return new TensorView(segment, info);
    }
}
