package io.surfworks.warpforge.benchmark;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for BenchmarkResult statistical calculations.
 */
class BenchmarkResultTest {

    @Test
    void testMeanCalculation() {
        long[] timings = {100, 200, 300, 400, 500};
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            timings,
            0.0
        );

        assertEquals(300.0, result.meanMicros(), 0.001);
    }

    @Test
    void testStdDevCalculation() {
        long[] timings = {100, 200, 300, 400, 500};
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            timings,
            0.0
        );

        // Standard deviation of 100, 200, 300, 400, 500 is ~158.11
        assertEquals(158.11, result.stdDevMicros(), 1.0);
    }

    @Test
    void testMinMax() {
        long[] timings = {150, 100, 300, 200, 500};
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            timings,
            0.0
        );

        assertEquals(100, result.minMicros());
        assertEquals(500, result.maxMicros());
    }

    @Test
    void testPercentiles() {
        long[] timings = new long[100];
        for (int i = 0; i < 100; i++) {
            timings[i] = i + 1; // 1 to 100
        }
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            100,
            timings,
            0.0
        );

        assertEquals(50, result.p50Micros());
        assertEquals(99, result.p99Micros());
    }

    @Test
    void testTeraflopsCalculation() {
        long[] timings = {1000}; // 1000 microseconds = 1 millisecond
        double flops = 2e12; // 2 trillion FLOPS
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark",
            "GEMM",
            "4Kx4K",
            KernelTier.PRODUCTION,
            5,
            1,
            timings,
            flops
        );

        // TFLOPS = FLOPS / (time_in_micros * 1e6)
        //        = 2e12 / (1000 * 1e6)
        //        = 2e12 / 1e9
        //        = 2000 TFLOPS
        // (Unrealistically high for validation, but mathematically correct)
        assertEquals(2000.0, result.teraflops(), 0.001);
    }

    @Test
    void testCoefficientOfVariation() {
        // All same values = 0% CV
        long[] timings = {100, 100, 100, 100, 100};
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            timings,
            0.0
        );

        assertEquals(0.0, result.coefficientOfVariationPercent(), 0.001);
    }

    @Test
    void testTierComparison() {
        long[] productionTimings = {1000, 1000, 1000, 1000, 1000};
        long[] observableTimings = {1070, 1070, 1070, 1070, 1070}; // ~7% overhead

        BenchmarkResult production = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            productionTimings,
            0.0
        );

        BenchmarkResult observable = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.OPTIMIZED_OBSERVABLE,
            5,
            5,
            observableTimings,
            0.0
        );

        BenchmarkResult.TierComparison comparison = observable.compareTo(production, 3.0);

        assertEquals(7.0, comparison.observedOverheadPercent(), 0.1);
        assertEquals(7.0, comparison.expectedOverheadPercent(), 0.1);
        assertTrue(comparison.withinTolerance());
        assertEquals(1.07, comparison.performanceRatio(), 0.01);
    }

    @Test
    void testTierComparisonFailure() {
        long[] productionTimings = {1000, 1000, 1000, 1000, 1000};
        long[] observableTimings = {1150, 1150, 1150, 1150, 1150}; // 15% overhead (too high)

        BenchmarkResult production = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            productionTimings,
            0.0
        );

        BenchmarkResult observable = new BenchmarkResult(
            "TestBenchmark",
            "Add",
            "1K elements",
            KernelTier.OPTIMIZED_OBSERVABLE,
            5,
            5,
            observableTimings,
            0.0
        );

        BenchmarkResult.TierComparison comparison = observable.compareTo(production, 3.0);

        assertEquals(15.0, comparison.observedOverheadPercent(), 0.1);
        assertEquals(7.0, comparison.expectedOverheadPercent(), 0.1);
        assertFalse(comparison.withinTolerance()); // 15% - 7% = 8% > 3% tolerance
    }

    @Test
    void testSummaryString() {
        long[] timings = {100, 110, 105, 95, 90};
        BenchmarkResult result = new BenchmarkResult(
            "TestBenchmark.benchmarkAdd",
            "Add",
            "1K elements",
            KernelTier.PRODUCTION,
            5,
            5,
            timings,
            0.0
        );

        String summary = result.toSummaryString();
        assertTrue(summary.contains("Add"));
        assertTrue(summary.contains("[PRODUCTION]"));
        assertTrue(summary.contains("1K elements"));
        assertTrue(summary.contains("μs"));
    }
}
