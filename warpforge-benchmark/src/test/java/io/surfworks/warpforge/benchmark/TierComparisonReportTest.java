package io.surfworks.warpforge.benchmark;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for TierComparisonReport.
 */
class TierComparisonReportTest {

    @Test
    void testAllPassedWhenWithinTolerance() {
        // Create results with expected overhead
        BenchmarkResult production = createResult(KernelTier.PRODUCTION, 1000);
        BenchmarkResult observable = createResult(KernelTier.OPTIMIZED_OBSERVABLE, 1070); // 7% overhead

        BenchmarkResult.TierComparison comparison = observable.compareTo(production, 3.0);

        TierComparisonReport report = new TierComparisonReport(
            List.of(production, observable),
            List.of(comparison),
            3.0
        );

        assertTrue(report.allPassed());
        assertEquals(1, report.passedCount());
        assertEquals(0, report.failedCount());
    }

    @Test
    void testFailedWhenOverheadExceedsTolerance() {
        BenchmarkResult production = createResult(KernelTier.PRODUCTION, 1000);
        BenchmarkResult observable = createResult(KernelTier.OPTIMIZED_OBSERVABLE, 1150); // 15% overhead

        BenchmarkResult.TierComparison comparison = observable.compareTo(production, 3.0);

        TierComparisonReport report = new TierComparisonReport(
            List.of(production, observable),
            List.of(comparison),
            3.0
        );

        assertFalse(report.allPassed());
        assertEquals(0, report.passedCount());
        assertEquals(1, report.failedCount());
    }

    @Test
    void testOptimizedObservableStats() {
        BenchmarkResult prod1 = createResult(KernelTier.PRODUCTION, 1000, "bench1");
        BenchmarkResult obs1 = createResult(KernelTier.OPTIMIZED_OBSERVABLE, 1070, "bench1"); // 7%
        BenchmarkResult prod2 = createResult(KernelTier.PRODUCTION, 2000, "bench2");
        BenchmarkResult obs2 = createResult(KernelTier.OPTIMIZED_OBSERVABLE, 2160, "bench2"); // 8%

        List<BenchmarkResult.TierComparison> comparisons = List.of(
            obs1.compareTo(prod1, 3.0),
            obs2.compareTo(prod2, 3.0)
        );

        TierComparisonReport report = new TierComparisonReport(
            List.of(prod1, obs1, prod2, obs2),
            comparisons,
            3.0
        );

        var stats = report.optimizedObservableOverheadStats();
        assertEquals(2, stats.getCount());
        assertEquals(7.5, stats.getAverage(), 0.1); // (7 + 8) / 2
        assertEquals(7.0, stats.getMin(), 0.1);
        assertEquals(8.0, stats.getMax(), 0.1);
    }

    @Test
    void testPrintReport() {
        BenchmarkResult production = createResult(KernelTier.PRODUCTION, 1000);
        BenchmarkResult observable = createResult(KernelTier.OPTIMIZED_OBSERVABLE, 1070);

        BenchmarkResult.TierComparison comparison = observable.compareTo(production, 3.0);

        TierComparisonReport report = new TierComparisonReport(
            List.of(production, observable),
            List.of(comparison),
            3.0
        );

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        report.print(ps);

        String output = baos.toString();

        // Verify key sections are present
        assertTrue(output.contains("GPU KERNEL TIER COMPARISON REPORT"));
        assertTrue(output.contains("BENCHMARK RESULTS"));
        assertTrue(output.contains("TIER COMPARISONS"));
        assertTrue(output.contains("OPTIMIZED_OBSERVABLE TIER OVERHEAD ANALYSIS"));
        assertTrue(output.contains("VALIDATION PASSED"));
    }

    @Test
    void testJsonExport() {
        BenchmarkResult production = createResult(KernelTier.PRODUCTION, 1000);
        BenchmarkResult observable = createResult(KernelTier.OPTIMIZED_OBSERVABLE, 1070);

        BenchmarkResult.TierComparison comparison = observable.compareTo(production, 3.0);

        TierComparisonReport report = new TierComparisonReport(
            List.of(production, observable),
            List.of(comparison),
            3.0
        );

        String json = report.toJson();

        // Verify JSON structure
        assertTrue(json.contains("\"summary\""));
        assertTrue(json.contains("\"totalBenchmarks\": 2"));
        assertTrue(json.contains("\"allPassed\": true"));
        assertTrue(json.contains("\"optimizedObservable\""));
        assertTrue(json.contains("\"comparisons\""));
        assertTrue(json.contains("\"observedOverhead\""));
    }

    private BenchmarkResult createResult(KernelTier tier, long meanMicros) {
        return createResult(tier, meanMicros, "TestBenchmark");
    }

    private BenchmarkResult createResult(KernelTier tier, long meanMicros, String name) {
        long[] timings = {meanMicros, meanMicros, meanMicros, meanMicros, meanMicros};
        return new BenchmarkResult(
            name,
            "Add",
            "1K elements",
            tier,
            5,
            5,
            timings,
            0.0
        );
    }
}
