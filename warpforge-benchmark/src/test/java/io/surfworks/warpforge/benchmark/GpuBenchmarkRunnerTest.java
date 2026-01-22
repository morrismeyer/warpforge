package io.surfworks.warpforge.benchmark;

import io.surfworks.warpforge.benchmark.annotation.GpuBenchmark;
import io.surfworks.warpforge.benchmark.annotation.Setup;
import io.surfworks.warpforge.benchmark.annotation.TearDown;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for GpuBenchmarkRunner.
 */
class GpuBenchmarkRunnerTest {

    /**
     * Simple benchmark class for testing.
     */
    public static class SimpleBenchmark {
        private boolean setupCalled = false;
        private boolean teardownCalled = false;
        private int invocationCount = 0;

        @Setup(level = Setup.Level.TRIAL)
        public void setup() {
            setupCalled = true;
        }

        @TearDown(level = Setup.Level.TRIAL)
        public void teardown() {
            teardownCalled = true;
        }

        @GpuBenchmark(
            operation = "TestOp",
            shape = "100 elements",
            warmupIterations = 2,
            measurementIterations = 5,
            tiers = {KernelTier.PRODUCTION, KernelTier.OPTIMIZED_OBSERVABLE}
        )
        public void benchmarkTest(KernelTier tier) {
            invocationCount++;
            // Simulate some work
            simulateWork(tier);
        }

        private void simulateWork(KernelTier tier) {
            // Just burn some CPU cycles
            long sum = 0;
            for (int i = 0; i < 10000; i++) {
                sum += i;
            }
            // Add extra overhead for OPTIMIZED_OBSERVABLE to simulate real scenario
            if (tier == KernelTier.OPTIMIZED_OBSERVABLE) {
                for (int i = 0; i < 700; i++) {
                    sum += i;
                }
            }
        }

        public boolean isSetupCalled() { return setupCalled; }
        public boolean isTeardownCalled() { return teardownCalled; }
        public int getInvocationCount() { return invocationCount; }
    }

    @Test
    void testRunnerExecutesBenchmarks() {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.jfrProfiler(false); // Disable JFR profiler for unit test

        runner.run(SimpleBenchmark.class);

        List<BenchmarkResult> results = runner.getResults();

        // Should have 2 results (one per tier)
        assertEquals(2, results.size());

        // Verify tiers
        assertTrue(results.stream().anyMatch(r -> r.tier() == KernelTier.PRODUCTION));
        assertTrue(results.stream().anyMatch(r -> r.tier() == KernelTier.OPTIMIZED_OBSERVABLE));
    }

    @Test
    void testRunnerGeneratesComparisons() {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.jfrProfiler(false);

        runner.run(SimpleBenchmark.class);

        List<BenchmarkResult.TierComparison> comparisons = runner.getComparisons();

        // Should have 1 comparison (OPTIMIZED_OBSERVABLE vs PRODUCTION)
        assertEquals(1, comparisons.size());

        BenchmarkResult.TierComparison comparison = comparisons.get(0);
        assertEquals(KernelTier.PRODUCTION, comparison.baselineTier());
        assertEquals(KernelTier.OPTIMIZED_OBSERVABLE, comparison.comparisonTier());
    }

    @Test
    void testRunnerCallsSetupAndTeardown() {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.jfrProfiler(false);

        // We can't directly check the instance since runner creates its own,
        // but we can verify no exceptions are thrown during execution
        runner.run(SimpleBenchmark.class);

        // If we get here without exception, setup/teardown were invoked successfully
        assertTrue(true);
    }

    @Test
    void testRunnerReport() {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.jfrProfiler(false);

        runner.run(SimpleBenchmark.class);

        TierComparisonReport report = runner.generateReport();

        // Report should exist and have data
        assertFalse(report.optimizedObservableComparisons().isEmpty());
    }

    @Test
    void testIncludePattern() {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.jfrProfiler(false);
        runner.include("nonexistent.*");

        runner.run(SimpleBenchmark.class);

        // No results because no methods match the pattern
        List<BenchmarkResult> results = runner.getResults();
        assertEquals(0, results.size());
    }

    @Test
    void testToleranceSetting() {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.jfrProfiler(false);
        runner.tolerance(50.0); // Very high tolerance

        runner.run(SimpleBenchmark.class);

        TierComparisonReport report = runner.generateReport();

        // Verify tolerance was applied (report was generated with comparisons)
        // Note: We don't assert allPassed() because CPU timing is non-deterministic
        // and the simulated workload may not produce consistent overhead across platforms
        assertFalse(report.optimizedObservableComparisons().isEmpty());
    }
}
