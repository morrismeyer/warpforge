package io.surfworks.warpforge.benchmark;

import java.util.Arrays;
import java.util.List;

/**
 * Results from a benchmark run including statistical analysis.
 *
 * <p>Captures timing data from multiple iterations and provides statistical
 * metrics following JMH conventions (mean, standard deviation, percentiles).
 */
public record BenchmarkResult(
    String benchmarkName,
    String operation,
    String shape,
    KernelTier tier,
    int warmupIterations,
    int measurementIterations,
    long[] timingsMicros,
    double computeFlops
) {

    /**
     * Calculates the mean execution time in microseconds.
     */
    public double meanMicros() {
        if (timingsMicros.length == 0) return 0.0;
        return Arrays.stream(timingsMicros).average().orElse(0.0);
    }

    /**
     * Calculates the standard deviation in microseconds.
     */
    public double stdDevMicros() {
        if (timingsMicros.length < 2) return 0.0;
        double mean = meanMicros();
        double sumSquaredDiff = Arrays.stream(timingsMicros)
            .mapToDouble(t -> t - mean)
            .map(d -> d * d)
            .sum();
        return Math.sqrt(sumSquaredDiff / (timingsMicros.length - 1));
    }

    /**
     * Returns the minimum execution time in microseconds.
     */
    public long minMicros() {
        return Arrays.stream(timingsMicros).min().orElse(0L);
    }

    /**
     * Returns the maximum execution time in microseconds.
     */
    public long maxMicros() {
        return Arrays.stream(timingsMicros).max().orElse(0L);
    }

    /**
     * Calculates the specified percentile execution time in microseconds.
     */
    public long percentileMicros(double percentile) {
        if (timingsMicros.length == 0) return 0L;
        long[] sorted = timingsMicros.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
    }

    /**
     * Returns the median (50th percentile) execution time.
     */
    public long p50Micros() {
        return percentileMicros(50);
    }

    /**
     * Returns the 99th percentile execution time.
     */
    public long p99Micros() {
        return percentileMicros(99);
    }

    /**
     * Calculates throughput in TFLOPS based on compute operations and mean time.
     */
    public double teraflops() {
        if (computeFlops <= 0 || meanMicros() <= 0) return 0.0;
        // TFLOPS = FLOPS / (time_in_seconds * 1e12)
        //        = FLOPS / (time_in_micros * 1e-6 * 1e12)
        //        = FLOPS / (time_in_micros * 1e6)
        return computeFlops / (meanMicros() * 1e6);
    }

    /**
     * Calculates the coefficient of variation (CV) as a percentage.
     * Lower is better - indicates more stable measurements.
     */
    public double coefficientOfVariationPercent() {
        double mean = meanMicros();
        if (mean <= 0) return 0.0;
        return (stdDevMicros() / mean) * 100.0;
    }

    /**
     * Compares this result against a baseline and returns the overhead analysis.
     */
    public TierComparison compareTo(BenchmarkResult baseline, double tolerancePercent) {
        if (!baseline.operation.equals(this.operation)) {
            throw new IllegalArgumentException(
                "Cannot compare different operations: " + baseline.operation + " vs " + this.operation);
        }

        double baselineMean = baseline.meanMicros();
        double thisMean = this.meanMicros();

        double observedOverhead = baselineMean > 0
            ? ((thisMean - baselineMean) / baselineMean) * 100.0
            : 0.0;

        double expectedOverhead = this.tier.expectedOverheadPercent();
        boolean withinTolerance = Math.abs(observedOverhead - expectedOverhead) <= tolerancePercent;

        return new TierComparison(
            benchmarkName,
            operation,
            shape,
            baseline.tier,
            this.tier,
            (long) baselineMean,
            (long) thisMean,
            observedOverhead,
            expectedOverhead,
            withinTolerance,
            tolerancePercent
        );
    }

    /**
     * Formats the result as a human-readable summary string.
     */
    public String toSummaryString() {
        return String.format(
            "%s [%s] %s: %.2f ± %.2f μs (min=%.0f, p50=%.0f, p99=%.0f, max=%.0f) CV=%.1f%% %s",
            operation,
            tier,
            shape,
            meanMicros(),
            stdDevMicros(),
            (double) minMicros(),
            (double) p50Micros(),
            (double) p99Micros(),
            (double) maxMicros(),
            coefficientOfVariationPercent(),
            teraflops() > 0 ? String.format("%.2f TFLOPS", teraflops()) : ""
        );
    }

    /**
     * Result of comparing two benchmark results across tiers.
     */
    public record TierComparison(
        String benchmarkName,
        String operation,
        String shape,
        KernelTier baselineTier,
        KernelTier comparisonTier,
        long baselineMeanMicros,
        long comparisonMeanMicros,
        double observedOverheadPercent,
        double expectedOverheadPercent,
        boolean withinTolerance,
        double tolerancePercent
    ) {
        /**
         * Returns the performance ratio (comparison / baseline).
         * A value of 1.07 means the comparison tier is 7% slower.
         */
        public double performanceRatio() {
            return baselineMeanMicros > 0
                ? (double) comparisonMeanMicros / baselineMeanMicros
                : 0.0;
        }

        /**
         * Formats the comparison as a human-readable string.
         */
        public String toSummaryString() {
            String status = withinTolerance ? "✓ PASS" : "✗ FAIL";
            return String.format(
                "%s %s vs %s: %.1f%% overhead (expected %.1f%% ± %.1f%%) ratio=%.3f %s",
                operation,
                comparisonTier,
                baselineTier,
                observedOverheadPercent,
                expectedOverheadPercent,
                tolerancePercent,
                performanceRatio(),
                status
            );
        }
    }
}
