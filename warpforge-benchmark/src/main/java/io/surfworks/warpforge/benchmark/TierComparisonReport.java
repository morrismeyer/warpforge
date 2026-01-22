package io.surfworks.warpforge.benchmark;

import java.io.PrintStream;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Report comparing GPU kernel performance across execution tiers.
 *
 * <p>Validates the key claims of the three-tier architecture:
 * <ul>
 *   <li>OPTIMIZED_OBSERVABLE achieves ~93% of PRODUCTION performance</li>
 *   <li>CORRECTNESS tier is significantly slower but provides full observability</li>
 * </ul>
 *
 * <p>The report provides:
 * <ul>
 *   <li>Per-benchmark comparison between tiers</li>
 *   <li>Aggregate statistics across all benchmarks</li>
 *   <li>Pass/fail status based on tolerance thresholds</li>
 *   <li>Recommendations for tuning if overhead exceeds expectations</li>
 * </ul>
 */
public class TierComparisonReport {

    private final List<BenchmarkResult> results;
    private final List<BenchmarkResult.TierComparison> comparisons;
    private final double tolerancePercent;

    public TierComparisonReport(
        List<BenchmarkResult> results,
        List<BenchmarkResult.TierComparison> comparisons,
        double tolerancePercent
    ) {
        this.results = List.copyOf(results);
        this.comparisons = List.copyOf(comparisons);
        this.tolerancePercent = tolerancePercent;
    }

    /**
     * Returns true if all tier comparisons are within tolerance.
     */
    public boolean allPassed() {
        return comparisons.stream().allMatch(BenchmarkResult.TierComparison::withinTolerance);
    }

    /**
     * Returns the number of passed comparisons.
     */
    public long passedCount() {
        return comparisons.stream().filter(BenchmarkResult.TierComparison::withinTolerance).count();
    }

    /**
     * Returns the number of failed comparisons.
     */
    public long failedCount() {
        return comparisons.stream().filter(c -> !c.withinTolerance()).count();
    }

    /**
     * Returns the comparisons for OPTIMIZED_OBSERVABLE tier.
     */
    public List<BenchmarkResult.TierComparison> optimizedObservableComparisons() {
        return comparisons.stream()
            .filter(c -> c.comparisonTier() == KernelTier.OPTIMIZED_OBSERVABLE)
            .toList();
    }

    /**
     * Returns aggregate statistics for OPTIMIZED_OBSERVABLE overhead.
     */
    public DoubleSummaryStatistics optimizedObservableOverheadStats() {
        return optimizedObservableComparisons().stream()
            .mapToDouble(BenchmarkResult.TierComparison::observedOverheadPercent)
            .summaryStatistics();
    }

    /**
     * Prints the report to stdout.
     */
    public void print() {
        print(System.out);
    }

    /**
     * Prints the report to the specified output stream.
     */
    public void print(PrintStream out) {
        out.println();
        out.println("╔════════════════════════════════════════════════════════════════════════════╗");
        out.println("║                    GPU KERNEL TIER COMPARISON REPORT                       ║");
        out.println("╠════════════════════════════════════════════════════════════════════════════╣");

        // Summary
        out.printf("║  Total Benchmarks: %-56d ║%n", results.size());
        out.printf("║  Total Comparisons: %-55d ║%n", comparisons.size());
        out.printf("║  Tolerance: ±%-61.1f%% ║%n", tolerancePercent);
        out.printf("║  Status: %-66s ║%n", allPassed() ? "✓ ALL PASSED" : "✗ FAILURES DETECTED");
        out.println("╚════════════════════════════════════════════════════════════════════════════╝");
        out.println();

        // Benchmark Results by Operation
        printBenchmarkResults(out);

        // Tier Comparisons
        printTierComparisons(out);

        // OPTIMIZED_OBSERVABLE Statistics
        printOptimizedObservableStats(out);

        // Recommendations
        printRecommendations(out);
    }

    private void printBenchmarkResults(PrintStream out) {
        out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
        out.println("│                           BENCHMARK RESULTS                                  │");
        out.println("├──────────────────────────────────────────────────────────────────────────────┤");

        // Group by operation
        Map<String, List<BenchmarkResult>> byOperation = results.stream()
            .collect(Collectors.groupingBy(BenchmarkResult::operation));

        for (Map.Entry<String, List<BenchmarkResult>> entry : byOperation.entrySet()) {
            out.printf("│  Operation: %-64s │%n", entry.getKey());
            out.println("│  ────────────────────────────────────────────────────────────────────────  │");

            for (BenchmarkResult result : entry.getValue()) {
                out.printf("│    %-12s %8.1f ± %6.1f μs  CV=%5.1f%%  %s │%n",
                    "[" + result.tier() + "]",
                    result.meanMicros(),
                    result.stdDevMicros(),
                    result.coefficientOfVariationPercent(),
                    padRight(result.shape(), 20)
                );
            }
            out.println("│                                                                              │");
        }
        out.println("└──────────────────────────────────────────────────────────────────────────────┘");
        out.println();
    }

    private void printTierComparisons(PrintStream out) {
        if (comparisons.isEmpty()) {
            return;
        }

        out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
        out.println("│                           TIER COMPARISONS                                   │");
        out.println("├──────────────────────────────────────────────────────────────────────────────┤");
        out.println("│  Comparison                      Observed    Expected    Ratio    Status     │");
        out.println("│  ────────────────────────────────────────────────────────────────────────    │");

        for (BenchmarkResult.TierComparison comp : comparisons) {
            String compLabel = comp.comparisonTier().name().substring(0, Math.min(15, comp.comparisonTier().name().length()));
            String status = comp.withinTolerance() ? "✓ PASS" : "✗ FAIL";
            out.printf("│  %-15s vs PRODUCTION  %+7.1f%%    %+6.1f%%    %.3fx    %-6s    │%n",
                compLabel,
                comp.observedOverheadPercent(),
                comp.expectedOverheadPercent(),
                comp.performanceRatio(),
                status
            );
        }
        out.println("└──────────────────────────────────────────────────────────────────────────────┘");
        out.println();
    }

    private void printOptimizedObservableStats(PrintStream out) {
        List<BenchmarkResult.TierComparison> optObs = optimizedObservableComparisons();
        if (optObs.isEmpty()) {
            return;
        }

        DoubleSummaryStatistics stats = optimizedObservableOverheadStats();

        out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
        out.println("│              OPTIMIZED_OBSERVABLE TIER OVERHEAD ANALYSIS                     │");
        out.println("├──────────────────────────────────────────────────────────────────────────────┤");
        out.printf("│  Expected overhead: ~7%% (achieving ~93%% of PRODUCTION performance)          │%n");
        out.println("│  ────────────────────────────────────────────────────────────────────────    │");
        out.printf("│  Observed Mean:    %+6.2f%%                                                   │%n", stats.getAverage());
        out.printf("│  Observed Min:     %+6.2f%%                                                   │%n", stats.getMin());
        out.printf("│  Observed Max:     %+6.2f%%                                                   │%n", stats.getMax());
        out.printf("│  Sample Count:     %6d                                                     │%n", stats.getCount());
        out.println("│  ────────────────────────────────────────────────────────────────────────    │");

        double delta = Math.abs(stats.getAverage() - 7.0);
        String assessment;
        if (delta < 2.0) {
            assessment = "✓ EXCELLENT: Overhead matches theoretical prediction";
        } else if (delta < 5.0) {
            assessment = "~ ACCEPTABLE: Overhead within reasonable bounds";
        } else {
            assessment = "✗ INVESTIGATE: Overhead significantly deviates from expected";
        }
        out.printf("│  Assessment: %-62s │%n", assessment);
        out.println("└──────────────────────────────────────────────────────────────────────────────┘");
        out.println();
    }

    private void printRecommendations(PrintStream out) {
        List<BenchmarkResult.TierComparison> failures = comparisons.stream()
            .filter(c -> !c.withinTolerance())
            .toList();

        if (failures.isEmpty()) {
            out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
            out.println("│                              VALIDATION PASSED                               │");
            out.println("├──────────────────────────────────────────────────────────────────────────────┤");
            out.println("│  All tier comparisons are within the expected tolerance bounds.              │");
            out.println("│  The OPTIMIZED_OBSERVABLE tier can be trusted as 'near-production' speed.    │");
            out.println("└──────────────────────────────────────────────────────────────────────────────┘");
        } else {
            out.println("┌──────────────────────────────────────────────────────────────────────────────┐");
            out.println("│                              VALIDATION FAILED                               │");
            out.println("├──────────────────────────────────────────────────────────────────────────────┤");
            out.printf("│  %d comparison(s) exceeded tolerance. Recommendations:                        │%n", failures.size());
            out.println("│                                                                              │");
            out.println("│  1. Check for measurement noise - increase iteration count                   │");
            out.println("│  2. Verify GPU is not throttling - check temperature and power state         │");
            out.println("│  3. Review salt instrumentation overhead - may need optimization             │");
            out.println("│  4. Consider if operation size is too small for meaningful comparison        │");
            out.println("│                                                                              │");
            out.println("│  Failed Comparisons:                                                         │");
            for (BenchmarkResult.TierComparison failure : failures) {
                out.printf("│    - %s: observed %.1f%% vs expected %.1f%%                            │%n",
                    padRight(failure.operation(), 20),
                    failure.observedOverheadPercent(),
                    failure.expectedOverheadPercent()
                );
            }
            out.println("└──────────────────────────────────────────────────────────────────────────────┘");
        }
        out.println();
    }

    private static String padRight(String s, int width) {
        if (s == null) s = "";
        if (s.length() >= width) return s.substring(0, width);
        return s + " ".repeat(width - s.length());
    }

    /**
     * Exports the report to JSON format for programmatic consumption.
     */
    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"summary\": {\n");
        sb.append("    \"totalBenchmarks\": ").append(results.size()).append(",\n");
        sb.append("    \"totalComparisons\": ").append(comparisons.size()).append(",\n");
        sb.append("    \"passed\": ").append(passedCount()).append(",\n");
        sb.append("    \"failed\": ").append(failedCount()).append(",\n");
        sb.append("    \"tolerance\": ").append(tolerancePercent).append(",\n");
        sb.append("    \"allPassed\": ").append(allPassed()).append("\n");
        sb.append("  },\n");

        // OPTIMIZED_OBSERVABLE stats
        DoubleSummaryStatistics stats = optimizedObservableOverheadStats();
        sb.append("  \"optimizedObservable\": {\n");
        sb.append("    \"expectedOverhead\": 7.0,\n");
        sb.append("    \"meanOverhead\": ").append(String.format("%.2f", stats.getAverage())).append(",\n");
        sb.append("    \"minOverhead\": ").append(String.format("%.2f", stats.getMin())).append(",\n");
        sb.append("    \"maxOverhead\": ").append(String.format("%.2f", stats.getMax())).append(",\n");
        sb.append("    \"sampleCount\": ").append(stats.getCount()).append("\n");
        sb.append("  },\n");

        // Comparisons
        sb.append("  \"comparisons\": [\n");
        for (int i = 0; i < comparisons.size(); i++) {
            BenchmarkResult.TierComparison comp = comparisons.get(i);
            sb.append("    {\n");
            sb.append("      \"benchmark\": \"").append(comp.benchmarkName()).append("\",\n");
            sb.append("      \"operation\": \"").append(comp.operation()).append("\",\n");
            sb.append("      \"comparisonTier\": \"").append(comp.comparisonTier()).append("\",\n");
            sb.append("      \"observedOverhead\": ").append(String.format("%.2f", comp.observedOverheadPercent())).append(",\n");
            sb.append("      \"expectedOverhead\": ").append(String.format("%.2f", comp.expectedOverheadPercent())).append(",\n");
            sb.append("      \"performanceRatio\": ").append(String.format("%.4f", comp.performanceRatio())).append(",\n");
            sb.append("      \"withinTolerance\": ").append(comp.withinTolerance()).append("\n");
            sb.append("    }").append(i < comparisons.size() - 1 ? "," : "").append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");
        return sb.toString();
    }
}
