package io.surfworks.warpforge.data.golden;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Result of comparing an actual tensor against a golden (expected) output.
 *
 * <p>Provides detailed statistics about the comparison:
 * <ul>
 *   <li>Whether the tensors match within tolerance</li>
 *   <li>Maximum difference found</li>
 *   <li>Mean absolute error</li>
 *   <li>Locations of mismatches</li>
 * </ul>
 *
 * @param matches Whether all elements are within tolerance
 * @param totalElements Total number of elements compared
 * @param mismatchCount Number of elements that exceeded tolerance
 * @param maxDifference Maximum absolute difference found
 * @param meanAbsoluteError Mean of absolute differences
 * @param tolerance Tolerance used for comparison
 * @param firstMismatches First few mismatch locations (for debugging)
 * @param shapeMismatch Whether shapes didn't match
 * @param dtypeMismatch Whether dtypes didn't match
 */
public record ComparisonResult(
        boolean matches,
        long totalElements,
        long mismatchCount,
        double maxDifference,
        double meanAbsoluteError,
        double tolerance,
        List<Mismatch> firstMismatches,
        boolean shapeMismatch,
        boolean dtypeMismatch
) {

    public ComparisonResult {
        Objects.requireNonNull(firstMismatches);
        firstMismatches = List.copyOf(firstMismatches);
    }

    /**
     * Maximum number of mismatch details to retain.
     */
    public static final int MAX_MISMATCH_DETAILS = 10;

    /**
     * Create a successful match result.
     */
    public static ComparisonResult success(long totalElements, double maxDiff, double mae, double tolerance) {
        return new ComparisonResult(
                true, totalElements, 0, maxDiff, mae, tolerance,
                List.of(), false, false
        );
    }

    /**
     * Create a result indicating shape mismatch.
     */
    public static ComparisonResult shapeMismatch(long[] expected, long[] actual) {
        return new ComparisonResult(
                false, 0, 0, Double.NaN, Double.NaN, 0,
                List.of(new Mismatch(new long[0], Double.NaN, Double.NaN,
                        "Shape mismatch: expected " + Arrays.toString(expected) +
                                ", got " + Arrays.toString(actual))),
                true, false
        );
    }

    /**
     * Create a result indicating dtype mismatch.
     */
    public static ComparisonResult dtypeMismatch(String expected, String actual) {
        return new ComparisonResult(
                false, 0, 0, Double.NaN, Double.NaN, 0,
                List.of(new Mismatch(new long[0], Double.NaN, Double.NaN,
                        "DType mismatch: expected " + expected + ", got " + actual)),
                false, true
        );
    }

    /**
     * Percentage of elements that matched.
     */
    public double matchPercentage() {
        if (totalElements == 0) return 100.0;
        return 100.0 * (totalElements - mismatchCount) / totalElements;
    }

    /**
     * Human-readable summary of the comparison.
     */
    public String summary() {
        if (shapeMismatch) {
            return "SHAPE MISMATCH: " + firstMismatches.get(0).message();
        }
        if (dtypeMismatch) {
            return "DTYPE MISMATCH: " + firstMismatches.get(0).message();
        }
        if (matches) {
            return String.format("MATCH: %d elements, max diff=%.2e, MAE=%.2e (tolerance=%.2e)",
                    totalElements, maxDifference, meanAbsoluteError, tolerance);
        } else {
            return String.format("MISMATCH: %d/%d elements (%.1f%%) exceeded tolerance %.2e, max diff=%.2e",
                    mismatchCount, totalElements, 100.0 * mismatchCount / totalElements,
                    tolerance, maxDifference);
        }
    }

    /**
     * Detailed report including first few mismatches.
     */
    public String detailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(summary()).append("\n");

        if (!matches && !firstMismatches.isEmpty() && !shapeMismatch && !dtypeMismatch) {
            sb.append("First mismatches:\n");
            for (Mismatch m : firstMismatches) {
                sb.append(String.format("  %s: expected=%.6f, actual=%.6f, diff=%.2e\n",
                        Arrays.toString(m.indices()), m.expected(), m.actual(),
                        Math.abs(m.expected() - m.actual())));
            }
            if (mismatchCount > MAX_MISMATCH_DETAILS) {
                sb.append(String.format("  ... and %d more mismatches\n",
                        mismatchCount - MAX_MISMATCH_DETAILS));
            }
        }

        return sb.toString();
    }

    /**
     * Details about a single element mismatch.
     */
    public record Mismatch(
            long[] indices,
            double expected,
            double actual,
            String message
    ) {
        public Mismatch(long[] indices, double expected, double actual) {
            this(indices.clone(), expected, actual, null);
        }

        public Mismatch {
            indices = indices.clone();
        }

        /**
         * Absolute difference between expected and actual.
         */
        public double difference() {
            return Math.abs(expected - actual);
        }
    }
}
