package io.surfworks.warpforge.data.golden;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utilities for comparing tensors with configurable tolerance.
 *
 * <p>Supports multiple comparison modes:
 * <ul>
 *   <li>Absolute tolerance: |expected - actual| &lt;= tolerance</li>
 *   <li>Relative tolerance: |expected - actual| &lt;= tolerance * |expected|</li>
 *   <li>Combined: uses whichever is more lenient</li>
 * </ul>
 */
public final class GoldenComparison {

    private GoldenComparison() {}

    /**
     * Default absolute tolerance for floating-point comparison.
     */
    public static final double DEFAULT_TOLERANCE = 1e-5;

    /**
     * Default relative tolerance for floating-point comparison.
     */
    public static final double DEFAULT_RELATIVE_TOLERANCE = 1e-4;

    /**
     * Compare two tensors using absolute tolerance.
     *
     * @param expected Expected tensor (golden)
     * @param actual Actual tensor to compare
     * @param tolerance Absolute tolerance
     * @return Comparison result
     */
    public static ComparisonResult compare(TensorView expected, TensorView actual, double tolerance) {
        // Check shape match
        if (!Arrays.equals(expected.shape(), actual.shape())) {
            return ComparisonResult.shapeMismatch(expected.shape(), actual.shape());
        }

        // Check dtype compatibility (must be convertible to float)
        if (!canCompareAsFloat(expected.dtype()) || !canCompareAsFloat(actual.dtype())) {
            return ComparisonResult.dtypeMismatch(expected.dtype().name(), actual.dtype().name());
        }

        long totalElements = expected.info().elementCount();
        long mismatchCount = 0;
        double maxDiff = 0;
        double sumAbsDiff = 0;
        List<ComparisonResult.Mismatch> mismatches = new ArrayList<>();

        // Compare element by element
        long[] shape = expected.shape();
        long[] indices = new long[shape.length];

        for (long i = 0; i < totalElements; i++) {
            // Convert flat index to multi-dimensional indices
            indexFromFlat(i, shape, indices);

            float exp = expected.getFloat(indices);
            float act = actual.getFloat(indices);

            // Handle NaN: both NaN = match, one NaN = mismatch
            if (bothNaN(exp, act)) {
                continue; // NaN matches NaN
            }
            if (Float.isNaN(exp) || Float.isNaN(act)) {
                // One is NaN, the other isn't - always a mismatch
                mismatchCount++;
                if (mismatches.size() < ComparisonResult.MAX_MISMATCH_DETAILS) {
                    mismatches.add(new ComparisonResult.Mismatch(indices, exp, act));
                }
                continue;
            }

            double diff = Math.abs(exp - act);
            maxDiff = Math.max(maxDiff, diff);
            sumAbsDiff += diff;

            if (diff > tolerance) {
                mismatchCount++;
                if (mismatches.size() < ComparisonResult.MAX_MISMATCH_DETAILS) {
                    mismatches.add(new ComparisonResult.Mismatch(indices, exp, act));
                }
            }
        }

        double mae = totalElements > 0 ? sumAbsDiff / totalElements : 0;
        boolean matches = mismatchCount == 0;

        return new ComparisonResult(
                matches, totalElements, mismatchCount, maxDiff, mae, tolerance,
                mismatches, false, false
        );
    }

    /**
     * Compare two tensors using combined absolute and relative tolerance.
     *
     * <p>An element matches if: |expected - actual| &lt;= atol + rtol * |expected|
     *
     * @param expected Expected tensor (golden)
     * @param actual Actual tensor to compare
     * @param atol Absolute tolerance
     * @param rtol Relative tolerance
     * @return Comparison result
     */
    public static ComparisonResult compareWithRelative(TensorView expected, TensorView actual,
                                                        double atol, double rtol) {
        // Check shape match
        if (!Arrays.equals(expected.shape(), actual.shape())) {
            return ComparisonResult.shapeMismatch(expected.shape(), actual.shape());
        }

        // Check dtype compatibility
        if (!canCompareAsFloat(expected.dtype()) || !canCompareAsFloat(actual.dtype())) {
            return ComparisonResult.dtypeMismatch(expected.dtype().name(), actual.dtype().name());
        }

        long totalElements = expected.info().elementCount();
        long mismatchCount = 0;
        double maxDiff = 0;
        double sumAbsDiff = 0;
        List<ComparisonResult.Mismatch> mismatches = new ArrayList<>();

        long[] shape = expected.shape();
        long[] indices = new long[shape.length];

        for (long i = 0; i < totalElements; i++) {
            indexFromFlat(i, shape, indices);

            float exp = expected.getFloat(indices);
            float act = actual.getFloat(indices);

            // Handle NaN: both NaN = match, one NaN = mismatch
            if (bothNaN(exp, act)) {
                continue; // NaN matches NaN
            }
            if (Float.isNaN(exp) || Float.isNaN(act)) {
                mismatchCount++;
                if (mismatches.size() < ComparisonResult.MAX_MISMATCH_DETAILS) {
                    mismatches.add(new ComparisonResult.Mismatch(indices, exp, act));
                }
                continue;
            }

            double diff = Math.abs(exp - act);
            double allowedDiff = atol + rtol * Math.abs(exp);

            maxDiff = Math.max(maxDiff, diff);
            sumAbsDiff += diff;

            if (diff > allowedDiff) {
                mismatchCount++;
                if (mismatches.size() < ComparisonResult.MAX_MISMATCH_DETAILS) {
                    mismatches.add(new ComparisonResult.Mismatch(indices, exp, act));
                }
            }
        }

        double mae = totalElements > 0 ? sumAbsDiff / totalElements : 0;
        boolean matches = mismatchCount == 0;

        // Use atol as the reported tolerance (combined tolerance varies per element)
        return new ComparisonResult(
                matches, totalElements, mismatchCount, maxDiff, mae, atol,
                mismatches, false, false
        );
    }

    /**
     * Compare using default tolerances (1e-5 absolute, 1e-4 relative).
     */
    public static ComparisonResult compare(TensorView expected, TensorView actual) {
        return compareWithRelative(expected, actual, DEFAULT_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE);
    }

    /**
     * Check if all elements match exactly (useful for integer tensors).
     */
    public static ComparisonResult compareExact(TensorView expected, TensorView actual) {
        return compare(expected, actual, 0.0);
    }

    /**
     * Calculate statistics for a single tensor.
     */
    public static TensorStats stats(TensorView tensor) {
        long count = tensor.info().elementCount();
        if (count == 0) {
            return new TensorStats(0, 0, 0, 0, 0);
        }

        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        double sum = 0;
        double sumSq = 0;

        long[] shape = tensor.shape();
        long[] indices = new long[shape.length];

        for (long i = 0; i < count; i++) {
            indexFromFlat(i, shape, indices);
            float val = tensor.getFloat(indices);

            if (!Float.isNaN(val)) {
                min = Math.min(min, val);
                max = Math.max(max, val);
                sum += val;
                sumSq += val * val;
            }
        }

        double mean = sum / count;
        double variance = (sumSq / count) - (mean * mean);
        double std = Math.sqrt(Math.max(0, variance));

        return new TensorStats(count, min, max, mean, std);
    }

    /**
     * Statistics for a tensor.
     */
    public record TensorStats(
            long count,
            double min,
            double max,
            double mean,
            double std
    ) {
        public String summary() {
            return String.format("count=%d, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                    count, min, max, mean, std);
        }
    }

    /**
     * Check if a dtype can be compared as float.
     */
    private static boolean canCompareAsFloat(DType dtype) {
        return dtype == DType.F32 || dtype == DType.F64 ||
                dtype == DType.F16 || dtype == DType.BF16;
    }

    /**
     * Convert flat index to multi-dimensional indices.
     */
    private static void indexFromFlat(long flat, long[] shape, long[] indices) {
        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = flat % shape[i];
            flat /= shape[i];
        }
    }

    /**
     * Check if both values are NaN (NaN == NaN should be considered a match).
     */
    private static boolean bothNaN(float a, float b) {
        return Float.isNaN(a) && Float.isNaN(b);
    }
}
