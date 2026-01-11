package io.surfworks.warpforge.core.testing;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Assertion utilities for tensor comparisons in tests.
 * Provides detailed failure messages showing where tensors differ.
 */
public final class TensorAssert {

    private TensorAssert() {} // Utility class

    /**
     * Assert that two tensors are equal within default tolerance.
     *
     * @param expected Expected tensor
     * @param actual   Actual tensor
     * @throws AssertionError if tensors differ
     */
    public static void assertEquals(Tensor expected, Tensor actual) {
        // Handle null case before accessing dtype
        if (expected == null && actual == null) {
            return;
        }
        ToleranceConfig tolerance = expected != null
            ? ToleranceConfig.forDtype(expected.dtype())
            : ToleranceConfig.forDtype(actual.dtype());
        assertEquals(expected, actual, tolerance);
    }

    /**
     * Assert that two tensors are equal within specified tolerance.
     *
     * @param expected  Expected tensor
     * @param actual    Actual tensor
     * @param tolerance Tolerance configuration
     * @throws AssertionError if tensors differ
     */
    public static void assertEquals(Tensor expected, Tensor actual, ToleranceConfig tolerance) {
        assertEquals(null, expected, actual, tolerance);
    }

    /**
     * Assert that two tensors are equal within specified tolerance with custom message.
     *
     * @param message   Custom failure message prefix
     * @param expected  Expected tensor
     * @param actual    Actual tensor
     * @param tolerance Tolerance configuration
     * @throws AssertionError if tensors differ
     */
    public static void assertEquals(String message, Tensor expected, Tensor actual, ToleranceConfig tolerance) {
        String prefix = message != null ? message + ": " : "";

        // Check nulls
        if (expected == null && actual == null) {
            return;
        }
        if (expected == null) {
            throw new AssertionError(prefix + "expected null but was " + describeTensor(actual));
        }
        if (actual == null) {
            throw new AssertionError(prefix + "expected " + describeTensor(expected) + " but was null");
        }

        // Check shapes
        if (!Arrays.equals(expected.shape(), actual.shape())) {
            throw new AssertionError(prefix + "shape mismatch: expected " +
                Arrays.toString(expected.shape()) + " but was " + Arrays.toString(actual.shape()));
        }

        // Check dtypes
        if (expected.dtype() != actual.dtype()) {
            throw new AssertionError(prefix + "dtype mismatch: expected " +
                expected.dtype() + " but was " + actual.dtype());
        }

        // Compare elements
        ComparisonResult result = compareElements(expected, actual, tolerance);
        if (!result.passed) {
            throw new AssertionError(prefix + result.message);
        }
    }

    /**
     * Assert that two tensors have the same shape.
     *
     * @param expected Expected tensor
     * @param actual   Actual tensor
     * @throws AssertionError if shapes differ
     */
    public static void assertShapeEquals(Tensor expected, Tensor actual) {
        if (!Arrays.equals(expected.shape(), actual.shape())) {
            throw new AssertionError("shape mismatch: expected " +
                Arrays.toString(expected.shape()) + " but was " + Arrays.toString(actual.shape()));
        }
    }

    /**
     * Assert that a tensor has the expected shape.
     *
     * @param expected Expected shape
     * @param actual   Actual tensor
     * @throws AssertionError if shape differs
     */
    public static void assertShapeEquals(int[] expected, Tensor actual) {
        if (!Arrays.equals(expected, actual.shape())) {
            throw new AssertionError("shape mismatch: expected " +
                Arrays.toString(expected) + " but was " + Arrays.toString(actual.shape()));
        }
    }

    /**
     * Assert that a tensor has the expected dtype.
     *
     * @param expected Expected dtype
     * @param actual   Actual tensor
     * @throws AssertionError if dtype differs
     */
    public static void assertDtypeEquals(ScalarType expected, Tensor actual) {
        if (expected != actual.dtype()) {
            throw new AssertionError("dtype mismatch: expected " + expected + " but was " + actual.dtype());
        }
    }

    /**
     * Assert that all tensor elements are close to a scalar value.
     *
     * @param expected  Expected scalar value
     * @param actual    Actual tensor
     * @param tolerance Tolerance configuration
     * @throws AssertionError if any element differs
     */
    public static void assertAllClose(double expected, Tensor actual, ToleranceConfig tolerance) {
        float[] data = actual.toFloatArray();
        List<Integer> failures = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            if (!tolerance.isClose(expected, data[i])) {
                failures.add(i);
                if (failures.size() >= 5) break; // Limit reported failures
            }
        }
        if (!failures.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("expected all elements close to ").append(expected);
            sb.append(" but found ").append(failures.size()).append(" failures");
            if (failures.size() < data.length) {
                sb.append(" (showing first ").append(failures.size()).append(")");
            }
            sb.append(":\n");
            for (int idx : failures) {
                sb.append("  [").append(idx).append("] = ").append(data[idx]).append("\n");
            }
            throw new AssertionError(sb.toString());
        }
    }

    /**
     * Assert that a tensor contains finite values (no NaN or Inf).
     *
     * @param tensor Tensor to check
     * @throws AssertionError if any non-finite values found
     */
    public static void assertFinite(Tensor tensor) {
        float[] data = tensor.toFloatArray();
        List<Integer> nonFinite = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            if (!Float.isFinite(data[i])) {
                nonFinite.add(i);
                if (nonFinite.size() >= 5) break;
            }
        }
        if (!nonFinite.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("tensor contains non-finite values at indices: ");
            for (int idx : nonFinite) {
                sb.append("[").append(idx).append("]=").append(data[idx]).append(" ");
            }
            throw new AssertionError(sb.toString());
        }
    }

    /**
     * Assert that a tensor contains no NaN values.
     *
     * @param tensor Tensor to check
     * @throws AssertionError if any NaN values found
     */
    public static void assertNoNaN(Tensor tensor) {
        float[] data = tensor.toFloatArray();
        for (int i = 0; i < data.length; i++) {
            if (Float.isNaN(data[i])) {
                throw new AssertionError("tensor contains NaN at index " + i);
            }
        }
    }

    // ==================== Internal Helpers ====================

    private static ComparisonResult compareElements(Tensor expected, Tensor actual, ToleranceConfig tolerance) {
        long elementCount = expected.spec().elementCount();
        float[] expectedData = expected.toFloatArray();
        float[] actualData = actual.toFloatArray();

        int mismatchCount = 0;
        double maxAbsDiff = 0;
        double maxRelDiff = 0;
        int maxDiffIndex = -1;
        List<MismatchInfo> mismatches = new ArrayList<>();

        for (int i = 0; i < elementCount; i++) {
            float e = expectedData[i];
            float a = actualData[i];

            if (!tolerance.isClose(e, a)) {
                mismatchCount++;
                double absDiff = Math.abs(e - a);
                double relDiff = e != 0 ? absDiff / Math.abs(e) : absDiff;

                if (absDiff > maxAbsDiff) {
                    maxAbsDiff = absDiff;
                    maxRelDiff = relDiff;
                    maxDiffIndex = i;
                }

                if (mismatches.size() < 5) {
                    mismatches.add(new MismatchInfo(i, e, a, absDiff, relDiff));
                }
            }
        }

        if (mismatchCount == 0) {
            return new ComparisonResult(true, null);
        }

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("tensors differ at %d/%d elements (%.2f%%)\n",
            mismatchCount, elementCount, 100.0 * mismatchCount / elementCount));
        sb.append(String.format("max diff: abs=%.6e, rel=%.6e at index %d\n",
            maxAbsDiff, maxRelDiff, maxDiffIndex));
        sb.append(String.format("tolerance: %s\n", tolerance));
        sb.append("first mismatches:\n");
        for (MismatchInfo m : mismatches) {
            long[] indices = flatToMultiIndex(m.index, expected.shape());
            sb.append(String.format("  %s: expected %.6e, actual %.6e, diff=%.6e\n",
                Arrays.toString(indices), m.expected, m.actual, m.absDiff));
        }

        return new ComparisonResult(false, sb.toString());
    }

    private static String describeTensor(Tensor t) {
        return String.format("Tensor(shape=%s, dtype=%s)", Arrays.toString(t.shape()), t.dtype());
    }

    private static long[] flatToMultiIndex(int flatIndex, int[] shape) {
        long[] indices = new long[shape.length];
        int remaining = flatIndex;
        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        return indices;
    }

    private record ComparisonResult(boolean passed, String message) {}

    private record MismatchInfo(int index, float expected, float actual, double absDiff, double relDiff) {}
}
