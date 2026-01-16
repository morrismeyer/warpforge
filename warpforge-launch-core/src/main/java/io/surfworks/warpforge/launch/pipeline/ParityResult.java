package io.surfworks.warpforge.launch.pipeline;

import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Result of parity testing between WarpForge and PyTorch.
 *
 * @param matches          true if all outputs match within tolerance
 * @param pytorchOutputs   output values from PyTorch
 * @param warpforgeOutputs output values from WarpForge
 * @param differences      per-output difference details
 * @param pytorchTime      PyTorch execution time
 * @param warpforgeTime    WarpForge execution time
 * @param tolerance        tolerance used for comparison
 * @param errorMessage     error message if testing failed (null if completed)
 */
public record ParityResult(
        boolean matches,
        List<double[]> pytorchOutputs,
        List<double[]> warpforgeOutputs,
        List<OutputDifference> differences,
        Duration pytorchTime,
        Duration warpforgeTime,
        double tolerance,
        String errorMessage
) {

    /**
     * Details about differences in a single output tensor.
     *
     * @param outputIndex    index of the output tensor
     * @param maxDifference  maximum absolute difference
     * @param meanDifference mean absolute difference
     * @param numMismatches  number of elements exceeding tolerance
     * @param totalElements  total number of elements
     */
    public record OutputDifference(
            int outputIndex,
            double maxDifference,
            double meanDifference,
            int numMismatches,
            int totalElements
    ) {
        /**
         * Returns true if this output matches within tolerance.
         */
        public boolean matches() {
            return numMismatches == 0;
        }

        /**
         * Returns the percentage of mismatched elements.
         */
        public double mismatchPercentage() {
            if (totalElements == 0) return 0.0;
            return (numMismatches * 100.0) / totalElements;
        }
    }

    /**
     * Creates a successful parity result where outputs match.
     */
    public static ParityResult matching(
            List<double[]> pytorchOutputs,
            List<double[]> warpforgeOutputs,
            Duration pytorchTime,
            Duration warpforgeTime,
            double tolerance) {
        return new ParityResult(
                true,
                pytorchOutputs,
                warpforgeOutputs,
                List.of(),
                pytorchTime,
                warpforgeTime,
                tolerance,
                null
        );
    }

    /**
     * Creates a parity result where outputs differ.
     */
    public static ParityResult different(
            List<double[]> pytorchOutputs,
            List<double[]> warpforgeOutputs,
            List<OutputDifference> differences,
            Duration pytorchTime,
            Duration warpforgeTime,
            double tolerance) {
        return new ParityResult(
                false,
                pytorchOutputs,
                warpforgeOutputs,
                differences,
                pytorchTime,
                warpforgeTime,
                tolerance,
                null
        );
    }

    /**
     * Creates a failure result when testing could not complete.
     */
    public static ParityResult error(String errorMessage, double tolerance) {
        return new ParityResult(
                false,
                List.of(),
                List.of(),
                List.of(),
                Duration.ZERO,
                Duration.ZERO,
                tolerance,
                errorMessage
        );
    }

    /**
     * Returns true if testing completed (regardless of match status).
     */
    public boolean completed() {
        return errorMessage == null;
    }

    /**
     * Returns the speedup factor (PyTorch time / WarpForge time).
     *
     * <p>Values greater than 1.0 indicate WarpForge is faster.
     */
    public double speedup() {
        if (warpforgeTime.isZero()) return 0.0;
        return (double) pytorchTime.toNanos() / warpforgeTime.toNanos();
    }
}
