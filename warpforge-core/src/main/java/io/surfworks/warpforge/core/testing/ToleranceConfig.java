package io.surfworks.warpforge.core.testing;

import io.surfworks.warpforge.core.tensor.ScalarType;

import java.util.Map;

/**
 * Configuration for numerical tolerance in tensor comparisons.
 * Provides per-operation and per-dtype tolerance settings.
 *
 * <p>Tolerance is computed using the standard formula:
 * {@code |expected - actual| <= atol + rtol * |expected|}
 *
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 */
public record ToleranceConfig(double atol, double rtol) {

    /**
     * Default tolerances by operation category.
     */
    private static final Map<String, ToleranceConfig> OP_DEFAULTS = Map.ofEntries(
        // Elementwise exact operations
        Map.entry("add", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("subtract", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("multiply", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("divide", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("maximum", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("minimum", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("negate", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("abs", new ToleranceConfig(1e-6, 1e-5)),
        Map.entry("clamp", new ToleranceConfig(1e-6, 1e-5)),

        // Transcendental functions (higher tolerance due to approximations)
        Map.entry("exp", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("log", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("tanh", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("sine", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("cosine", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("sqrt", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("rsqrt", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("logistic", new ToleranceConfig(1e-5, 1e-4)),

        // Custom call transformer operations (higher tolerance for composite ops)
        Map.entry("gelu", new ToleranceConfig(5e-4, 5e-4)),
        Map.entry("gelu_tanh", new ToleranceConfig(5e-4, 5e-4)),
        Map.entry("silu", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("softmax", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("layer_norm", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("batch_norm", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("rms_norm", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("custom_call", new ToleranceConfig(1e-3, 1e-2)),

        // Composite transformer patterns (accumulated errors from multiple ops)
        Map.entry("ffn_block", new ToleranceConfig(5e-4, 5e-4)),
        Map.entry("pre_norm_residual", new ToleranceConfig(5e-4, 5e-4)),
        Map.entry("attention_scores", new ToleranceConfig(5e-4, 5e-4)),
        Map.entry("scaled_dot_product_attention", new ToleranceConfig(5e-4, 5e-4)),
        Map.entry("multi_head_attention", new ToleranceConfig(1e-3, 1e-3)),
        Map.entry("transformer_encoder_block", new ToleranceConfig(1e-3, 1e-3)),
        Map.entry("transformer_block", new ToleranceConfig(1e-3, 1e-3)),

        // Matrix operations (accumulated errors)
        Map.entry("dot_general", new ToleranceConfig(1e-4, 1e-3)),
        Map.entry("convolution", new ToleranceConfig(1e-3, 1e-2)),

        // Reduction operations
        Map.entry("reduce", new ToleranceConfig(1e-5, 1e-4)),
        Map.entry("reduce_window", new ToleranceConfig(1e-4, 1e-3)),

        // Comparison and selection (exact)
        Map.entry("compare", new ToleranceConfig(0, 0)),
        Map.entry("select", new ToleranceConfig(1e-6, 1e-5)),

        // Shape manipulation (exact)
        Map.entry("reshape", new ToleranceConfig(0, 0)),
        Map.entry("transpose", new ToleranceConfig(0, 0)),
        Map.entry("broadcast_in_dim", new ToleranceConfig(0, 0)),
        Map.entry("slice", new ToleranceConfig(0, 0)),
        Map.entry("concatenate", new ToleranceConfig(0, 0)),
        Map.entry("gather", new ToleranceConfig(0, 0)),
        Map.entry("scatter", new ToleranceConfig(0, 0))
    );

    /**
     * Default tolerances by scalar type.
     */
    private static final Map<ScalarType, ToleranceConfig> DTYPE_DEFAULTS = Map.of(
        ScalarType.F16, new ToleranceConfig(1e-2, 1e-2),
        ScalarType.BF16, new ToleranceConfig(1e-2, 1e-2),
        ScalarType.F32, new ToleranceConfig(1e-5, 1e-4),
        ScalarType.F64, new ToleranceConfig(1e-10, 1e-9),
        ScalarType.I8, new ToleranceConfig(0, 0),
        ScalarType.I16, new ToleranceConfig(0, 0),
        ScalarType.I32, new ToleranceConfig(0, 0),
        ScalarType.I64, new ToleranceConfig(0, 0),
        ScalarType.BOOL, new ToleranceConfig(0, 0)
    );

    /**
     * Strict tolerance (exact match).
     */
    public static final ToleranceConfig STRICT = new ToleranceConfig(0, 0);

    /**
     * Loose tolerance for debugging.
     */
    public static final ToleranceConfig LOOSE = new ToleranceConfig(1e-2, 1e-2);

    /**
     * Get tolerance for a specific operation.
     *
     * @param opName Operation name (e.g., "add", "dot_general")
     * @return Tolerance configuration for the operation
     */
    public static ToleranceConfig forOp(String opName) {
        String normalizedName = opName.toLowerCase().replace("stablehlo.", "");
        return OP_DEFAULTS.getOrDefault(normalizedName, new ToleranceConfig(1e-5, 1e-4));
    }

    /**
     * Get tolerance for a specific data type.
     *
     * @param dtype Scalar type
     * @return Tolerance configuration for the dtype
     */
    public static ToleranceConfig forDtype(ScalarType dtype) {
        return DTYPE_DEFAULTS.getOrDefault(dtype, new ToleranceConfig(1e-5, 1e-4));
    }

    /**
     * Get tolerance considering both operation and dtype.
     * Uses the more permissive of the two tolerances.
     *
     * @param opName Operation name
     * @param dtype  Scalar type
     * @return Combined tolerance configuration
     */
    public static ToleranceConfig forOp(String opName, ScalarType dtype) {
        ToleranceConfig opTol = forOp(opName);
        ToleranceConfig dtypeTol = forDtype(dtype);
        return new ToleranceConfig(
            Math.max(opTol.atol, dtypeTol.atol),
            Math.max(opTol.rtol, dtypeTol.rtol)
        );
    }

    /**
     * Check if two values are close within tolerance.
     *
     * @param expected Expected value
     * @param actual   Actual value
     * @return true if values are within tolerance
     */
    public boolean isClose(double expected, double actual) {
        if (Double.isNaN(expected) && Double.isNaN(actual)) {
            return true;
        }
        // If either is NaN (but not both), they're not close
        if (Double.isNaN(expected) || Double.isNaN(actual)) {
            return false;
        }
        // Both infinite: must be same sign
        if (Double.isInfinite(expected) && Double.isInfinite(actual)) {
            return expected == actual;
        }
        // One infinite, one finite: not close
        if (Double.isInfinite(expected) || Double.isInfinite(actual)) {
            return false;
        }
        double diff = Math.abs(expected - actual);
        return diff <= atol + rtol * Math.abs(expected);
    }

    /**
     * Create a new tolerance with scaled values.
     *
     * @param factor Scale factor
     * @return Scaled tolerance config
     */
    public ToleranceConfig scaled(double factor) {
        return new ToleranceConfig(atol * factor, rtol * factor);
    }

    /**
     * Create a tolerance that is the maximum of this and another.
     *
     * @param other Other tolerance config
     * @return Combined tolerance (more permissive)
     */
    public ToleranceConfig or(ToleranceConfig other) {
        return new ToleranceConfig(
            Math.max(this.atol, other.atol),
            Math.max(this.rtol, other.rtol)
        );
    }

    /**
     * Create a tolerance that is the minimum of this and another.
     *
     * @param other Other tolerance config
     * @return Combined tolerance (more strict)
     */
    public ToleranceConfig and(ToleranceConfig other) {
        return new ToleranceConfig(
            Math.min(this.atol, other.atol),
            Math.min(this.rtol, other.rtol)
        );
    }

    @Override
    public String toString() {
        return String.format("ToleranceConfig(atol=%.2e, rtol=%.2e)", atol, rtol);
    }
}
