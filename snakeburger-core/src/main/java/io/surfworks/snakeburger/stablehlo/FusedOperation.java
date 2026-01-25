package io.surfworks.snakeburger.stablehlo;

import java.util.List;
import java.util.Map;

import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Represents a fused operation that replaces a sequence of primitive operations.
 *
 * <p>FusedOperation is the output of the fusion pass. It captures:
 * <ul>
 *   <li>The fusion type (softmax, layer_norm, rms_norm, etc.)</li>
 *   <li>Input operands (from the original subgraph)</li>
 *   <li>Output results (same as the final operation in the subgraph)</li>
 *   <li>Configuration parameters (epsilon, axis, etc.)</li>
 *   <li>The original operations that were fused (for debugging/analysis)</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * // Original StableHLO:
 * %0 = stablehlo.reduce(%input, max, dim=-1)  // max
 * %1 = stablehlo.subtract(%input, %0)          // x - max
 * %2 = stablehlo.exponential(%1)               // exp(x - max)
 * %3 = stablehlo.reduce(%2, add, dim=-1)       // sum(exp)
 * %4 = stablehlo.divide(%2, %3)                // exp / sum
 *
 * // After fusion:
 * %4 = fused.softmax(%input, axis=-1)
 * }</pre>
 */
public record FusedOperation(
        String fusionType,
        List<Value> operands,
        List<Value> results,
        Map<String, Object> config,
        List<Operation> originalOps
) implements Operation {

    /**
     * Supported fusion types.
     */
    public static final String SOFTMAX = "softmax";
    public static final String LAYER_NORM = "layer_norm";
    public static final String RMS_NORM = "rms_norm";
    public static final String BIAS_ACTIVATION = "bias_activation";
    public static final String ATTENTION = "attention";

    @Override
    public String opName() {
        return "fused." + fusionType;
    }

    @Override
    public TensorType tensorResultType() {
        if (results.isEmpty()) {
            throw new IllegalStateException("FusedOperation has no results");
        }
        return (TensorType) results.get(0).type();
    }

    /**
     * Get a configuration value by key.
     *
     * @param key the configuration key
     * @param <T> the expected type
     * @return the configuration value, or null if not present
     */
    @SuppressWarnings("unchecked")
    public <T> T configValue(String key) {
        return (T) config.get(key);
    }

    /**
     * Get the axis configuration (common for softmax, layer_norm, etc.).
     *
     * @return the axis value, or -1 if not specified
     */
    public int axis() {
        Object axisObj = config.get("axis");
        if (axisObj instanceof Number num) {
            return num.intValue();
        }
        return -1;
    }

    /**
     * Get the epsilon configuration (common for layer_norm, rms_norm).
     *
     * @return the epsilon value, or 1e-5 if not specified
     */
    public double epsilon() {
        Object epsObj = config.get("epsilon");
        if (epsObj instanceof Number num) {
            return num.doubleValue();
        }
        return 1e-5;
    }

    /**
     * Returns the number of original operations that were fused.
     */
    public int fusedOpCount() {
        return originalOps.size();
    }

    @Override
    public String toString() {
        return String.format("FusedOperation[type=%s, operands=%d, results=%d, fusedOps=%d]",
                fusionType, operands.size(), results.size(), originalOps.size());
    }
}
