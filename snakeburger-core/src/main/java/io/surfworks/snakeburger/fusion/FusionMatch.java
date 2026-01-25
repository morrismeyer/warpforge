package io.surfworks.snakeburger.fusion;

import java.util.List;
import java.util.Map;

import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Represents a successful pattern match for fusion.
 *
 * <p>FusionMatch captures:
 * <ul>
 *   <li>The name of the matched pattern</li>
 *   <li>The operations that were matched (to be replaced)</li>
 *   <li>Named value captures (input, output, weights, etc.)</li>
 *   <li>Extracted configuration attributes (axis, epsilon, etc.)</li>
 * </ul>
 *
 * <p>Example for a softmax match:
 * <pre>{@code
 * FusionMatch match = new FusionMatch(
 *     "softmax",
 *     List.of(reduceMaxOp, subOp, expOp, reduceSumOp, divOp),
 *     Map.of("input", inputValue, "output", divResultValue),
 *     Map.of("axis", -1)
 * );
 * }</pre>
 *
 * @param patternName the name of the fusion pattern that matched
 * @param matchedOps the operations that will be replaced by the fused op
 * @param capturedValues named captures from the pattern (input, output, etc.)
 * @param attributes extracted configuration (axis, epsilon, etc.)
 */
public record FusionMatch(
        String patternName,
        List<Operation> matchedOps,
        Map<String, Value> capturedValues,
        Map<String, Object> attributes
) {

    /**
     * Gets the primary input value.
     *
     * <p>By convention, patterns should capture the main input as "input".
     *
     * @return the input value
     */
    public Value input() {
        return capturedValues.get("input");
    }

    /**
     * Gets the final output value.
     *
     * <p>By convention, patterns should capture the final result as "output".
     *
     * @return the output value
     */
    public Value output() {
        return capturedValues.get("output");
    }

    /**
     * Gets a captured value by name.
     *
     * @param name the capture name
     * @return the captured value, or null if not present
     */
    public Value capture(String name) {
        return capturedValues.get(name);
    }

    /**
     * Gets an attribute value.
     *
     * @param name the attribute name
     * @param <T> the expected type
     * @return the attribute value, or null if not present
     */
    @SuppressWarnings("unchecked")
    public <T> T attribute(String name) {
        return (T) attributes.get(name);
    }

    /**
     * Gets an attribute value with a default.
     *
     * @param name the attribute name
     * @param defaultValue the default value if not present
     * @param <T> the expected type
     * @return the attribute value or default
     */
    @SuppressWarnings("unchecked")
    public <T> T attribute(String name, T defaultValue) {
        Object value = attributes.get(name);
        return value != null ? (T) value : defaultValue;
    }

    /**
     * Returns the number of operations that will be fused.
     */
    public int fusedOpCount() {
        return matchedOps.size();
    }

    /**
     * Checks if this match includes a specific operation.
     *
     * @param op the operation to check
     * @return true if the operation is part of this match
     */
    public boolean includes(Operation op) {
        return matchedOps.contains(op);
    }

    @Override
    public String toString() {
        return String.format("FusionMatch[pattern=%s, ops=%d, captures=%s, attrs=%s]",
                patternName, matchedOps.size(), capturedValues.keySet(), attributes.keySet());
    }
}
