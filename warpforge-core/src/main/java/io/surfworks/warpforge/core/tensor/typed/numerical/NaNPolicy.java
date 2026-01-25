package io.surfworks.warpforge.core.tensor.typed.numerical;

/**
 * Policy for handling NaN (Not a Number) and Inf (Infinity) values in tensors.
 *
 * <p>NaN and Inf values can propagate silently through neural network computations,
 * leading to garbage outputs. This policy controls what happens when such values
 * are detected:
 *
 * <ul>
 *   <li>{@link #ERROR} - Throw a {@link NumericalException} immediately</li>
 *   <li>{@link #WARN} - Log a warning and continue execution</li>
 *   <li>{@link #IGNORE} - No checking (best performance)</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Enable strict checking during debugging
 * try (var _ = new NumericalContext(NaNPolicy.ERROR)) {
 *     var result = MatrixOps.matmul(a, b);
 *     NumericalContext.check(result, "matmul");  // throws if NaN/Inf
 * }
 *
 * // Production: no overhead
 * try (var _ = new NumericalContext(NaNPolicy.IGNORE)) {
 *     var result = MatrixOps.matmul(a, b);
 *     // No checking performed
 * }
 * }</pre>
 *
 * @see NumericalContext
 * @see NumericalException
 */
public enum NaNPolicy {

    /**
     * Throw a {@link NumericalException} when NaN or Inf is detected.
     *
     * <p>Use this during development and debugging to catch numerical issues
     * at their source rather than letting them propagate.
     */
    ERROR,

    /**
     * Log a warning when NaN or Inf is detected, but continue execution.
     *
     * <p>Useful for monitoring production systems where you want visibility
     * into numerical issues without interrupting execution.
     */
    WARN,

    /**
     * Perform no NaN/Inf checking (default).
     *
     * <p>Use this for maximum performance when you're confident in numerical
     * stability or after debugging is complete.
     */
    IGNORE
}
