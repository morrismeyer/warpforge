package io.surfworks.warpforge.core.tensor.typed.training;

/**
 * Sealed interface for training/inference mode selection.
 *
 * <p>Mode determines the behavior of stochastic operations during execution:
 * <ul>
 *   <li>{@link Training} - Enables dropout, uses batch statistics for normalization</li>
 *   <li>{@link Inference} - Disables dropout, uses running statistics for normalization</li>
 * </ul>
 *
 * <p>Mode is managed via {@link ModeContext}, which provides thread-local scoping:
 * <pre>{@code
 * // Default is Training mode
 * assert ModeContext.isTraining();
 *
 * // Switch to Inference mode
 * try (var _ = new ModeContext(Inference.INSTANCE)) {
 *     assert !ModeContext.isTraining();
 *     var output = model.forward(input);  // dropout disabled
 * }
 *
 * // Back to Training mode
 * assert ModeContext.isTraining();
 * }</pre>
 */
public sealed interface Mode permits Training, Inference {

    /**
     * Returns true if this mode is training mode.
     *
     * @return true for Training, false for Inference
     */
    boolean isTraining();

    /**
     * Returns a human-readable name for this mode.
     *
     * @return the mode name ("training" or "inference")
     */
    String modeName();
}
