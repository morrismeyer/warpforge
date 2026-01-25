package io.surfworks.warpforge.core.tensor.typed.training;

/**
 * Inference mode for neural network execution.
 *
 * <p>In inference mode:
 * <ul>
 *   <li>Dropout layers pass through all activations (no dropout)</li>
 *   <li>Batch normalization uses running/estimated statistics</li>
 *   <li>Data augmentation is disabled</li>
 *   <li>No gradients are computed (for efficiency)</li>
 * </ul>
 *
 * <p>Use inference mode during evaluation, testing, and production deployment.
 *
 * <p>Example:
 * <pre>{@code
 * try (var _ = new ModeContext(Inference.INSTANCE)) {
 *     var output = model.forward(input);  // dropout disabled
 *     var prediction = output.argmax();
 * }
 * }</pre>
 *
 * @see ModeContext
 * @see Training
 */
public record Inference() implements Mode {

    /**
     * Singleton instance for inference mode.
     */
    public static final Inference INSTANCE = new Inference();

    @Override
    public boolean isTraining() {
        return false;
    }

    @Override
    public String modeName() {
        return "inference";
    }
}
