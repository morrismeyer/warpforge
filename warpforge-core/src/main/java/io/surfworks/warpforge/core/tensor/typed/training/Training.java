package io.surfworks.warpforge.core.tensor.typed.training;

/**
 * Training mode for neural network execution.
 *
 * <p>In training mode:
 * <ul>
 *   <li>Dropout layers randomly drop activations</li>
 *   <li>Batch normalization uses batch statistics</li>
 *   <li>Data augmentation is active</li>
 *   <li>Gradients are computed and accumulated</li>
 * </ul>
 *
 * <p>Training mode is the default when no {@link ModeContext} is active.
 *
 * @see ModeContext
 * @see Inference
 */
public record Training() implements Mode {

    /**
     * Singleton instance for training mode.
     */
    public static final Training INSTANCE = new Training();

    @Override
    public boolean isTraining() {
        return true;
    }

    @Override
    public String modeName() {
        return "training";
    }
}
