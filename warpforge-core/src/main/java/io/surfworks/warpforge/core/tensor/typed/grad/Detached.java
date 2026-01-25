package io.surfworks.warpforge.core.tensor.typed.grad;

/**
 * Gradient mode indicating that a tensor has been detached from the computation graph.
 *
 * <p>A detached tensor was previously part of a computation graph but has been
 * explicitly removed. Unlike {@link NoGrad}, which indicates a tensor that never
 * tracked gradients, {@link Detached} indicates a tensor that was deliberately
 * frozen to prevent gradient flow.
 *
 * <p>Common use cases:
 * <ul>
 *   <li>Freezing part of a model during fine-tuning</li>
 *   <li>Creating a copy that doesn't affect the original's gradients</li>
 *   <li>Breaking gradient chains to reduce memory usage</li>
 * </ul>
 *
 * <p>Like {@link NoGrad}, gradient operations throw on detached tensors:
 * <pre>{@code
 * var frozen = gradTensor.detach();
 * // frozen.zeroGrad();  // Throws IllegalStateException
 * // frozen.grad();  // Throws IllegalStateException
 * }</pre>
 */
public record Detached() implements GradMode {

    /**
     * Singleton instance for use as phantom type marker.
     */
    public static final Detached INSTANCE = new Detached();

    @Override
    public boolean tracksGradient() {
        return false;
    }

    @Override
    public String modeName() {
        return "detached";
    }
}
