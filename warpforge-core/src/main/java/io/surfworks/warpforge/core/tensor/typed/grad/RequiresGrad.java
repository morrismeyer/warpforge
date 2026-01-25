package io.surfworks.warpforge.core.tensor.typed.grad;

/**
 * Gradient mode indicating that a tensor tracks gradients.
 *
 * <p>Tensors with this mode will accumulate gradients during the backward pass.
 * Operations like {@code zeroGrad()} and {@code grad()} are available on
 * {@code GradTensor<S, D, V, RequiresGrad>}.
 *
 * <p>This is the mode to use for model parameters during training:
 * <pre>{@code
 * var weights = GradTensor.requiresGrad(tensor);
 * weights.zeroGrad();  // OK - clears accumulated gradients
 * var g = weights.grad();  // OK - access gradient tensor
 * }</pre>
 */
public record RequiresGrad() implements GradMode {

    /**
     * Singleton instance for use as phantom type marker.
     */
    public static final RequiresGrad INSTANCE = new RequiresGrad();

    @Override
    public boolean tracksGradient() {
        return true;
    }

    @Override
    public String modeName() {
        return "requires_grad";
    }
}
