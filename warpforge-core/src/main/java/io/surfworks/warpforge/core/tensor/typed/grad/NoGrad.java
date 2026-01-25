package io.surfworks.warpforge.core.tensor.typed.grad;

/**
 * Gradient mode indicating that a tensor does not track gradients.
 *
 * <p>Tensors with this mode will not accumulate gradients. This is the default
 * mode for inference and for tensors that should not participate in gradient
 * computation (e.g., input data, labels).
 *
 * <p>Operations like {@code zeroGrad()} and {@code grad()} are NOT available
 * on {@code GradTensor<S, D, V, NoGrad>} - they will throw at runtime if called.
 *
 * <p>Example:
 * <pre>{@code
 * var input = GradTensor.noGrad(tensor);
 * // input.zeroGrad();  // Throws IllegalStateException
 * // input.grad();  // Throws IllegalStateException
 * }</pre>
 */
public record NoGrad() implements GradMode {

    /**
     * Singleton instance for use as phantom type marker.
     */
    public static final NoGrad INSTANCE = new NoGrad();

    @Override
    public boolean tracksGradient() {
        return false;
    }

    @Override
    public String modeName() {
        return "no_grad";
    }
}
