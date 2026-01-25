package io.surfworks.warpforge.core.tensor.typed.grad;

/**
 * Sealed interface for gradient tracking modes.
 *
 * <p>GradMode is a phantom type parameter that encodes at compile time whether
 * a tensor tracks gradients. This enables type-safe APIs where methods like
 * {@code zeroGrad()} are only available on tensors that track gradients.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link RequiresGrad} - Tensor accumulates gradients during backward pass</li>
 *   <li>{@link NoGrad} - Tensor does not track gradients (inference mode)</li>
 *   <li>{@link Detached} - Tensor was detached from computation graph</li>
 * </ul>
 */
public sealed interface GradMode permits RequiresGrad, NoGrad, Detached {

    /**
     * Returns true if this mode tracks gradients.
     *
     * @return true for RequiresGrad, false for NoGrad and Detached
     */
    boolean tracksGradient();

    /**
     * Returns a human-readable name for this mode.
     *
     * @return the mode name (e.g., "requires_grad", "no_grad", "detached")
     */
    String modeName();
}
