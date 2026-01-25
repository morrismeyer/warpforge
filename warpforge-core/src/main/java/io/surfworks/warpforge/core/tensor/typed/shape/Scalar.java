package io.surfworks.warpforge.core.tensor.typed.shape;

/**
 * Shape type for 0-dimensional tensors (single values).
 *
 * <p>A scalar tensor contains exactly one element and has no dimensions.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Scalar, F32, Cpu> loss = computeLoss(predictions, targets);
 * float lossValue = loss.underlying().getFloat();
 * }</pre>
 */
public record Scalar() implements Shape {

    /**
     * Singleton instance for scalar shape.
     */
    public static final Scalar INSTANCE = new Scalar();

    @Override
    public int rank() {
        return 0;
    }

    @Override
    public int[] dimensions() {
        return new int[0];
    }

    @Override
    public boolean isFullyKnown() {
        return true;
    }

    @Override
    public long elementCount() {
        return 1;
    }

    @Override
    public String toString() {
        return "Scalar[]";
    }
}
