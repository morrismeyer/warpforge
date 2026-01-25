package io.surfworks.warpforge.core.tensor.typed.shape;

/**
 * Phantom type representing tensor shape at compile time.
 *
 * <p>This sealed interface limits shape representations to known patterns,
 * enabling compile-time rank checking. The shape types carry dimension
 * information as record fields, but the primary purpose is type-level
 * differentiation between tensor ranks.
 *
 * <p>Example usage:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> weights = TypedTensor.zeros(
 *     new Matrix(768, 512), F32.INSTANCE, Cpu.INSTANCE);
 * }</pre>
 *
 * @see Scalar for 0D tensors
 * @see Vector for 1D tensors
 * @see Matrix for 2D tensors
 * @see Rank3 for 3D tensors
 * @see Rank4 for 4D tensors (e.g., NCHW image batches)
 * @see Dynamic for tensors with unknown shape
 */
public sealed interface Shape permits Scalar, Vector, Matrix, Rank3, Rank4, Dynamic {

    /**
     * Returns the rank (number of dimensions) for this shape type.
     *
     * @return the rank, or -1 for Dynamic shapes with unknown rank
     */
    int rank();

    /**
     * Returns the dimensions as an int array.
     *
     * @return the dimensions, or an empty array for scalars
     * @throws IllegalStateException if any dimension is unknown (negative)
     */
    int[] dimensions();

    /**
     * Returns true if all dimensions are known (non-negative).
     *
     * @return true if shape is fully specified
     */
    boolean isFullyKnown();

    /**
     * Computes the total number of elements for this shape.
     *
     * @return the element count (product of dimensions)
     * @throws IllegalStateException if any dimension is unknown
     */
    default long elementCount() {
        if (!isFullyKnown()) {
            throw new IllegalStateException("Cannot compute element count for shape with unknown dimensions");
        }
        int[] dims = dimensions();
        if (dims.length == 0) {
            return 1; // Scalar
        }
        long count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        return count;
    }
}
