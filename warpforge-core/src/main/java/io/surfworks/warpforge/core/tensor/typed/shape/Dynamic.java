package io.surfworks.warpforge.core.tensor.typed.shape;

import java.util.Arrays;

/**
 * Shape type for tensors with runtime-determined shape.
 *
 * <p>This is an escape hatch for interoperability with code that doesn't
 * know tensor shapes at compile time. It should be used sparingly, as it
 * opts out of compile-time shape checking.
 *
 * <p>Common uses:
 * <ul>
 *   <li>Loading tensors from files where shape isn't known until runtime</li>
 *   <li>Generic code that works with any tensor shape</li>
 *   <li>Transitioning from untyped to typed tensor code</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * // Load tensor with unknown shape
 * Tensor untyped = loadFromFile("weights.safetensors", "layer1.weight");
 * TypedTensor<Dynamic, F32, Cpu> dynamic = TypedTensor.fromDynamic(
 *     untyped, F32.INSTANCE, Cpu.INSTANCE);
 *
 * // Later, convert to known shape
 * if (dynamic.underlying().rank() == 2) {
 *     int[] dims = dynamic.underlying().shape();
 *     TypedTensor<Matrix, F32, Cpu> matrix = dynamic.reshape(
 *         new Matrix(dims[0], dims[1]));
 * }
 * }</pre>
 */
public final class Dynamic implements Shape {

    /**
     * Completely unknown shape (dimensions not known).
     */
    public static final Dynamic ANY = new Dynamic(new int[0], false);

    private final int[] dims;
    private final boolean known;

    /**
     * Creates a dynamic shape with known dimensions.
     *
     * @param dimensions the actual dimensions
     */
    public Dynamic(int... dimensions) {
        this.dims = dimensions.clone();
        this.known = true;
    }

    private Dynamic(int[] dimensions, boolean known) {
        this.dims = dimensions.clone();
        this.known = known;
    }

    /**
     * Creates a dynamic shape with known rank but unknown dimensions.
     *
     * @param rank the tensor rank
     * @return dynamic shape with specified rank
     */
    public static Dynamic ofRank(int rank) {
        if (rank < 0) {
            throw new IllegalArgumentException("Rank must be non-negative, got: " + rank);
        }
        return new Dynamic(new int[rank], false);
    }

    /**
     * Creates a dynamic shape from an existing dimension array.
     *
     * @param dimensions the dimensions
     * @return a new dynamic shape
     */
    public static Dynamic fromDimensions(int[] dimensions) {
        return new Dynamic(dimensions);
    }

    /**
     * Returns true if the dimensions are known.
     */
    public boolean isDimensionsKnown() {
        return known;
    }

    @Override
    public int rank() {
        return dims.length;
    }

    @Override
    public int[] dimensions() {
        if (!known) {
            throw new IllegalStateException("Dimensions not known for this dynamic shape");
        }
        return dims.clone();
    }

    @Override
    public boolean isFullyKnown() {
        return known;
    }

    @Override
    public long elementCount() {
        if (!known) {
            throw new IllegalStateException("Cannot compute element count when dimensions unknown");
        }
        long count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        return count;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Dynamic other)) return false;
        return known == other.known && Arrays.equals(dims, other.dims);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(dims) + Boolean.hashCode(known);
    }

    @Override
    public String toString() {
        if (known) {
            return "Dynamic" + Arrays.toString(dims);
        }
        return dims.length > 0 ? "Dynamic[rank=" + dims.length + "]" : "Dynamic[?]";
    }
}
