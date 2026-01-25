package io.surfworks.warpforge.core.tensor.typed.shape;

/**
 * Shape type for 1-dimensional tensors (vectors).
 *
 * <p>A vector has a single dimension representing its length.
 * Use {@code length = -1} to indicate a dynamic/unknown length.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Vector, F32, Cpu> embedding = TypedTensor.zeros(
 *     new Vector(768), F32.INSTANCE, Cpu.INSTANCE);
 * }</pre>
 *
 * @param length the length of the vector, or -1 if unknown
 */
public record Vector(int length) implements Shape {

    /**
     * Vector with unknown length (dynamic).
     */
    public static final Vector DYNAMIC = new Vector(-1);

    /**
     * Creates a vector shape with the given length.
     *
     * @param length the length (must be non-negative or -1 for dynamic)
     * @throws IllegalArgumentException if length is invalid
     */
    public Vector {
        if (length < -1) {
            throw new IllegalArgumentException("Vector length must be non-negative or -1 for dynamic, got: " + length);
        }
    }

    /**
     * Returns true if the length is known (non-negative).
     */
    public boolean isLengthKnown() {
        return length >= 0;
    }

    @Override
    public int rank() {
        return 1;
    }

    @Override
    public int[] dimensions() {
        if (!isLengthKnown()) {
            throw new IllegalStateException("Cannot get dimensions for dynamic vector");
        }
        return new int[]{length};
    }

    @Override
    public boolean isFullyKnown() {
        return length >= 0;
    }

    @Override
    public String toString() {
        return length >= 0 ? "Vector[" + length + "]" : "Vector[?]";
    }
}
