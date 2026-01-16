package io.surfworks.warpforge.launch.job;

import java.util.Arrays;
import java.util.Objects;

/**
 * Specification for an input tensor.
 * Mirrors snakegrinder's input specification format.
 *
 * @param shape Tensor dimensions (e.g., [1, 3, 224, 224] for a batch of images)
 * @param dtype Data type string: "f32", "f64", "i32", "i64", "bf16", etc.
 */
public record InputSpec(int[] shape, String dtype) {

    public InputSpec {
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(dtype, "dtype cannot be null");
        if (shape.length == 0) {
            throw new IllegalArgumentException("shape must have at least one dimension");
        }
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("all dimensions must be positive: " + Arrays.toString(shape));
            }
        }
    }

    /**
     * Creates a float32 input spec with the given shape.
     */
    public static InputSpec f32(int... shape) {
        return new InputSpec(shape, "f32");
    }

    /**
     * Creates a float64 input spec with the given shape.
     */
    public static InputSpec f64(int... shape) {
        return new InputSpec(shape, "f64");
    }

    /**
     * Creates an int32 input spec with the given shape.
     */
    public static InputSpec i32(int... shape) {
        return new InputSpec(shape, "i32");
    }

    /**
     * Creates an int64 input spec with the given shape.
     */
    public static InputSpec i64(int... shape) {
        return new InputSpec(shape, "i64");
    }

    /**
     * Returns the total number of elements in this tensor.
     */
    public long elementCount() {
        long count = 1;
        for (int dim : shape) {
            count *= dim;
        }
        return count;
    }

    /**
     * Formats as snakegrinder-compatible string: "(1,3,224,224):f32"
     */
    public String toSpecString() {
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        sb.append("):").append(dtype);
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof InputSpec that)) return false;
        return Arrays.equals(shape, that.shape) && dtype.equals(that.dtype);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(shape) + dtype.hashCode();
    }

    @Override
    public String toString() {
        return toSpecString();
    }
}
