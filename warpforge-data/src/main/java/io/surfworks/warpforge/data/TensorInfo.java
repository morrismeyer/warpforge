package io.surfworks.warpforge.data;

import java.util.Arrays;
import java.util.List;

/**
 * Immutable metadata about a tensor in a model file.
 *
 * @param name   Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
 * @param dtype  Data type
 * @param shape  Shape dimensions
 * @param offset Byte offset in the data section
 * @param size   Size in bytes
 */
public record TensorInfo(
        String name,
        DType dtype,
        long[] shape,
        long offset,
        long size
) {
    /**
     * Total number of elements in the tensor.
     */
    public long elementCount() {
        if (shape.length == 0) {
            return 1; // scalar
        }
        long count = 1;
        for (long dim : shape) {
            count *= dim;
        }
        return count;
    }

    /**
     * Calculate expected byte size from shape and dtype.
     */
    public long expectedByteSize() {
        if (dtype.isQuantized()) {
            throw new UnsupportedOperationException("Cannot calculate byte size for quantized tensor: " + dtype);
        }
        return elementCount() * dtype.byteSize();
    }

    /**
     * Number of dimensions.
     */
    public int rank() {
        return shape.length;
    }

    /**
     * Shape as a list (for convenience).
     */
    public List<Long> shapeAsList() {
        return Arrays.stream(shape).boxed().toList();
    }

    @Override
    public String toString() {
        return String.format("TensorInfo[%s: %s %s, offset=%d, size=%d]",
                name, dtype, Arrays.toString(shape), offset, size);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof TensorInfo other)) return false;
        return name.equals(other.name)
                && dtype == other.dtype
                && Arrays.equals(shape, other.shape)
                && offset == other.offset
                && size == other.size;
    }

    @Override
    public int hashCode() {
        int result = name.hashCode();
        result = 31 * result + dtype.hashCode();
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Long.hashCode(offset);
        result = 31 * result + Long.hashCode(size);
        return result;
    }
}
