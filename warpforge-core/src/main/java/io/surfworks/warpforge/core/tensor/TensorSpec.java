package io.surfworks.warpforge.core.tensor;

import io.surfworks.snakeburger.stablehlo.StableHloAst;

import java.util.Arrays;

/**
 * Tensor specification: shape, dtype, and computed strides.
 * Immutable metadata describing tensor layout.
 */
public record TensorSpec(
    int[] shape,
    ScalarType dtype,
    long[] strides
) {
    /**
     * Create a TensorSpec with row-major (C-contiguous) strides.
     */
    public static TensorSpec of(ScalarType dtype, int... shape) {
        long[] strides = computeRowMajorStrides(shape);
        return new TensorSpec(shape.clone(), dtype, strides);
    }

    /**
     * Create from StableHLO AST TensorType.
     */
    public static TensorSpec fromAst(StableHloAst.TensorType tensorType) {
        int[] shape = tensorType.shape().stream().mapToInt(Integer::intValue).toArray();
        ScalarType dtype = ScalarType.fromAst(tensorType.elementType());
        return of(dtype, shape);
    }

    /**
     * Create a TensorSpec with explicit strides (e.g., for column-major layout).
     */
    public static TensorSpec withStrides(ScalarType dtype, int[] shape, long[] strides) {
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("Shape and strides must have same length");
        }
        return new TensorSpec(shape.clone(), dtype, strides.clone());
    }

    /**
     * Number of dimensions.
     */
    public int rank() {
        return shape.length;
    }

    /**
     * Total number of elements.
     */
    public long elementCount() {
        if (shape.length == 0) {
            return 1; // Scalar tensor
        }
        long count = 1;
        for (int dim : shape) {
            count *= dim;
        }
        return count;
    }

    /**
     * Total size in bytes.
     */
    public long byteSize() {
        return elementCount() * dtype.byteSize();
    }

    /**
     * Compute flat index from multi-dimensional indices.
     */
    public long flatIndex(long... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "Expected " + shape.length + " indices, got " + indices.length);
        }
        long idx = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Index " + indices[i] + " out of bounds for dimension " + i + " with size " + shape[i]);
            }
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    /**
     * Compute flat index from multi-dimensional int indices.
     */
    public long flatIndex(int... indices) {
        long[] longIndices = new long[indices.length];
        for (int i = 0; i < indices.length; i++) {
            longIndices[i] = indices[i];
        }
        return flatIndex(longIndices);
    }

    /**
     * Check if this tensor is contiguous in memory.
     */
    public boolean isContiguous() {
        if (shape.length == 0) {
            return true;
        }
        long expectedStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape[i];
        }
        return true;
    }

    /**
     * Check if shapes are equal.
     */
    public boolean shapeEquals(TensorSpec other) {
        return Arrays.equals(this.shape, other.shape);
    }

    /**
     * Check if shapes are broadcastable.
     */
    public boolean isBroadcastableWith(TensorSpec other) {
        int maxRank = Math.max(this.rank(), other.rank());
        for (int i = 0; i < maxRank; i++) {
            int thisDim = i < this.rank() ? this.shape[this.rank() - 1 - i] : 1;
            int otherDim = i < other.rank() ? other.shape[other.rank() - 1 - i] : 1;
            if (thisDim != otherDim && thisDim != 1 && otherDim != 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * Compute row-major (C-contiguous) strides for a shape.
     */
    public static long[] computeRowMajorStrides(int[] shape) {
        if (shape.length == 0) {
            return new long[0];
        }
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /**
     * Compute column-major (Fortran-contiguous) strides for a shape.
     */
    public static long[] computeColumnMajorStrides(int[] shape) {
        if (shape.length == 0) {
            return new long[0];
        }
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = 0; i < shape.length; i++) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TensorSpec that)) return false;
        return Arrays.equals(shape, that.shape) &&
               dtype == that.dtype &&
               Arrays.equals(strides, that.strides);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + dtype.hashCode();
        result = 31 * result + Arrays.hashCode(strides);
        return result;
    }

    @Override
    public String toString() {
        return "TensorSpec[shape=" + Arrays.toString(shape) +
               ", dtype=" + dtype +
               ", strides=" + Arrays.toString(strides) + "]";
    }
}
