package io.surfworks.warpforge.core.tensor.typed.shape;

import io.surfworks.warpforge.core.tensor.typed.dim.Dim;

/**
 * Vector shape with compile-time dimension type parameter.
 *
 * <p>DimVector encodes the vector length dimension at the type level, enabling
 * compile-time verification of dimension compatibility in operations.
 *
 * <p>Example:
 * <pre>{@code
 * interface Hidden extends Dim {}
 * interface Vocab extends Dim {}
 *
 * TypedTensor<DimVector<Hidden>, F32, Cpu> hiddenVec = ...;
 * TypedTensor<DimVector<Vocab>, F32, Cpu> vocabVec = ...;
 *
 * // Operations that require matching dimensions will be type-checked
 * }</pre>
 *
 * @param <N> the dimension type for the vector length
 * @param length the runtime length value
 */
public record DimVector<N extends Dim>(int length) implements Shape {

    /**
     * Creates a DimVector with the specified length.
     *
     * @param length the vector length (must be non-negative)
     * @throws IllegalArgumentException if length is negative
     */
    public DimVector {
        if (length < 0) {
            throw new IllegalArgumentException("Vector length must be non-negative: " + length);
        }
    }

    @Override
    public int rank() {
        return 1;
    }

    @Override
    public int[] dimensions() {
        return new int[]{length};
    }

    @Override
    public boolean isFullyKnown() {
        return true;
    }

    @Override
    public long elementCount() {
        return length;
    }

    @Override
    public String toString() {
        return "DimVector[" + length + "]";
    }
}
