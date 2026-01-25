package io.surfworks.warpforge.core.tensor.typed.shape;

import io.surfworks.warpforge.core.tensor.typed.dim.Dim;

/**
 * Rank-3 tensor shape with compile-time dimension type parameters.
 *
 * <p>DimRank3 encodes three dimensions at the type level. Common uses include:
 * <ul>
 *   <li>Batch of sequences: [Batch, SeqLen, Hidden]
 *   <li>Single image: [Channels, Height, Width]
 *   <li>Batched matmul operands: [Batch, M, N]
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * interface Batch extends Dim {}
 * interface SeqLen extends Dim {}
 * interface Hidden extends Dim {}
 *
 * TypedTensor<DimRank3<Batch, SeqLen, Hidden>, F32, Cpu> hiddenStates = ...;
 * }</pre>
 *
 * @param <D0> dimension type for axis 0
 * @param <D1> dimension type for axis 1
 * @param <D2> dimension type for axis 2
 * @param dim0 runtime size of axis 0
 * @param dim1 runtime size of axis 1
 * @param dim2 runtime size of axis 2
 */
public record DimRank3<D0 extends Dim, D1 extends Dim, D2 extends Dim>(
        int dim0, int dim1, int dim2) implements Shape {

    /**
     * Creates a DimRank3 with the specified dimensions.
     *
     * @param dim0 size of axis 0 (must be non-negative)
     * @param dim1 size of axis 1 (must be non-negative)
     * @param dim2 size of axis 2 (must be non-negative)
     * @throws IllegalArgumentException if any dimension is negative
     */
    public DimRank3 {
        if (dim0 < 0) {
            throw new IllegalArgumentException("dim0 must be non-negative: " + dim0);
        }
        if (dim1 < 0) {
            throw new IllegalArgumentException("dim1 must be non-negative: " + dim1);
        }
        if (dim2 < 0) {
            throw new IllegalArgumentException("dim2 must be non-negative: " + dim2);
        }
    }

    @Override
    public int rank() {
        return 3;
    }

    @Override
    public int[] dimensions() {
        return new int[]{dim0, dim1, dim2};
    }

    @Override
    public boolean isFullyKnown() {
        return true;
    }

    @Override
    public long elementCount() {
        return (long) dim0 * dim1 * dim2;
    }

    /**
     * Transposes the last two dimensions, swapping D1 and D2.
     *
     * @return a new DimRank3 with the last two dimensions swapped
     */
    public DimRank3<D0, D2, D1> transposeLastTwo() {
        return new DimRank3<>(dim0, dim2, dim1);
    }

    @Override
    public String toString() {
        return "DimRank3[" + dim0 + ", " + dim1 + ", " + dim2 + "]";
    }
}
