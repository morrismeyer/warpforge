package io.surfworks.warpforge.core.tensor.typed.shape;

import io.surfworks.warpforge.core.tensor.typed.dim.Dim;

/**
 * Rank-4 tensor shape with compile-time dimension type parameters.
 *
 * <p>DimRank4 encodes four dimensions at the type level. Common uses include:
 * <ul>
 *   <li>Image batches (NCHW): [Batch, Channels, Height, Width]
 *   <li>Multi-head attention: [Batch, NumHeads, SeqLen, HeadDim]
 *   <li>Batched matrix operations: [Batch, Heads, M, N]
 * </ul>
 *
 * <p>Example for multi-head attention:
 * <pre>{@code
 * interface Batch extends Dim {}
 * interface NumHeads extends Dim {}
 * interface SeqLen extends Dim {}
 * interface HeadDim extends Dim {}
 *
 * TypedTensor<DimRank4<Batch, NumHeads, SeqLen, HeadDim>, F32, Cpu> Q = ...;
 * TypedTensor<DimRank4<Batch, NumHeads, HeadDim, SeqLen>, F32, Cpu> K_T = ...;
 *
 * // Q @ K^T produces [Batch, NumHeads, SeqLen, SeqLen]
 * TypedTensor<DimRank4<Batch, NumHeads, SeqLen, SeqLen>, F32, Cpu> scores =
 *     DimOps.batchedMatmulRank4(Q, K_T);
 * }</pre>
 *
 * @param <D0> dimension type for axis 0 (typically batch)
 * @param <D1> dimension type for axis 1 (typically heads/channels)
 * @param <D2> dimension type for axis 2
 * @param <D3> dimension type for axis 3
 * @param dim0 runtime size of axis 0
 * @param dim1 runtime size of axis 1
 * @param dim2 runtime size of axis 2
 * @param dim3 runtime size of axis 3
 */
public record DimRank4<D0 extends Dim, D1 extends Dim, D2 extends Dim, D3 extends Dim>(
        int dim0, int dim1, int dim2, int dim3) implements Shape {

    /**
     * Creates a DimRank4 with the specified dimensions.
     *
     * @param dim0 size of axis 0 (must be non-negative)
     * @param dim1 size of axis 1 (must be non-negative)
     * @param dim2 size of axis 2 (must be non-negative)
     * @param dim3 size of axis 3 (must be non-negative)
     * @throws IllegalArgumentException if any dimension is negative
     */
    public DimRank4 {
        if (dim0 < 0) {
            throw new IllegalArgumentException("dim0 must be non-negative: " + dim0);
        }
        if (dim1 < 0) {
            throw new IllegalArgumentException("dim1 must be non-negative: " + dim1);
        }
        if (dim2 < 0) {
            throw new IllegalArgumentException("dim2 must be non-negative: " + dim2);
        }
        if (dim3 < 0) {
            throw new IllegalArgumentException("dim3 must be non-negative: " + dim3);
        }
    }

    @Override
    public int rank() {
        return 4;
    }

    @Override
    public int[] dimensions() {
        return new int[]{dim0, dim1, dim2, dim3};
    }

    @Override
    public boolean isFullyKnown() {
        return true;
    }

    @Override
    public long elementCount() {
        return (long) dim0 * dim1 * dim2 * dim3;
    }

    /**
     * Transposes the last two dimensions, swapping D2 and D3.
     *
     * <p>This is commonly used in attention to transpose K: [B, H, S, D] -> [B, H, D, S]
     *
     * @return a new DimRank4 with the last two dimensions swapped
     */
    public DimRank4<D0, D1, D3, D2> transposeLastTwo() {
        return new DimRank4<>(dim0, dim1, dim3, dim2);
    }

    @Override
    public String toString() {
        return "DimRank4[" + dim0 + ", " + dim1 + ", " + dim2 + ", " + dim3 + "]";
    }
}
