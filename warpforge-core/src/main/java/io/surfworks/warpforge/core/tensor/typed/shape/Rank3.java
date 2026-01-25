package io.surfworks.warpforge.core.tensor.typed.shape;

/**
 * Shape type for 3-dimensional tensors.
 *
 * <p>Common uses include:
 * <ul>
 *   <li>Sequence data: [batch, sequence_length, features]</li>
 *   <li>Single images: [channels, height, width]</li>
 *   <li>Audio: [batch, channels, samples]</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Rank3, F32, Cpu> embeddings = TypedTensor.zeros(
 *     new Rank3(32, 128, 768), F32.INSTANCE, Cpu.INSTANCE);  // [batch, seq, hidden]
 * }</pre>
 *
 * @param dim0 first dimension, or -1 if unknown
 * @param dim1 second dimension, or -1 if unknown
 * @param dim2 third dimension, or -1 if unknown
 */
public record Rank3(int dim0, int dim1, int dim2) implements Shape {

    /**
     * Rank-3 tensor with unknown dimensions (fully dynamic).
     */
    public static final Rank3 DYNAMIC = new Rank3(-1, -1, -1);

    /**
     * Creates a rank-3 shape with the given dimensions.
     *
     * @param dim0 first dimension (must be positive or -1 for dynamic)
     * @param dim1 second dimension (must be positive or -1 for dynamic)
     * @param dim2 third dimension (must be positive or -1 for dynamic)
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public Rank3 {
        if ((dim0 < -1 || dim0 == 0) || (dim1 < -1 || dim1 == 0) || (dim2 < -1 || dim2 == 0)) {
            throw new IllegalArgumentException(
                "Dimensions must be positive or -1 for dynamic, got: [" + dim0 + ", " + dim1 + ", " + dim2 + "]");
        }
    }

    /**
     * Creates a rank-3 shape for sequence data with dynamic batch size.
     *
     * @param seqLen sequence length
     * @param features feature dimension
     * @return shape [?, seqLen, features]
     */
    public static Rank3 sequenceWithDynamicBatch(int seqLen, int features) {
        return new Rank3(-1, seqLen, features);
    }

    @Override
    public int rank() {
        return 3;
    }

    @Override
    public int[] dimensions() {
        if (!isFullyKnown()) {
            throw new IllegalStateException("Cannot get dimensions for rank-3 tensor with unknown dimensions");
        }
        return new int[]{dim0, dim1, dim2};
    }

    @Override
    public boolean isFullyKnown() {
        return dim0 > 0 && dim1 > 0 && dim2 > 0;
    }

    @Override
    public String toString() {
        String d0 = dim0 > 0 ? String.valueOf(dim0) : "?";
        String d1 = dim1 > 0 ? String.valueOf(dim1) : "?";
        String d2 = dim2 > 0 ? String.valueOf(dim2) : "?";
        return "Rank3[" + d0 + ", " + d1 + ", " + d2 + "]";
    }
}
