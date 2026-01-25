package io.surfworks.warpforge.core.tensor.typed.shape;

/**
 * Shape type for 4-dimensional tensors.
 *
 * <p>Common uses include:
 * <ul>
 *   <li>Image batches (NCHW): [batch, channels, height, width]</li>
 *   <li>Image batches (NHWC): [batch, height, width, channels]</li>
 *   <li>Attention tensors: [batch, heads, seq_len, head_dim]</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Rank4, F32, Nvidia> images = TypedTensor.zeros(
 *     Rank4.nchw(32, 3, 224, 224), F32.INSTANCE, Nvidia.DEFAULT);
 * }</pre>
 *
 * @param dim0 first dimension (typically batch), or -1 if unknown
 * @param dim1 second dimension (typically channels or height), or -1 if unknown
 * @param dim2 third dimension (typically height or width), or -1 if unknown
 * @param dim3 fourth dimension (typically width or channels), or -1 if unknown
 */
public record Rank4(int dim0, int dim1, int dim2, int dim3) implements Shape {

    /**
     * Rank-4 tensor with unknown dimensions (fully dynamic).
     */
    public static final Rank4 DYNAMIC = new Rank4(-1, -1, -1, -1);

    /**
     * Creates a rank-4 shape with the given dimensions.
     *
     * @param dim0 first dimension (must be positive or -1 for dynamic)
     * @param dim1 second dimension (must be positive or -1 for dynamic)
     * @param dim2 third dimension (must be positive or -1 for dynamic)
     * @param dim3 fourth dimension (must be positive or -1 for dynamic)
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public Rank4 {
        if ((dim0 < -1 || dim0 == 0) || (dim1 < -1 || dim1 == 0) ||
            (dim2 < -1 || dim2 == 0) || (dim3 < -1 || dim3 == 0)) {
            throw new IllegalArgumentException(
                "Dimensions must be positive or -1 for dynamic, got: [" +
                dim0 + ", " + dim1 + ", " + dim2 + ", " + dim3 + "]");
        }
    }

    /**
     * Creates an NCHW shape (batch, channels, height, width).
     * Standard layout for PyTorch and many CUDA kernels.
     *
     * @param n batch size
     * @param c channels
     * @param h height
     * @param w width
     * @return NCHW shape
     */
    public static Rank4 nchw(int n, int c, int h, int w) {
        return new Rank4(n, c, h, w);
    }

    /**
     * Creates an NCHW shape with dynamic batch size.
     *
     * @param c channels
     * @param h height
     * @param w width
     * @return NCHW shape with dynamic N
     */
    public static Rank4 nchwDynamicBatch(int c, int h, int w) {
        return new Rank4(-1, c, h, w);
    }

    /**
     * Creates an NHWC shape (batch, height, width, channels).
     * Standard layout for TensorFlow.
     *
     * @param n batch size
     * @param h height
     * @param w width
     * @param c channels
     * @return NHWC shape
     */
    public static Rank4 nhwc(int n, int h, int w, int c) {
        return new Rank4(n, h, w, c);
    }

    /**
     * Creates an attention tensor shape [batch, heads, seq_len, head_dim].
     *
     * @param batch batch size
     * @param heads number of attention heads
     * @param seqLen sequence length
     * @param headDim dimension per head
     * @return attention shape
     */
    public static Rank4 attention(int batch, int heads, int seqLen, int headDim) {
        return new Rank4(batch, heads, seqLen, headDim);
    }

    @Override
    public int rank() {
        return 4;
    }

    @Override
    public int[] dimensions() {
        if (!isFullyKnown()) {
            throw new IllegalStateException("Cannot get dimensions for rank-4 tensor with unknown dimensions");
        }
        return new int[]{dim0, dim1, dim2, dim3};
    }

    @Override
    public boolean isFullyKnown() {
        return dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0;
    }

    @Override
    public String toString() {
        String d0 = dim0 > 0 ? String.valueOf(dim0) : "?";
        String d1 = dim1 > 0 ? String.valueOf(dim1) : "?";
        String d2 = dim2 > 0 ? String.valueOf(dim2) : "?";
        String d3 = dim3 > 0 ? String.valueOf(dim3) : "?";
        return "Rank4[" + d0 + ", " + d1 + ", " + d2 + ", " + d3 + "]";
    }
}
