package io.surfworks.warpforge.core.tensor.typed.dim;

/**
 * Predefined semantic dimension markers for common ML patterns.
 *
 * <p>These markers provide meaningful names for tensor dimensions, making code
 * more readable and enabling compile-time shape checking.
 *
 * <p>Example usage:
 * <pre>{@code
 * import static io.surfworks.warpforge.core.tensor.typed.dim.Semantic.*;
 *
 * // Transformer hidden state: [batch, sequence, hidden]
 * TypedTensor<DimRank3<Batch, SeqLen, Hidden>, F32, Cpu> hiddenState = ...;
 *
 * // Attention weights: [batch, heads, query_len, key_len]
 * TypedTensor<DimRank4<Batch, NumHeads, QueryLen, KeyLen>, F32, Cpu> attnWeights = ...;
 * }</pre>
 */
public final class Semantic {

    private Semantic() {
        // Utility class - no instantiation
    }

    // ==================== Common Dimensions ====================

    /**
     * Batch dimension - number of samples processed together.
     */
    public interface Batch extends Dim {}

    /**
     * Sequence length dimension - number of tokens in a sequence.
     */
    public interface SeqLen extends Dim {}

    /**
     * Hidden dimension - size of hidden representations.
     */
    public interface Hidden extends Dim {}

    /**
     * Vocabulary dimension - size of token vocabulary.
     */
    public interface Vocab extends Dim {}

    /**
     * Embedding dimension - size of embedding vectors.
     */
    public interface Embed extends Dim {}

    /**
     * Feature dimension - generic feature size.
     */
    public interface Features extends Dim {}

    // ==================== Attention Dimensions ====================

    /**
     * Number of attention heads.
     */
    public interface NumHeads extends Dim {}

    /**
     * Per-head dimension size.
     */
    public interface HeadDim extends Dim {}

    /**
     * Query sequence length (may differ from key/value length in cross-attention).
     */
    public interface QueryLen extends Dim {}

    /**
     * Key sequence length.
     */
    public interface KeyLen extends Dim {}

    /**
     * Value sequence length.
     */
    public interface ValueLen extends Dim {}

    // ==================== CNN Dimensions ====================

    /**
     * Number of channels (e.g., RGB=3, feature maps).
     */
    public interface Channels extends Dim {}

    /**
     * Height dimension (spatial).
     */
    public interface Height extends Dim {}

    /**
     * Width dimension (spatial).
     */
    public interface Width extends Dim {}

    /**
     * Depth dimension (for 3D convolutions).
     */
    public interface Depth extends Dim {}

    /**
     * Number of convolutional filters/kernels.
     */
    public interface Filters extends Dim {}

    // ==================== Generic Named Dimensions ====================

    /**
     * Generic dimension M (for ad-hoc use in matrix operations).
     */
    public interface M extends Dim {}

    /**
     * Generic dimension N (for ad-hoc use in matrix operations).
     */
    public interface N extends Dim {}

    /**
     * Generic dimension K (for ad-hoc use, typically inner dimension in matmul).
     */
    public interface K extends Dim {}

    /**
     * Generic dimension P (for ad-hoc use).
     */
    public interface P extends Dim {}

    /**
     * Generic dimension Q (for ad-hoc use).
     */
    public interface Q extends Dim {}

    /**
     * Generic dimension R (for ad-hoc use).
     */
    public interface R extends Dim {}

    // ==================== RNN/Sequence Dimensions ====================

    /**
     * Number of RNN layers.
     */
    public interface NumLayers extends Dim {}

    /**
     * Number of directions (1 for unidirectional, 2 for bidirectional).
     */
    public interface Directions extends Dim {}

    /**
     * Time steps dimension.
     */
    public interface TimeSteps extends Dim {}
}
