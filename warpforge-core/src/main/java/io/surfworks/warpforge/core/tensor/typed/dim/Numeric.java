package io.surfworks.warpforge.core.tensor.typed.dim;

/**
 * Predefined numeric dimension markers for common sizes.
 *
 * <p>Use these when you know the exact dimension value at compile time:
 * <pre>{@code
 * // Fixed batch size of 32, BERT hidden size of 768
 * TypedTensor<DimMatrix<Numeric._32, Numeric._768>, F32, Cpu> input = ...;
 * }</pre>
 *
 * <p>Numeric markers are useful for:
 * <ul>
 *   <li>Fixed architecture dimensions (e.g., BERT hidden=768)
 *   <li>Known batch sizes in production
 *   <li>Compile-time verification of dimension values
 * </ul>
 *
 * <p>For variable dimensions (like dynamic batch sizes), use {@link Semantic}
 * markers instead.
 */
public final class Numeric {

    private Numeric() {
        // Utility class - no instantiation
    }

    // ==================== Small Dimensions ====================

    /**
     * Dimension of size 1.
     */
    public interface _1 extends Dim {}

    /**
     * Dimension of size 2.
     */
    public interface _2 extends Dim {}

    /**
     * Dimension of size 3.
     */
    public interface _3 extends Dim {}

    /**
     * Dimension of size 4.
     */
    public interface _4 extends Dim {}

    /**
     * Dimension of size 8.
     */
    public interface _8 extends Dim {}

    /**
     * Dimension of size 12 (common for attention heads in base models).
     */
    public interface _12 extends Dim {}

    /**
     * Dimension of size 16.
     */
    public interface _16 extends Dim {}

    // ==================== Powers of 2 ====================

    /**
     * Dimension of size 32.
     */
    public interface _32 extends Dim {}

    /**
     * Dimension of size 64.
     */
    public interface _64 extends Dim {}

    /**
     * Dimension of size 128.
     */
    public interface _128 extends Dim {}

    /**
     * Dimension of size 256.
     */
    public interface _256 extends Dim {}

    /**
     * Dimension of size 512.
     */
    public interface _512 extends Dim {}

    /**
     * Dimension of size 1024.
     */
    public interface _1024 extends Dim {}

    /**
     * Dimension of size 2048.
     */
    public interface _2048 extends Dim {}

    /**
     * Dimension of size 4096.
     */
    public interface _4096 extends Dim {}

    /**
     * Dimension of size 8192.
     */
    public interface _8192 extends Dim {}

    /**
     * Dimension of size 16384.
     */
    public interface _16384 extends Dim {}

    // ==================== Common Transformer Sizes ====================

    /**
     * Dimension of size 768 (BERT-base hidden size).
     */
    public interface _768 extends Dim {}

    /**
     * Dimension of size 1536 (GPT-2 medium hidden size).
     */
    public interface _1536 extends Dim {}

    /**
     * Dimension of size 3072 (BERT-base intermediate size).
     */
    public interface _3072 extends Dim {}

    /**
     * Dimension of size 12288 (GPT-3 intermediate size).
     */
    public interface _12288 extends Dim {}

    // ==================== Common Vocabulary Sizes ====================

    /**
     * Dimension of size 30522 (BERT vocabulary size).
     */
    public interface _30522 extends Dim {}

    /**
     * Dimension of size 50257 (GPT-2 vocabulary size).
     */
    public interface _50257 extends Dim {}

    /**
     * Dimension of size 50304 (common rounded vocab size).
     */
    public interface _50304 extends Dim {}

    /**
     * Dimension of size 32000 (LLaMA vocabulary size).
     */
    public interface _32000 extends Dim {}

    // ==================== Common Image Sizes ====================

    /**
     * Dimension of size 224 (ImageNet standard).
     */
    public interface _224 extends Dim {}

    /**
     * Dimension of size 384 (ViT-Large input).
     */
    public interface _384 extends Dim {}
}
