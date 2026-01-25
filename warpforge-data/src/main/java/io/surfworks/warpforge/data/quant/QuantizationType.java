package io.surfworks.warpforge.data.quant;

/**
 * Supported quantization types for model weights.
 *
 * <p>These correspond to common quantization formats used in GGUF and other
 * quantized model formats.
 */
public enum QuantizationType {

    /** Full 32-bit floating point (no quantization) */
    F32(32, false),

    /** 16-bit floating point */
    F16(16, false),

    /** Brain floating point (16-bit) */
    BF16(16, false),

    /** 8-bit integer quantization */
    Q8_0(8, true),

    /** 8-bit integer with per-channel scaling */
    Q8_1(8, true),

    /** 4-bit integer quantization (32 values per block) */
    Q4_0(4, true),

    /** 4-bit integer with per-block offset */
    Q4_1(4, true),

    /** 4-bit integer, K-quant variant */
    Q4_K(4, true),

    /** 5-bit integer, K-quant variant */
    Q5_K(5, true),

    /** 6-bit integer, K-quant variant */
    Q6_K(6, true),

    /** 2-bit integer quantization */
    Q2_K(2, true),

    /** 3-bit integer quantization */
    Q3_K(3, true),

    /** INT8 symmetric quantization */
    INT8(8, true),

    /** INT4 symmetric quantization */
    INT4(4, true),

    /** NF4 (4-bit normal float) used by QLoRA */
    NF4(4, true),

    /** FP4 (4-bit floating point) */
    FP4(4, true);

    private final int bits;
    private final boolean quantized;

    QuantizationType(int bits, boolean quantized) {
        this.bits = bits;
        this.quantized = quantized;
    }

    /**
     * Number of bits per weight element.
     */
    public int bits() {
        return bits;
    }

    /**
     * Whether this is a quantized format (vs full precision).
     */
    public boolean isQuantized() {
        return quantized;
    }

    /**
     * Compression ratio compared to FP32.
     */
    public double compressionRatio() {
        return 32.0 / bits;
    }

    /**
     * Bytes per element (for non-block quantization).
     * For block quantization, use bytesPerBlock() instead.
     */
    public double bytesPerElement() {
        return bits / 8.0;
    }
}
