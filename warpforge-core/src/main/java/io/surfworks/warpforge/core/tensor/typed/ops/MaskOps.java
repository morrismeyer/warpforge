package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe attention mask operations for transformers.
 *
 * <p>These operations generate masks used in self-attention:
 * <ul>
 *   <li>Causal mask - prevents attending to future tokens (GPT-2, decoder)</li>
 *   <li>Padding mask - prevents attending to padding tokens (BERT, encoder)</li>
 *   <li>Combined mask - both causal and padding for decoder with padding</li>
 * </ul>
 *
 * <p>Masks use the convention:
 * <ul>
 *   <li>0.0 = attend (keep)</li>
 *   <li>-inf = don't attend (mask out)</li>
 * </ul>
 *
 * <p>This convention allows masks to be added directly to attention scores
 * before softmax: softmax(QK^T / sqrt(d) + mask).
 *
 * <p>Example:
 * <pre>{@code
 * // Causal mask for autoregressive decoding
 * TypedTensor<Matrix, F32, Cpu> causal = MaskOps.causalMask(seqLen, Cpu.INSTANCE);
 * // Result: [seqLen, seqLen] with 0 in lower triangle, -inf elsewhere
 *
 * // Padding mask for variable-length sequences
 * int[] lengths = {5, 8, 3};  // actual lengths per batch
 * TypedTensor<Matrix, F32, Cpu> padding = MaskOps.paddingMask(lengths, maxLen, Cpu.INSTANCE);
 * // Result: [batch, maxLen] with 0 for valid positions, -inf for padding
 * }</pre>
 */
public final class MaskOps {

    private MaskOps() {
        // Utility class
    }

    /** Large negative value used for masking (not -Infinity to avoid NaN in softmax) */
    private static final float MASK_VALUE = -1e9f;

    // ==================== Causal Mask ====================

    /**
     * Creates a causal (autoregressive) attention mask.
     *
     * <p>Returns a lower triangular matrix where:
     * - mask[i][j] = 0 if j <= i (can attend)
     * - mask[i][j] = -inf if j > i (cannot attend to future)
     *
     * <p>Used in GPT-2, LLaMA, and other autoregressive models.
     *
     * @param seqLen sequence length
     * @param deviceType device to create mask on
     * @param <V> device type
     * @return causal mask [seqLen, seqLen]
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> causalMask(int seqLen, V deviceType) {
        Tensor result = Tensor.zeros(ScalarType.F32, seqLen, seqLen);
        causalMaskF32(result.data(), seqLen);
        return TypedTensor.from(result, new Matrix(seqLen, seqLen), F32.INSTANCE, deviceType);
    }

    /**
     * Creates a batched causal mask for multi-head attention.
     *
     * <p>Returns [1, 1, seqLen, seqLen] which can be broadcast across
     * batch and head dimensions in attention computation.
     *
     * @param seqLen sequence length
     * @param deviceType device to create mask on
     * @param <V> device type
     * @return causal mask [1, 1, seqLen, seqLen]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank4, F32, V> causalMaskRank4(int seqLen, V deviceType) {
        Tensor result = Tensor.zeros(ScalarType.F32, 1, 1, seqLen, seqLen);
        causalMaskF32(result.data(), seqLen);
        return TypedTensor.from(result, new Rank4(1, 1, seqLen, seqLen), F32.INSTANCE, deviceType);
    }

    // ==================== Padding Mask ====================

    /**
     * Creates a padding mask from sequence lengths.
     *
     * <p>For each sequence in the batch, positions >= length are masked out.
     *
     * @param lengths actual sequence lengths for each batch element
     * @param maxLen maximum sequence length (padding length)
     * @param deviceType device to create mask on
     * @param <V> device type
     * @return padding mask [batch, maxLen]
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> paddingMask(int[] lengths, int maxLen, V deviceType) {
        int batch = lengths.length;
        Tensor result = Tensor.zeros(ScalarType.F32, batch, maxLen);
        paddingMaskF32(result.data(), lengths, batch, maxLen);
        return TypedTensor.from(result, new Matrix(batch, maxLen), F32.INSTANCE, deviceType);
    }

    /**
     * Creates a 4D padding mask for attention.
     *
     * <p>Returns [batch, 1, 1, seqLen] which broadcasts to [batch, heads, q_len, k_len]
     * in attention computation.
     *
     * @param lengths actual sequence lengths for each batch element
     * @param maxLen maximum sequence length
     * @param deviceType device to create mask on
     * @param <V> device type
     * @return padding mask [batch, 1, 1, maxLen]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank4, F32, V> paddingMaskRank4(int[] lengths, int maxLen, V deviceType) {
        int batch = lengths.length;
        Tensor result = Tensor.zeros(ScalarType.F32, batch, 1, 1, maxLen);
        paddingMask4DF32(result.data(), lengths, batch, maxLen);
        return TypedTensor.from(result, new Rank4(batch, 1, 1, maxLen), F32.INSTANCE, deviceType);
    }

    /**
     * Creates a padding mask from a boolean attention mask.
     *
     * <p>Input is [batch, seqLen] with 1.0 for valid and 0.0 for padding.
     * Output uses mask convention (0.0 for valid, -inf for padding).
     *
     * @param attentionMask boolean-like mask [batch, seqLen] (1=valid, 0=padding)
     * @param <V> device type
     * @return padding mask [batch, seqLen]
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> fromAttentionMask(TypedTensor<Matrix, F32, V> attentionMask) {
        int[] dims = attentionMask.dimensions();
        int batch = dims[0];
        int seqLen = dims[1];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        fromAttentionMaskF32(attentionMask.underlying().data(), result.data(), batch, seqLen);

        return TypedTensor.from(result, new Matrix(batch, seqLen), F32.INSTANCE, attentionMask.deviceType());
    }

    // ==================== Combined Masks ====================

    /**
     * Creates a combined causal + padding mask.
     *
     * <p>Combines causal masking (can't attend to future) with padding masking
     * (can't attend to padding tokens). Used in decoder-only models with
     * variable-length input.
     *
     * @param lengths actual sequence lengths [batch]
     * @param maxLen maximum sequence length
     * @param deviceType device to create mask on
     * @param <V> device type
     * @return combined mask [batch, 1, maxLen, maxLen]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank4, F32, V> causalPaddingMask(int[] lengths, int maxLen, V deviceType) {
        int batch = lengths.length;
        Tensor result = Tensor.zeros(ScalarType.F32, batch, 1, maxLen, maxLen);
        causalPaddingMaskF32(result.data(), lengths, batch, maxLen);
        return TypedTensor.from(result, new Rank4(batch, 1, maxLen, maxLen), F32.INSTANCE, deviceType);
    }

    /**
     * Adds a causal mask to an existing attention mask.
     *
     * <p>Takes an attention mask [batch, 1, q_len, k_len] and adds causal
     * constraints (can't attend to future positions).
     *
     * @param mask existing attention mask
     * @param <V> device type
     * @return mask with causal constraints added
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank4, F32, V> addCausalMask(TypedTensor<Rank4, F32, V> mask) {
        int[] dims = mask.dimensions();
        int batch = dims[0];
        int heads = dims[1];
        int qLen = dims[2];
        int kLen = dims[3];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        addCausalMaskF32(mask.underlying().data(), result.data(), batch, heads, qLen, kLen);

        return TypedTensor.from(result, new Rank4(batch, heads, qLen, kLen), F32.INSTANCE, mask.deviceType());
    }

    // ==================== Mask Application ====================

    /**
     * Applies an attention mask to attention scores.
     *
     * <p>scores_masked = scores + mask
     *
     * <p>Since mask uses 0 for valid and -inf for masked, this effectively
     * sets masked positions to -inf before softmax.
     *
     * @param scores attention scores [batch, heads, q_len, k_len]
     * @param mask attention mask [batch, 1, 1, k_len] or [batch, 1, q_len, k_len]
     * @param <V> device type
     * @return masked scores
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank4, F32, V> applyMask(
            TypedTensor<Rank4, F32, V> scores,
            TypedTensor<Rank4, F32, V> mask) {

        int[] scoreDims = scores.dimensions();
        int[] maskDims = mask.dimensions();

        // Validate broadcastable
        if (!isBroadcastable(scoreDims, maskDims)) {
            throw new IllegalArgumentException(
                    "Mask shape " + java.util.Arrays.toString(maskDims) +
                    " not broadcastable to scores shape " + java.util.Arrays.toString(scoreDims));
        }

        int batch = scoreDims[0];
        int heads = scoreDims[1];
        int qLen = scoreDims[2];
        int kLen = scoreDims[3];

        Tensor result = Tensor.zeros(ScalarType.F32, scoreDims);
        applyMaskF32(
                scores.underlying().data(),
                mask.underlying().data(),
                result.data(),
                batch, heads, qLen, kLen,
                maskDims[0], maskDims[1], maskDims[2], maskDims[3]);

        return TypedTensor.from(result, new Rank4(batch, heads, qLen, kLen), F32.INSTANCE, scores.deviceType());
    }

    // ==================== Internal Implementation ====================

    private static boolean isBroadcastable(int[] target, int[] source) {
        if (target.length != source.length) return false;
        for (int i = 0; i < target.length; i++) {
            if (source[i] != 1 && source[i] != target[i]) return false;
        }
        return true;
    }

    private static void causalMaskF32(MemorySegment dst, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            long rowOffset = (long) i * seqLen;
            for (int j = 0; j < seqLen; j++) {
                // 0 if j <= i (can attend), MASK_VALUE if j > i (future)
                float val = j <= i ? 0.0f : MASK_VALUE;
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, rowOffset + j, val);
            }
        }
    }

    private static void paddingMaskF32(MemorySegment dst, int[] lengths, int batch, int maxLen) {
        for (int b = 0; b < batch; b++) {
            int len = lengths[b];
            long rowOffset = (long) b * maxLen;
            for (int j = 0; j < maxLen; j++) {
                // 0 if j < len (valid), MASK_VALUE if j >= len (padding)
                float val = j < len ? 0.0f : MASK_VALUE;
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, rowOffset + j, val);
            }
        }
    }

    private static void paddingMask4DF32(MemorySegment dst, int[] lengths, int batch, int maxLen) {
        // Shape: [batch, 1, 1, maxLen]
        for (int b = 0; b < batch; b++) {
            int len = lengths[b];
            long offset = (long) b * maxLen;  // 1*1*maxLen = maxLen
            for (int j = 0; j < maxLen; j++) {
                float val = j < len ? 0.0f : MASK_VALUE;
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + j, val);
            }
        }
    }

    private static void fromAttentionMaskF32(MemorySegment src, MemorySegment dst, int batch, int seqLen) {
        for (int b = 0; b < batch; b++) {
            long offset = (long) b * seqLen;
            for (int j = 0; j < seqLen; j++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + j);
                // If input is 1.0 (valid), output 0.0
                // If input is 0.0 (padding), output MASK_VALUE
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + j, val > 0.5f ? 0.0f : MASK_VALUE);
            }
        }
    }

    private static void causalPaddingMaskF32(MemorySegment dst, int[] lengths, int batch, int maxLen) {
        // Shape: [batch, 1, maxLen, maxLen]
        long planeSize = (long) maxLen * maxLen;

        for (int b = 0; b < batch; b++) {
            int len = lengths[b];
            long planeOffset = b * planeSize;

            for (int i = 0; i < maxLen; i++) {
                long rowOffset = planeOffset + (long) i * maxLen;
                for (int j = 0; j < maxLen; j++) {
                    // Combined conditions:
                    // 1. Can't attend to future (j > i)
                    // 2. Can't attend to padding (j >= len)
                    boolean canAttend = (j <= i) && (j < len);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, rowOffset + j, canAttend ? 0.0f : MASK_VALUE);
                }
            }
        }
    }

    private static void addCausalMaskF32(MemorySegment src, MemorySegment dst,
                                         int batch, int heads, int qLen, int kLen) {
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int q = 0; q < qLen; q++) {
                    long offset = (((long) b * heads + h) * qLen + q) * kLen;
                    for (int k = 0; k < kLen; k++) {
                        float srcVal = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + k);
                        // Add causal constraint
                        float causal = k <= q ? 0.0f : MASK_VALUE;
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + k, srcVal + causal);
                    }
                }
            }
        }
    }

    private static void applyMaskF32(MemorySegment scores, MemorySegment mask, MemorySegment dst,
                                     int batch, int heads, int qLen, int kLen,
                                     int maskBatch, int maskHeads, int maskQ, int maskK) {
        for (int b = 0; b < batch; b++) {
            int mb = maskBatch == 1 ? 0 : b;
            for (int h = 0; h < heads; h++) {
                int mh = maskHeads == 1 ? 0 : h;
                for (int q = 0; q < qLen; q++) {
                    int mq = maskQ == 1 ? 0 : q;
                    long scoreOffset = (((long) b * heads + h) * qLen + q) * kLen;
                    long maskOffset = (((long) mb * maskHeads + mh) * maskQ + mq) * maskK;

                    for (int k = 0; k < kLen; k++) {
                        int mk = maskK == 1 ? 0 : k;
                        float scoreVal = scores.getAtIndex(ValueLayout.JAVA_FLOAT, scoreOffset + k);
                        float maskVal = mask.getAtIndex(ValueLayout.JAVA_FLOAT, maskOffset + mk);
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, scoreOffset + k, scoreVal + maskVal);
                    }
                }
            }
        }
    }
}
