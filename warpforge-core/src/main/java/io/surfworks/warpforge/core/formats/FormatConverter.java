package io.surfworks.warpforge.core.formats;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * High-performance format conversion between tensor types.
 *
 * <p>This class provides bulk conversion operations between different
 * floating-point formats, optimized for tensor data exchange with GPUs.
 *
 * <h2>Conversion Paths</h2>
 * <ul>
 *   <li>FP32/FP64 ↔ FP8 (E5M2, E4M3, E4M3FN)</li>
 *   <li>FP32/FP64 ↔ FP6 (E3M2, E2M3)</li>
 *   <li>FP32/FP64 ↔ FP4 (E2M1, E1M2)</li>
 *   <li>FP8/FP6/FP4 ↔ FP8/FP6/FP4 (cross-format)</li>
 *   <li>Block formats (NVFP4, MX) ↔ FP32</li>
 * </ul>
 *
 * <h2>Performance</h2>
 * All operations use direct MemorySegment access for zero-copy semantics
 * where possible. Bulk operations are vectorized where beneficial.
 */
public final class FormatConverter {

    private FormatConverter() {
        // Static utility class
    }

    // ==================== Tensor-level Conversion ====================

    /**
     * Convert a tensor to a different scalar type.
     *
     * @param source Source tensor
     * @param targetType Target scalar type
     * @return New tensor with converted data
     */
    public static Tensor convert(Tensor source, ScalarType targetType) {
        if (source.dtype() == targetType) {
            return source.copy();
        }

        int[] shape = source.shape();
        TensorSpec targetSpec = TensorSpec.of(targetType, shape);
        Arena arena = Arena.ofConfined();

        try {
            MemorySegment destSegment = arena.allocate(targetSpec.byteSize());
            convert(source.data(), source.dtype(), destSegment, targetType, (int) source.elementCount());
            return Tensor.fromMemorySegment(destSegment, targetSpec, arena);
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Convert between MemorySegments with different formats.
     *
     * @param source Source segment
     * @param sourceType Source scalar type
     * @param dest Destination segment
     * @param destType Destination scalar type
     * @param count Number of elements
     */
    public static void convert(MemorySegment source, ScalarType sourceType,
                               MemorySegment dest, ScalarType destType, int count) {
        // Fast path: same type
        if (sourceType == destType) {
            MemorySegment.copy(source, 0, dest, 0, sourceType.packedByteSize(count));
            return;
        }

        // Convert through float intermediate
        // This is the portable path; optimized paths can be added for specific conversions
        float[] intermediate = new float[count];

        // Step 1: Decode source to float
        decodeToFloat(source, sourceType, intermediate);

        // Step 2: Encode float to destination
        encodeFromFloat(intermediate, dest, destType);
    }

    // ==================== Decode to Float ====================

    /**
     * Decode data from any supported format to float array.
     *
     * @param source Source segment
     * @param sourceType Source scalar type
     * @param dest Destination float array
     */
    public static void decodeToFloat(MemorySegment source, ScalarType sourceType, float[] dest) {
        int count = dest.length;

        switch (sourceType) {
            case F32 -> {
                MemorySegment.copy(source, ValueLayout.JAVA_FLOAT, 0, dest, 0, count);
            }
            case F64 -> {
                for (int i = 0; i < count; i++) {
                    dest[i] = (float) source.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                }
            }
            case F16 -> {
                for (int i = 0; i < count; i++) {
                    short bits = source.getAtIndex(ValueLayout.JAVA_SHORT, i);
                    dest[i] = Float.float16ToFloat(bits);
                }
            }
            case BF16 -> {
                for (int i = 0; i < count; i++) {
                    short bits = source.getAtIndex(ValueLayout.JAVA_SHORT, i);
                    // BF16: upper 16 bits of FP32
                    dest[i] = Float.intBitsToFloat(bits << 16);
                }
            }
            case F8_E5M2 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP8_E5M2);
            case F8_E4M3 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP8_E4M3);
            case F8_E4M3FN -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP8_E4M3FN);
            case F8_E8M0 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP8_E8M0);
            case F6_E3M2 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP6_E3M2);
            case F6_E2M3 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP6_E2M3);
            case F4_E2M1 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP4_E2M1);
            case F4_E1M2 -> MiniFloat.decodeBulk(source, dest, FormatParameters.FP4_E1M2);
            case I8 -> {
                for (int i = 0; i < count; i++) {
                    dest[i] = source.get(ValueLayout.JAVA_BYTE, i);
                }
            }
            case I16 -> {
                for (int i = 0; i < count; i++) {
                    dest[i] = source.getAtIndex(ValueLayout.JAVA_SHORT, i);
                }
            }
            case I32 -> {
                for (int i = 0; i < count; i++) {
                    dest[i] = source.getAtIndex(ValueLayout.JAVA_INT, i);
                }
            }
            case I64 -> {
                for (int i = 0; i < count; i++) {
                    dest[i] = source.getAtIndex(ValueLayout.JAVA_LONG, i);
                }
            }
            case I1, BOOL -> {
                for (int i = 0; i < count; i++) {
                    dest[i] = source.get(ValueLayout.JAVA_BYTE, i) != 0 ? 1.0f : 0.0f;
                }
            }
            default -> throw new IllegalArgumentException("Unsupported source type: " + sourceType);
        }
    }

    // ==================== Encode from Float ====================

    /**
     * Encode float array to any supported format.
     *
     * @param source Source float array
     * @param dest Destination segment
     * @param destType Destination scalar type
     */
    public static void encodeFromFloat(float[] source, MemorySegment dest, ScalarType destType) {
        int count = source.length;

        switch (destType) {
            case F32 -> {
                MemorySegment.copy(source, 0, dest, ValueLayout.JAVA_FLOAT, 0, count);
            }
            case F64 -> {
                for (int i = 0; i < count; i++) {
                    dest.setAtIndex(ValueLayout.JAVA_DOUBLE, i, source[i]);
                }
            }
            case F16 -> {
                for (int i = 0; i < count; i++) {
                    short bits = Float.floatToFloat16(source[i]);
                    dest.setAtIndex(ValueLayout.JAVA_SHORT, i, bits);
                }
            }
            case BF16 -> {
                for (int i = 0; i < count; i++) {
                    // BF16: truncate FP32 to upper 16 bits (with rounding)
                    int fp32Bits = Float.floatToRawIntBits(source[i]);
                    // Round to nearest even
                    int roundBit = (fp32Bits >> 15) & 1;
                    int stickyBits = fp32Bits & 0x7FFF;
                    if (stickyBits > 0x8000 || (stickyBits == 0x8000 && roundBit == 1)) {
                        fp32Bits += 0x10000;
                    }
                    short bf16Bits = (short) (fp32Bits >> 16);
                    dest.setAtIndex(ValueLayout.JAVA_SHORT, i, bf16Bits);
                }
            }
            case F8_E5M2 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP8_E5M2);
            case F8_E4M3 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP8_E4M3);
            case F8_E4M3FN -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP8_E4M3FN);
            case F8_E8M0 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP8_E8M0);
            case F6_E3M2 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP6_E3M2);
            case F6_E2M3 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP6_E2M3);
            case F4_E2M1 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP4_E2M1);
            case F4_E1M2 -> MiniFloat.encodeBulk(source, dest, FormatParameters.FP4_E1M2);
            case I8 -> {
                for (int i = 0; i < count; i++) {
                    dest.set(ValueLayout.JAVA_BYTE, i, (byte) Math.round(source[i]));
                }
            }
            case I16 -> {
                for (int i = 0; i < count; i++) {
                    dest.setAtIndex(ValueLayout.JAVA_SHORT, i, (short) Math.round(source[i]));
                }
            }
            case I32 -> {
                for (int i = 0; i < count; i++) {
                    dest.setAtIndex(ValueLayout.JAVA_INT, i, Math.round(source[i]));
                }
            }
            case I64 -> {
                for (int i = 0; i < count; i++) {
                    dest.setAtIndex(ValueLayout.JAVA_LONG, i, (long) source[i]);
                }
            }
            case I1, BOOL -> {
                for (int i = 0; i < count; i++) {
                    dest.set(ValueLayout.JAVA_BYTE, i, (byte) (source[i] != 0 ? 1 : 0));
                }
            }
            default -> throw new IllegalArgumentException("Unsupported destination type: " + destType);
        }
    }

    // ==================== Block Format Conversion ====================

    /**
     * Convert float array to NVFP4 block format.
     *
     * @param source Source float values
     * @param dest Destination segment (must be at least NvFp4Block.INSTANCE.byteSize(source.length))
     */
    public static void toNvFp4(float[] source, MemorySegment dest) {
        NvFp4Block.INSTANCE.encode(source, dest);
    }

    /**
     * Convert NVFP4 block format to float array.
     *
     * @param source Source segment in NVFP4 format
     * @param dest Destination float array
     */
    public static void fromNvFp4(MemorySegment source, float[] dest) {
        NvFp4Block.INSTANCE.decode(source, dest);
    }

    /**
     * Convert float array to OCP MX block format.
     *
     * @param source Source float values
     * @param dest Destination segment
     * @param format The MX format variant (MXFP4, MXFP6_E3M2, etc.)
     */
    public static void toMx(float[] source, MemorySegment dest, MxBlock format) {
        format.encode(source, dest);
    }

    /**
     * Convert OCP MX block format to float array.
     *
     * @param source Source segment in MX format
     * @param dest Destination float array
     * @param format The MX format variant
     */
    public static void fromMx(MemorySegment source, float[] dest, MxBlock format) {
        format.decode(source, dest);
    }

    // ==================== Quantization Statistics ====================

    /**
     * Calculate quantization error statistics.
     *
     * @param original Original float values
     * @param quantized Quantized and dequantized float values
     * @return Statistics record
     */
    public static QuantizationStats calculateStats(float[] original, float[] quantized) {
        if (original.length != quantized.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }

        double sumSquaredError = 0;
        double sumAbsError = 0;
        double maxAbsError = 0;
        double sumOriginalSquared = 0;

        for (int i = 0; i < original.length; i++) {
            float error = original[i] - quantized[i];
            double absError = Math.abs(error);
            sumSquaredError += error * error;
            sumAbsError += absError;
            maxAbsError = Math.max(maxAbsError, absError);
            sumOriginalSquared += original[i] * original[i];
        }

        double mse = sumSquaredError / original.length;
        double mae = sumAbsError / original.length;
        double rmse = Math.sqrt(mse);
        double snr = sumOriginalSquared > 0 ? 10 * Math.log10(sumOriginalSquared / sumSquaredError) : Double.POSITIVE_INFINITY;

        return new QuantizationStats(mse, mae, rmse, maxAbsError, snr);
    }

    /**
     * Statistics for quantization error analysis.
     *
     * @param mse Mean Squared Error
     * @param mae Mean Absolute Error
     * @param rmse Root Mean Squared Error
     * @param maxAbsError Maximum absolute error
     * @param snrDb Signal-to-Noise Ratio in dB
     */
    public record QuantizationStats(
            double mse,
            double mae,
            double rmse,
            double maxAbsError,
            double snrDb
    ) {
        @Override
        public String toString() {
            return String.format("QuantizationStats[MSE=%.6e, MAE=%.6e, RMSE=%.6e, MaxErr=%.6e, SNR=%.2fdB]",
                    mse, mae, rmse, maxAbsError, snrDb);
        }
    }

    // ==================== Convenience Methods ====================

    /**
     * Round-trip conversion to measure quantization error.
     *
     * @param original Original float values
     * @param format Target format
     * @return Quantization statistics
     */
    public static QuantizationStats measureQuantizationError(float[] original, FormatParameters format) {
        // Encode
        long byteSize = MiniFloat.byteSize(original.length, format);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment encoded = arena.allocate(byteSize);
            MiniFloat.encodeBulk(original, encoded, format);

            // Decode
            float[] decoded = new float[original.length];
            MiniFloat.decodeBulk(encoded, decoded, format);

            return calculateStats(original, decoded);
        }
    }

    /**
     * Round-trip conversion through block format to measure quantization error.
     *
     * @param original Original float values
     * @param format Block format
     * @return Quantization statistics
     */
    public static QuantizationStats measureBlockQuantizationError(float[] original, BlockFormat format) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment encoded = arena.allocate(format.byteSize(original.length));
            format.encode(original, encoded);

            float[] decoded = new float[original.length];
            format.decode(encoded, decoded);

            return calculateStats(original, decoded);
        }
    }
}
