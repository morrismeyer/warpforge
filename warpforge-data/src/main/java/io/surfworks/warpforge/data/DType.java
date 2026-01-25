package io.surfworks.warpforge.data;

import java.lang.foreign.ValueLayout;

/**
 * Data types for tensors, matching PyTorch, SafeTensors, and P3109 conventions.
 *
 * <p>Includes support for P3109 small floating-point formats used in ML quantization:
 * <ul>
 *   <li>FP8: E5M2, E4M3, E4M3FN, E8M0</li>
 *   <li>FP6: E3M2, E2M3</li>
 *   <li>FP4: E2M1, E1M2</li>
 * </ul>
 */
public enum DType {

    // Standard IEEE 754 floating point
    F32("F32", 4, true, 32),
    F64("F64", 8, true, 64),
    F16("F16", 2, true, 16),
    BF16("BF16", 2, true, 16),

    // P3109 FP8 types
    F8_E5M2("F8_E5M2", 1, true, 8),    // Binary8p3se: wide dynamic range
    F8_E4M3("F8_E4M3", 1, true, 8),    // Binary8p4se: higher precision
    F8_E4M3FN("F8_E4M3FN", 1, true, 8), // Binary8p4sf: NVIDIA finite-only variant
    F8_E8M0("F8_E8M0", 1, true, 8),    // Binary8p1uf: exponent-only scale factor

    // P3109 FP6 types (packed: 4 values in 3 bytes)
    F6_E3M2("F6_E3M2", 1, true, 6),    // Binary6p3sf
    F6_E2M3("F6_E2M3", 1, true, 6),    // Binary6p4sf

    // P3109 FP4 types (packed: 2 values per byte)
    F4_E2M1("F4_E2M1", 1, true, 4),    // Binary4p2sf
    F4_E1M2("F4_E1M2", 1, true, 4),    // Binary4p3sf

    // Integer types
    I8("I8", 1, false, 8),
    I16("I16", 2, false, 16),
    I32("I32", 4, false, 32),
    I64("I64", 8, false, 64),
    U8("U8", 1, false, 8),
    U16("U16", 2, false, 16),
    U32("U32", 4, false, 32),
    U64("U64", 8, false, 64),

    // Boolean
    BOOL("BOOL", 1, false, 1),

    // Quantized types (GGUF block-based)
    Q4_0("Q4_0", -1, false, 4),
    Q4_1("Q4_1", -1, false, 4),
    Q4_K_M("Q4_K_M", -1, false, 4),
    Q5_0("Q5_0", -1, false, 5),
    Q5_1("Q5_1", -1, false, 5),
    Q5_K_M("Q5_K_M", -1, false, 5),
    Q8_0("Q8_0", -1, false, 8),
    Q8_K("Q8_K", -1, false, 8);

    private final String safetensorsName;
    private final int byteSize;
    private final boolean floating;
    private final int bitWidth;

    DType(String safetensorsName, int byteSize, boolean floating, int bitWidth) {
        this.safetensorsName = safetensorsName;
        this.byteSize = byteSize;
        this.floating = floating;
        this.bitWidth = bitWidth;
    }

    /**
     * Bytes per element, or -1 for block-quantized types (variable block size).
     */
    public int byteSize() {
        return byteSize;
    }

    /**
     * Bit width for this type.
     */
    public int bitWidth() {
        return bitWidth;
    }

    public boolean isFloating() {
        return floating;
    }

    /**
     * Check if this is a block-quantized type (GGUF Q4, Q5, Q8, etc.).
     */
    public boolean isBlockQuantized() {
        return byteSize < 0;
    }

    /**
     * Check if this is a P3109 small floating-point type.
     */
    public boolean isP3109() {
        return switch (this) {
            case F8_E5M2, F8_E4M3, F8_E4M3FN, F8_E8M0,
                 F6_E3M2, F6_E2M3,
                 F4_E2M1, F4_E1M2 -> true;
            default -> false;
        };
    }

    /**
     * Check if this is a sub-byte type (FP4 or FP6).
     */
    public boolean isSubByte() {
        return bitWidth < 8 && bitWidth > 0;
    }

    /**
     * @deprecated Use {@link #isBlockQuantized()} instead.
     */
    @Deprecated
    public boolean isQuantized() {
        return byteSize < 0;
    }

    /**
     * Calculate packed byte size for count elements.
     * Handles sub-byte formats (FP4, FP6) correctly.
     */
    public long packedByteSize(long count) {
        // Check sub-byte first (FP4, FP6) - these have byteSize=1 for element access
        // but pack multiple values per byte
        if (isSubByte()) {
            // Sub-byte packing: ceiling division
            long totalBits = count * bitWidth;
            return (totalBits + 7) / 8;
        }
        if (byteSize > 0) {
            return count * byteSize;
        }
        throw new UnsupportedOperationException("Cannot calculate packed size for: " + this);
    }

    /**
     * Parse from SafeTensors dtype string.
     * Also supports P3109 FP8/FP6/FP4 format names.
     */
    public static DType fromSafeTensors(String dtype) {
        return switch (dtype.toUpperCase()) {
            case "F32", "FLOAT32" -> F32;
            case "F64", "FLOAT64" -> F64;
            case "F16", "FLOAT16" -> F16;
            case "BF16", "BFLOAT16" -> BF16;
            // P3109 FP8 types
            case "F8_E5M2", "FLOAT8_E5M2", "E5M2" -> F8_E5M2;
            case "F8_E4M3", "FLOAT8_E4M3", "E4M3" -> F8_E4M3;
            case "F8_E4M3FN", "FLOAT8_E4M3FN", "E4M3FN" -> F8_E4M3FN;
            case "F8_E8M0", "FLOAT8_E8M0", "E8M0" -> F8_E8M0;
            // P3109 FP6 types
            case "F6_E3M2", "FLOAT6_E3M2", "E3M2" -> F6_E3M2;
            case "F6_E2M3", "FLOAT6_E2M3", "E2M3" -> F6_E2M3;
            // P3109 FP4 types
            case "F4_E2M1", "FLOAT4_E2M1", "E2M1" -> F4_E2M1;
            case "F4_E1M2", "FLOAT4_E1M2", "E1M2" -> F4_E1M2;
            // Integer types
            case "I8", "INT8" -> I8;
            case "I16", "INT16" -> I16;
            case "I32", "INT32" -> I32;
            case "I64", "INT64" -> I64;
            case "U8", "UINT8" -> U8;
            case "U16", "UINT16" -> U16;
            case "U32", "UINT32" -> U32;
            case "U64", "UINT64" -> U64;
            case "BOOL" -> BOOL;
            default -> throw new IllegalArgumentException("Unknown dtype: " + dtype);
        };
    }

    /**
     * Get the ValueLayout for FFM MemorySegment access.
     */
    public ValueLayout valueLayout() {
        return switch (this) {
            case F32 -> ValueLayout.JAVA_FLOAT;
            case F64 -> ValueLayout.JAVA_DOUBLE;
            case F16, BF16 -> ValueLayout.JAVA_SHORT;
            case I8, U8, BOOL -> ValueLayout.JAVA_BYTE;
            case I16, U16 -> ValueLayout.JAVA_SHORT;
            case I32, U32 -> ValueLayout.JAVA_INT;
            case I64, U64 -> ValueLayout.JAVA_LONG;
            // P3109 FP8 types use JAVA_BYTE
            case F8_E5M2, F8_E4M3, F8_E4M3FN, F8_E8M0 -> ValueLayout.JAVA_BYTE;
            // P3109 FP6/FP4 also use JAVA_BYTE (packed access)
            case F6_E3M2, F6_E2M3, F4_E2M1, F4_E1M2 -> ValueLayout.JAVA_BYTE;
            default -> throw new UnsupportedOperationException("No value layout for type: " + this);
        };
    }

    /**
     * Convert BF16 bits to float.
     */
    public static float bf16ToFloat(short bits) {
        return Float.intBitsToFloat(((int) bits) << 16);
    }

    /**
     * Convert float to BF16 bits.
     */
    public static short floatToBf16(float f) {
        return (short) (Float.floatToRawIntBits(f) >>> 16);
    }

    /**
     * Convert F16 bits to float using IEEE 754 half-precision format.
     */
    public static float f16ToFloat(short bits) {
        int sign = (bits >>> 15) & 0x1;
        int exp = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x3FF;

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Denormalized
            float val = (float) (mantissa / 1024.0 * Math.pow(2, -14));
            return sign == 0 ? val : -val;
        } else if (exp == 31) {
            if (mantissa == 0) {
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        float val = (float) ((1.0 + mantissa / 1024.0) * Math.pow(2, exp - 15));
        return sign == 0 ? val : -val;
    }

    // =========================================================================
    // P3109 FP8/FP6/FP4 Conversion Utilities
    // =========================================================================

    /**
     * Convert FP8 E5M2 bits to float.
     * E5M2: 1 sign + 5 exponent + 2 mantissa, bias = 15
     */
    public static float f8e5m2ToFloat(byte bits) {
        int b = bits & 0xFF;
        int sign = (b >>> 7) & 0x1;
        int exp = (b >>> 2) & 0x1F;
        int mantissa = b & 0x3;

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal
            float val = (float) (mantissa / 4.0 * Math.pow(2, -14));
            return sign == 0 ? val : -val;
        } else if (exp == 31) {
            if (mantissa == 0) {
                return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        float val = (float) ((1.0 + mantissa / 4.0) * Math.pow(2, exp - 15));
        return sign == 0 ? val : -val;
    }

    /**
     * Convert FP8 E4M3 bits to float.
     * E4M3: 1 sign + 4 exponent + 3 mantissa, bias = 7
     */
    public static float f8e4m3ToFloat(byte bits) {
        int b = bits & 0xFF;
        int sign = (b >>> 7) & 0x1;
        int exp = (b >>> 3) & 0xF;
        int mantissa = b & 0x7;

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal
            float val = (float) (mantissa / 8.0 * Math.pow(2, -6));
            return sign == 0 ? val : -val;
        } else if (exp == 15) {
            if (mantissa == 7) {
                // In E4M3, only 0x7F and 0xFF are NaN (sign bit varies)
                return Float.NaN;
            }
            // P3109: exp=15 with mantissa<7 is a valid finite number
            float val = (float) ((1.0 + mantissa / 8.0) * Math.pow(2, 8)); // exp - bias = 15 - 7 = 8
            return sign == 0 ? val : -val;
        }

        float val = (float) ((1.0 + mantissa / 8.0) * Math.pow(2, exp - 7));
        return sign == 0 ? val : -val;
    }

    /**
     * Convert FP8 E4M3FN (finite-only, NVIDIA variant) bits to float.
     * Same as E4M3 but NaN is replaced with max finite value.
     */
    public static float f8e4m3fnToFloat(byte bits) {
        int b = bits & 0xFF;
        int sign = (b >>> 7) & 0x1;
        int exp = (b >>> 3) & 0xF;
        int mantissa = b & 0x7;

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal
            float val = (float) (mantissa / 8.0 * Math.pow(2, -6));
            return sign == 0 ? val : -val;
        }

        // In E4M3FN, all values including exp=15, mantissa=7 are finite
        float val = (float) ((1.0 + mantissa / 8.0) * Math.pow(2, exp - 7));
        return sign == 0 ? val : -val;
    }

    /**
     * Convert FP4 E2M1 bits to float.
     * E2M1: 1 sign + 2 exponent + 1 mantissa, bias = 1, finite-only
     */
    public static float f4e2m1ToFloat(int nibble) {
        int b = nibble & 0xF;
        int sign = (b >>> 3) & 0x1;
        int exp = (b >>> 1) & 0x3;
        int mantissa = b & 0x1;

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal: 2^(1-bias) * (0.mantissa) = 2^0 * 0.5 = 0.5
            float val = 0.5f;
            return sign == 0 ? val : -val;
        }

        // Normal: 2^(exp-bias) * 1.mantissa
        float val = (float) ((1.0 + mantissa / 2.0) * Math.pow(2, exp - 1));
        return sign == 0 ? val : -val;
    }

    /**
     * Convert FP4 E1M2 bits to float.
     * E1M2: 1 sign + 1 exponent + 2 mantissa, bias = 0, finite-only
     */
    public static float f4e1m2ToFloat(int nibble) {
        int b = nibble & 0xF;
        int sign = (b >>> 3) & 0x1;
        int exp = (b >>> 2) & 0x1;
        int mantissa = b & 0x3;

        if (exp == 0) {
            if (mantissa == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Subnormal
            float val = mantissa / 4.0f;
            return sign == 0 ? val : -val;
        }

        // Normal: 2^(exp-bias) * 1.mantissa = 2^1 * (1 + m/4)
        float val = 2.0f * (1.0f + mantissa / 4.0f);
        return sign == 0 ? val : -val;
    }

    /**
     * Convert float to FP8 E5M2 bits.
     */
    public static byte floatToF8e5m2(float f) {
        if (Float.isNaN(f)) {
            return (byte) 0x7F; // NaN
        }
        if (Float.isInfinite(f)) {
            return f > 0 ? (byte) 0x7C : (byte) 0xFC; // +/-Inf
        }

        int bits = Float.floatToRawIntBits(f);
        int sign = (bits >>> 31) & 0x1;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15; // Rebias from F32 to E5M2
        int mantissa = (bits >>> 21) & 0x3; // Take top 2 mantissa bits

        if (exp <= 0) {
            return (byte) (sign << 7); // Underflow to zero
        }
        if (exp >= 31) {
            return (byte) ((sign << 7) | 0x7C); // Overflow to infinity
        }

        return (byte) ((sign << 7) | (exp << 2) | mantissa);
    }

    /**
     * Convert float to FP8 E4M3 bits.
     */
    public static byte floatToF8e4m3(float f) {
        if (Float.isNaN(f)) {
            return (byte) 0x7F; // NaN (positive)
        }

        int bits = Float.floatToRawIntBits(f);
        int sign = (bits >>> 31) & 0x1;

        if (Float.isInfinite(f)) {
            // E4M3 maps infinity to max finite value
            return (byte) ((sign << 7) | 0x7E); // exp=15, mantissa=6
        }

        int exp = ((bits >>> 23) & 0xFF) - 127 + 7; // Rebias from F32 to E4M3
        int mantissa = (bits >>> 20) & 0x7; // Take top 3 mantissa bits

        if (exp <= 0) {
            return (byte) (sign << 7); // Underflow to zero
        }
        if (exp >= 15) {
            // Clamp to max finite (exp=15, mantissa=6, since 7 is NaN)
            return (byte) ((sign << 7) | 0x7E);
        }

        return (byte) ((sign << 7) | (exp << 3) | mantissa);
    }

    /**
     * Convert float to FP4 E2M1 bits (returns 4-bit value in low nibble).
     * E2M1 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
     * Uses midpoint rounding: e.g., values in [0.75, 1.25) map to 1.0
     */
    public static int floatToF4e2m1(float f) {
        if (Float.isNaN(f) || Float.isInfinite(f)) {
            // Clamp to max finite
            return f >= 0 ? 0x7 : 0xF; // +/-6.0
        }

        int sign = f < 0 ? 1 : 0;
        float absVal = Math.abs(f);

        // Midpoints between E2M1 values: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
        if (absVal < 0.25f) {
            return sign << 3; // Zero (0x0)
        }
        if (absVal < 0.75f) {
            return (sign << 3) | 0x1; // 0.5 (subnormal)
        }
        if (absVal < 1.25f) {
            return (sign << 3) | 0x2; // 1.0
        }
        if (absVal < 1.75f) {
            return (sign << 3) | 0x3; // 1.5
        }
        if (absVal < 2.5f) {
            return (sign << 3) | 0x4; // 2.0
        }
        if (absVal < 3.5f) {
            return (sign << 3) | 0x5; // 3.0
        }
        if (absVal < 5.0f) {
            return (sign << 3) | 0x6; // 4.0
        }
        // Max value: 6.0 (0x7)
        return (sign << 3) | 0x7;
    }
}
