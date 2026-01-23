package io.surfworks.warpforge.core.formats;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Core implementation of P3109 small binary floating-point operations.
 *
 * <p>This class provides encoding, decoding, and arithmetic operations for
 * P3109-compliant formats. It is designed for extreme speed and portability,
 * using only primitive operations and avoiding object allocation in hot paths.
 *
 * <h2>P3109 Encoding Differences from IEEE 754</h2>
 * <ul>
 *   <li>Zero is always encoded as all-zero bits (code 0)</li>
 *   <li>No negative zero (its bit pattern encodes NaN for signed formats)</li>
 *   <li>Exactly one NaN (no payload, no signaling/quiet distinction)</li>
 *   <li>For signed extended formats: NaN at bit pattern 2^(K-1)</li>
 *   <li>For signed finite-only formats: no NaN, no infinity</li>
 * </ul>
 *
 * <h2>Performance Notes</h2>
 * <ul>
 *   <li>All encode/decode methods are branchless where possible</li>
 *   <li>Bulk operations use MemorySegment for zero-copy access</li>
 *   <li>No heap allocation in encode/decode paths</li>
 * </ul>
 */
public final class MiniFloat {

    private MiniFloat() {
        // Static utility class
    }

    // ==================== Encoding (float/double to bits) ====================

    /**
     * Encode a float value to P3109 format bits.
     *
     * @param value The float value to encode
     * @param params Format parameters
     * @return The encoded bits (right-aligned in the int)
     */
    public static int encode(float value, FormatParameters params) {
        return encode((double) value, params);
    }

    /**
     * Encode a double value to P3109 format bits.
     *
     * @param value The double value to encode
     * @param params Format parameters
     * @return The encoded bits (right-aligned in the int)
     */
    public static int encode(double value, FormatParameters params) {
        int K = params.bitWidth();
        int E = params.exponentWidth();
        int M = params.mantissaWidth();
        int bias = params.exponentBias();
        boolean signed = params.signed();
        boolean finiteOnly = params.finiteOnly();

        // Handle special values first
        if (Double.isNaN(value)) {
            return encodeNaN(params);
        }

        // Extract sign
        int sign = 0;
        if (value < 0) {
            if (!signed) {
                // Unsigned format: clamp negative to zero
                return 0;
            }
            sign = 1;
            value = -value;
        }

        // Handle zero
        if (value == 0.0) {
            return 0; // P3109: zero is always code 0
        }

        // Handle infinity
        if (Double.isInfinite(value)) {
            if (finiteOnly) {
                // Finite-only: clamp to max value
                return encodeMaxFinite(params, sign);
            }
            return encodeInfinity(params, sign);
        }

        // Handle values outside representable range
        // Values exceeding max should clamp to max finite (not overflow to infinity)
        double maxVal = params.maxValue();
        if (value > maxVal) {
            return encodeMaxFinite(params, sign);
        }

        double minSubnormal = params.minSubnormal();
        if (value < minSubnormal) {
            // Round to nearest: either 0 or minSubnormal
            double roundingThreshold = minSubnormal / 2.0;
            if (value < roundingThreshold) {
                return 0; // Closer to 0
            }
            // Closer to minSubnormal: return smallest positive subnormal
            // For signed formats, the smallest subnormal has sign bit | mantissa=1
            if (signed) {
                return (sign << (K - 1)) | 1;
            } else {
                return 1;
            }
        }

        // Decompose value into exponent and mantissa
        int unbiasedExp = Math.getExponent(value);
        double significand = Math.scalb(value, -unbiasedExp);

        // Adjust for subnormal range
        int minExp = params.minExponent();
        if (unbiasedExp < minExp) {
            // Subnormal: shift mantissa right
            int shift = minExp - unbiasedExp;
            significand = Math.scalb(significand, -shift);
            unbiasedExp = minExp - 1; // Subnormal exponent is stored as 0
        }

        // Calculate biased exponent
        int biasedExp = unbiasedExp + bias;
        if (biasedExp < 0) {
            biasedExp = 0; // Clamp to subnormal
        }

        // Extract mantissa bits (removing implicit leading 1 for normals)
        // significand is in [1, 2) for normals, [0, 1) for subnormals
        double mantissaFrac;
        if (biasedExp == 0) {
            // Subnormal: no implicit 1
            mantissaFrac = significand;
        } else {
            // Normal: remove implicit 1
            mantissaFrac = significand - 1.0;
        }

        // Convert to integer mantissa with rounding
        int mantissaMax = (1 << M) - 1;
        double scaledMantissa = mantissaFrac * (1 << M);
        int mantissa = (int) Math.round(scaledMantissa);

        // Handle rounding overflow
        if (mantissa > mantissaMax) {
            mantissa = 0;
            biasedExp++;
            // Check for exponent overflow
            int maxBiasedExp = params.maxBiasedExponent();
            if (biasedExp > maxBiasedExp) {
                if (finiteOnly) {
                    return encodeMaxFinite(params, sign);
                }
                return encodeInfinity(params, sign);
            }
        }

        // Assemble the bits
        int bits = (sign << (K - 1)) | (biasedExp << M) | mantissa;

        // For P3109 extended formats, check if we accidentally created an infinity code
        // and clamp to max finite instead
        if (!finiteOnly && isInfinityBits(bits, params)) {
            return encodeMaxFinite(params, sign);
        }

        return bits;
    }

    /**
     * Encode NaN according to P3109 rules.
     * For signed formats: NaN is at bit pattern 2^(K-1) (sign=1, exp=0, mant=0)
     * For unsigned formats: NaN is at max code (all 1s) for extended, doesn't exist for finite
     */
    private static int encodeNaN(FormatParameters params) {
        if (params.finiteOnly()) {
            // Finite-only formats don't have NaN; return zero as fallback
            return 0;
        }
        if (params.signed()) {
            // Signed extended: NaN at 2^(K-1)
            return 1 << (params.bitWidth() - 1);
        } else {
            // Unsigned extended: NaN at all-1s
            return (1 << params.bitWidth()) - 1;
        }
    }

    /**
     * Encode positive or negative infinity.
     */
    private static int encodeInfinity(FormatParameters params, int sign) {
        int K = params.bitWidth();
        int M = params.mantissaWidth();
        int maxExp = params.maxBiasedExponent();

        if (params.signed()) {
            if (sign == 0) {
                // +Inf: sign=0, exp=all-1s, mant=all-1s (one below NaN)
                return (maxExp << M) | ((1 << M) - 1);
            } else {
                // -Inf: sign=1, exp=all-1s, mant=all-1s (max code)
                return (1 << (K - 1)) | (maxExp << M) | ((1 << M) - 1);
            }
        } else {
            // Unsigned: only +Inf at second-to-last code
            return (1 << K) - 2;
        }
    }

    /**
     * Encode the maximum finite value.
     */
    private static int encodeMaxFinite(FormatParameters params, int sign) {
        int K = params.bitWidth();
        int M = params.mantissaWidth();
        int maxExp = params.maxBiasedExponent();

        if (params.finiteOnly()) {
            // Finite-only: max value uses all exponent and mantissa bits
            if (params.signed()) {
                return (sign << (K - 1)) | (maxExp << M) | ((1 << M) - 1);
            } else {
                return (maxExp << M) | ((1 << M) - 1);
            }
        } else {
            // P3109 Extended: infinity is only at max exp + max mantissa
            // So max finite uses max exp but (maxMant - 1)
            int maxMant = (1 << M) - 1;
            if (params.signed()) {
                // For signed: max finite is (maxExp, maxMant - 1)
                return (sign << (K - 1)) | (maxExp << M) | (maxMant - 1);
            } else {
                // For unsigned: NaN at max code, +Inf at second-to-last
                // So max finite is (maxExp, maxMant - 2)
                return (maxExp << M) | (maxMant - 2);
            }
        }
    }

    // ==================== Decoding (bits to float/double) ====================

    /**
     * Decode P3109 format bits to a float.
     *
     * @param bits The encoded bits (right-aligned)
     * @param params Format parameters
     * @return The decoded float value
     */
    public static float decodeToFloat(int bits, FormatParameters params) {
        return (float) decodeToDouble(bits, params);
    }

    /**
     * Decode P3109 format bits to a double.
     *
     * @param bits The encoded bits (right-aligned)
     * @param params Format parameters
     * @return The decoded double value
     */
    public static double decodeToDouble(int bits, FormatParameters params) {
        int K = params.bitWidth();
        int E = params.exponentWidth();
        int M = params.mantissaWidth();
        int bias = params.exponentBias();
        boolean signed = params.signed();
        boolean finiteOnly = params.finiteOnly();

        // Mask to valid bits
        int mask = (1 << K) - 1;
        bits = bits & mask;

        // Handle zero
        if (bits == 0) {
            return 0.0;
        }

        // Check for NaN
        if (isNaNBits(bits, params)) {
            return Double.NaN;
        }

        // Check for infinity
        if (!finiteOnly && isInfinityBits(bits, params)) {
            int sign = signed ? (bits >> (K - 1)) & 1 : 0;
            return sign == 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }

        // Extract fields
        int sign = signed ? (bits >> (K - 1)) & 1 : 0;
        int biasedExp = (bits >> M) & ((1 << E) - 1);
        int mantissa = bits & ((1 << M) - 1);

        // Compute value
        double value;
        if (biasedExp == 0) {
            // Subnormal: value = (-1)^s * 2^(1-bias) * (0.mantissa)
            double mantissaFrac = (double) mantissa / (1 << M);
            value = Math.scalb(mantissaFrac, 1 - bias);
        } else {
            // Normal: value = (-1)^s * 2^(exp-bias) * (1.mantissa)
            double mantissaFrac = 1.0 + (double) mantissa / (1 << M);
            value = Math.scalb(mantissaFrac, biasedExp - bias);
        }

        return sign == 0 ? value : -value;
    }

    /**
     * Check if bits represent NaN.
     */
    public static boolean isNaNBits(int bits, FormatParameters params) {
        if (params.finiteOnly()) {
            return false;
        }
        int K = params.bitWidth();
        int mask = (1 << K) - 1;
        bits = bits & mask;

        if (params.signed()) {
            // Signed extended: NaN is exactly 2^(K-1)
            return bits == (1 << (K - 1));
        } else {
            // Unsigned extended: NaN is all-1s
            return bits == mask;
        }
    }

    /**
     * Check if bits represent infinity.
     */
    public static boolean isInfinityBits(int bits, FormatParameters params) {
        if (params.finiteOnly()) {
            return false;
        }
        int K = params.bitWidth();
        int M = params.mantissaWidth();
        int maxExp = params.maxBiasedExponent();
        int mask = (1 << K) - 1;
        bits = bits & mask;

        if (params.signed()) {
            // +Inf: 0|exp=all1s|mant=all1s = (maxExp << M) | ((1<<M)-1)
            // -Inf: 1|exp=all1s|mant=all1s
            int posInf = (maxExp << M) | ((1 << M) - 1);
            int negInf = (1 << (K - 1)) | posInf;
            return bits == posInf || bits == negInf;
        } else {
            // Unsigned: +Inf at (1 << K) - 2
            return bits == ((1 << K) - 2);
        }
    }

    // ==================== Arithmetic Operations ====================

    /**
     * Add two P3109 values.
     */
    public static int add(int a, int b, FormatParameters params) {
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return encode(va + vb, params);
    }

    /**
     * Subtract two P3109 values (a - b).
     */
    public static int subtract(int a, int b, FormatParameters params) {
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return encode(va - vb, params);
    }

    /**
     * Multiply two P3109 values.
     */
    public static int multiply(int a, int b, FormatParameters params) {
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return encode(va * vb, params);
    }

    /**
     * Divide two P3109 values (a / b).
     */
    public static int divide(int a, int b, FormatParameters params) {
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return encode(va / vb, params);
    }

    /**
     * Fused multiply-add: (a * b) + c with single rounding.
     */
    public static int fma(int a, int b, int c, FormatParameters params) {
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        double vc = decodeToDouble(c, params);
        return encode(Math.fma(va, vb, vc), params);
    }

    /**
     * Negate a P3109 value.
     */
    public static int negate(int a, FormatParameters params) {
        if (!params.signed()) {
            // Unsigned: can't negate positive values
            return 0;
        }
        // Flip sign bit
        int K = params.bitWidth();
        int signBit = 1 << (K - 1);

        // Handle special cases
        if (a == 0) return 0; // -0 = 0 in P3109
        if (isNaNBits(a, params)) return a; // NaN stays NaN

        return a ^ signBit;
    }

    /**
     * Absolute value of a P3109 value.
     */
    public static int abs(int a, FormatParameters params) {
        if (!params.signed()) {
            return a; // Unsigned is always non-negative
        }
        if (isNaNBits(a, params)) {
            return a; // abs(NaN) = NaN
        }
        int K = params.bitWidth();
        int signBit = 1 << (K - 1);
        return a & ~signBit;
    }

    /**
     * Square root of a P3109 value.
     */
    public static int sqrt(int a, FormatParameters params) {
        double va = decodeToDouble(a, params);
        return encode(Math.sqrt(va), params);
    }

    /**
     * Minimum of two P3109 values (with IEEE 754 NaN semantics).
     */
    public static int min(int a, int b, FormatParameters params) {
        if (isNaNBits(a, params)) return b;
        if (isNaNBits(b, params)) return a;
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return encode(Math.min(va, vb), params);
    }

    /**
     * Maximum of two P3109 values (with IEEE 754 NaN semantics).
     */
    public static int max(int a, int b, FormatParameters params) {
        if (isNaNBits(a, params)) return b;
        if (isNaNBits(b, params)) return a;
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return encode(Math.max(va, vb), params);
    }

    // ==================== Comparison Operations ====================

    /**
     * Compare two P3109 values.
     * @return negative if a < b, zero if a == b, positive if a > b
     */
    public static int compare(int a, int b, FormatParameters params) {
        if (isNaNBits(a, params) || isNaNBits(b, params)) {
            return 0; // NaN comparisons return unordered (0)
        }
        double va = decodeToDouble(a, params);
        double vb = decodeToDouble(b, params);
        return Double.compare(va, vb);
    }

    /**
     * Check if two P3109 values are equal.
     */
    public static boolean equals(int a, int b, FormatParameters params) {
        if (isNaNBits(a, params) || isNaNBits(b, params)) {
            return false; // NaN != NaN
        }
        // Note: In P3109 there's only one zero, so no -0 == +0 issue
        return a == b;
    }

    /**
     * Check if a P3109 value is less than another.
     */
    public static boolean lessThan(int a, int b, FormatParameters params) {
        return compare(a, b, params) < 0;
    }

    /**
     * Check if a P3109 value is less than or equal to another.
     */
    public static boolean lessOrEqual(int a, int b, FormatParameters params) {
        if (isNaNBits(a, params) || isNaNBits(b, params)) {
            return false;
        }
        return compare(a, b, params) <= 0;
    }

    // ==================== Bulk Operations on MemorySegment ====================

    /**
     * Encode a float array to a MemorySegment of packed P3109 values.
     *
     * @param source Source float values
     * @param dest Destination segment (must be large enough)
     * @param params Format parameters
     */
    public static void encodeBulk(float[] source, MemorySegment dest, FormatParameters params) {
        int K = params.bitWidth();
        int count = source.length;

        switch (K) {
            case 4 -> encodeBulk4Bit(source, dest, params, count);
            case 6 -> encodeBulk6Bit(source, dest, params, count);
            case 8 -> encodeBulk8Bit(source, dest, params, count);
            default -> encodeBulkGeneric(source, dest, params, count);
        }
    }

    private static void encodeBulk4Bit(float[] source, MemorySegment dest, FormatParameters params, int count) {
        // Pack two 4-bit values per byte
        int byteIdx = 0;
        for (int i = 0; i < count - 1; i += 2) {
            int lo = encode(source[i], params) & 0x0F;
            int hi = encode(source[i + 1], params) & 0x0F;
            dest.set(ValueLayout.JAVA_BYTE, byteIdx++, (byte) (lo | (hi << 4)));
        }
        // Handle odd count
        if ((count & 1) != 0) {
            int lo = encode(source[count - 1], params) & 0x0F;
            dest.set(ValueLayout.JAVA_BYTE, byteIdx, (byte) lo);
        }
    }

    private static void encodeBulk6Bit(float[] source, MemorySegment dest, FormatParameters params, int count) {
        // Pack four 6-bit values into 3 bytes
        int byteIdx = 0;
        int i = 0;
        while (i + 3 < count) {
            int v0 = encode(source[i], params) & 0x3F;
            int v1 = encode(source[i + 1], params) & 0x3F;
            int v2 = encode(source[i + 2], params) & 0x3F;
            int v3 = encode(source[i + 3], params) & 0x3F;

            // Pack: [v0:6][v1:2] [v1:4][v2:4] [v2:2][v3:6]
            dest.set(ValueLayout.JAVA_BYTE, byteIdx++, (byte) (v0 | ((v1 & 0x03) << 6)));
            dest.set(ValueLayout.JAVA_BYTE, byteIdx++, (byte) ((v1 >> 2) | ((v2 & 0x0F) << 4)));
            dest.set(ValueLayout.JAVA_BYTE, byteIdx++, (byte) ((v2 >> 4) | (v3 << 2)));
            i += 4;
        }
        // Handle remaining (pad with zeros)
        if (i < count) {
            int remaining = count - i;
            int[] vals = new int[4];
            for (int j = 0; j < remaining; j++) {
                vals[j] = encode(source[i + j], params) & 0x3F;
            }
            dest.set(ValueLayout.JAVA_BYTE, byteIdx++, (byte) (vals[0] | ((vals[1] & 0x03) << 6)));
            if (remaining > 1) {
                dest.set(ValueLayout.JAVA_BYTE, byteIdx++, (byte) ((vals[1] >> 2) | ((vals[2] & 0x0F) << 4)));
            }
            if (remaining > 2) {
                dest.set(ValueLayout.JAVA_BYTE, byteIdx, (byte) ((vals[2] >> 4) | (vals[3] << 2)));
            }
        }
    }

    private static void encodeBulk8Bit(float[] source, MemorySegment dest, FormatParameters params, int count) {
        // One byte per value
        for (int i = 0; i < count; i++) {
            dest.set(ValueLayout.JAVA_BYTE, i, (byte) encode(source[i], params));
        }
    }

    private static void encodeBulkGeneric(float[] source, MemorySegment dest, FormatParameters params, int count) {
        // Generic bit-packing for unusual widths
        int K = params.bitWidth();
        long bitOffset = 0;
        for (int i = 0; i < count; i++) {
            int bits = encode(source[i], params);
            setBitsInSegment(dest, bitOffset, K, bits);
            bitOffset += K;
        }
    }

    /**
     * Decode a MemorySegment of packed P3109 values to a float array.
     *
     * @param source Source segment of packed values
     * @param dest Destination float array
     * @param params Format parameters
     */
    public static void decodeBulk(MemorySegment source, float[] dest, FormatParameters params) {
        int K = params.bitWidth();
        int count = dest.length;

        switch (K) {
            case 4 -> decodeBulk4Bit(source, dest, params, count);
            case 6 -> decodeBulk6Bit(source, dest, params, count);
            case 8 -> decodeBulk8Bit(source, dest, params, count);
            default -> decodeBulkGeneric(source, dest, params, count);
        }
    }

    private static void decodeBulk4Bit(MemorySegment source, float[] dest, FormatParameters params, int count) {
        int byteIdx = 0;
        for (int i = 0; i < count - 1; i += 2) {
            int packed = source.get(ValueLayout.JAVA_BYTE, byteIdx++) & 0xFF;
            dest[i] = decodeToFloat(packed & 0x0F, params);
            dest[i + 1] = decodeToFloat(packed >> 4, params);
        }
        if ((count & 1) != 0) {
            int packed = source.get(ValueLayout.JAVA_BYTE, byteIdx) & 0xFF;
            dest[count - 1] = decodeToFloat(packed & 0x0F, params);
        }
    }

    private static void decodeBulk6Bit(MemorySegment source, float[] dest, FormatParameters params, int count) {
        int byteIdx = 0;
        int i = 0;
        while (i + 3 < count) {
            int b0 = source.get(ValueLayout.JAVA_BYTE, byteIdx++) & 0xFF;
            int b1 = source.get(ValueLayout.JAVA_BYTE, byteIdx++) & 0xFF;
            int b2 = source.get(ValueLayout.JAVA_BYTE, byteIdx++) & 0xFF;

            dest[i++] = decodeToFloat(b0 & 0x3F, params);
            dest[i++] = decodeToFloat(((b0 >> 6) | (b1 << 2)) & 0x3F, params);
            dest[i++] = decodeToFloat(((b1 >> 4) | (b2 << 4)) & 0x3F, params);
            dest[i++] = decodeToFloat(b2 >> 2, params);
        }
        // Handle remaining
        if (i < count) {
            int remaining = count - i;
            int b0 = source.get(ValueLayout.JAVA_BYTE, byteIdx++) & 0xFF;
            dest[i++] = decodeToFloat(b0 & 0x3F, params);
            if (i < count && remaining > 1) {
                int b1 = source.get(ValueLayout.JAVA_BYTE, byteIdx++) & 0xFF;
                dest[i++] = decodeToFloat(((b0 >> 6) | (b1 << 2)) & 0x3F, params);
                if (i < count && remaining > 2) {
                    int b2 = source.get(ValueLayout.JAVA_BYTE, byteIdx) & 0xFF;
                    dest[i] = decodeToFloat(((b1 >> 4) | (b2 << 4)) & 0x3F, params);
                }
            }
        }
    }

    private static void decodeBulk8Bit(MemorySegment source, float[] dest, FormatParameters params, int count) {
        for (int i = 0; i < count; i++) {
            dest[i] = decodeToFloat(source.get(ValueLayout.JAVA_BYTE, i) & 0xFF, params);
        }
    }

    private static void decodeBulkGeneric(MemorySegment source, float[] dest, FormatParameters params, int count) {
        int K = params.bitWidth();
        long bitOffset = 0;
        for (int i = 0; i < count; i++) {
            dest[i] = decodeToFloat(getBitsFromSegment(source, bitOffset, K), params);
            bitOffset += K;
        }
    }

    // ==================== Bit Manipulation Helpers ====================

    private static void setBitsInSegment(MemorySegment seg, long bitOffset, int numBits, int value) {
        long byteOffset = bitOffset / 8;
        int bitInByte = (int) (bitOffset % 8);

        while (numBits > 0) {
            int bitsThisByte = Math.min(8 - bitInByte, numBits);
            int mask = ((1 << bitsThisByte) - 1) << bitInByte;
            int currentByte = seg.get(ValueLayout.JAVA_BYTE, byteOffset) & 0xFF;
            currentByte = (currentByte & ~mask) | ((value << bitInByte) & mask);
            seg.set(ValueLayout.JAVA_BYTE, byteOffset, (byte) currentByte);

            value >>= bitsThisByte;
            numBits -= bitsThisByte;
            bitInByte = 0;
            byteOffset++;
        }
    }

    private static int getBitsFromSegment(MemorySegment seg, long bitOffset, int numBits) {
        long byteOffset = bitOffset / 8;
        int bitInByte = (int) (bitOffset % 8);
        int result = 0;
        int resultBit = 0;

        while (numBits > 0) {
            int bitsThisByte = Math.min(8 - bitInByte, numBits);
            int currentByte = seg.get(ValueLayout.JAVA_BYTE, byteOffset) & 0xFF;
            int extracted = (currentByte >> bitInByte) & ((1 << bitsThisByte) - 1);
            result |= extracted << resultBit;

            resultBit += bitsThisByte;
            numBits -= bitsThisByte;
            bitInByte = 0;
            byteOffset++;
        }

        return result;
    }

    // ==================== Utility Methods ====================

    /**
     * Calculate required byte size for storing count elements.
     */
    public static long byteSize(int count, FormatParameters params) {
        long totalBits = (long) count * params.bitWidth();
        return (totalBits + 7) / 8;
    }

    /**
     * Convert between two P3109 formats.
     */
    public static int convert(int bits, FormatParameters fromParams, FormatParameters toParams) {
        double value = decodeToDouble(bits, fromParams);
        return encode(value, toParams);
    }
}
