package io.surfworks.warpforge.core.formats;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Sealed interface for 8-bit floating-point formats.
 *
 * <p>This hierarchy provides type-safe wrappers around the raw bit operations
 * in {@link MiniFloat}, ensuring format parameters are not accidentally mixed.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link E5M2} - Binary8p3se, wide dynamic range</li>
 *   <li>{@link E4M3} - Binary8p4se, higher precision</li>
 *   <li>{@link E4M3FN} - Binary8p4sf, finite-only variant</li>
 *   <li>{@link E8M0} - Binary8p1uf, exponent-only scale factor</li>
 * </ul>
 */
public sealed interface Float8 permits Float8.E5M2, Float8.E4M3, Float8.E4M3FN, Float8.E8M0 {

    /**
     * Get the raw 8-bit encoding.
     */
    byte bits();

    /**
     * Decode to float.
     */
    float toFloat();

    /**
     * Decode to double.
     */
    double toDouble();

    /**
     * Check if this value is NaN.
     */
    boolean isNaN();

    /**
     * Check if this value is infinite.
     */
    boolean isInfinite();

    /**
     * Check if this value is zero.
     */
    boolean isZero();

    /**
     * Get the format parameters for this type.
     */
    FormatParameters parameters();

    // ==================== E5M2 (Binary8p3se) ====================

    /**
     * FP8 E5M2 format (Binary8p3se).
     *
     * <ul>
     *   <li>5-bit exponent, 2-bit mantissa</li>
     *   <li>Range: ±65504</li>
     *   <li>Smallest subnormal: 2^-16</li>
     *   <li>Precision: ~0.5 decimal digits</li>
     *   <li>Used for: gradients, activations requiring wide range</li>
     * </ul>
     */
    record E5M2(byte bits) implements Float8 {
        private static final FormatParameters PARAMS = FormatParameters.FP8_E5M2;

        public static E5M2 fromFloat(float value) {
            return new E5M2((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E5M2 fromDouble(double value) {
            return new E5M2((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E5M2 fromBits(int bits) {
            return new E5M2((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0xFF, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isNaN() {
            return MiniFloat.isNaNBits(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isInfinite() {
            return MiniFloat.isInfinityBits(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isZero() {
            return bits == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E5M2 add(E5M2 other) {
            return new E5M2((byte) MiniFloat.add(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E5M2 subtract(E5M2 other) {
            return new E5M2((byte) MiniFloat.subtract(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E5M2 multiply(E5M2 other) {
            return new E5M2((byte) MiniFloat.multiply(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E5M2 divide(E5M2 other) {
            return new E5M2((byte) MiniFloat.divide(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E5M2 negate() {
            return new E5M2((byte) MiniFloat.negate(bits & 0xFF, PARAMS));
        }

        public E5M2 abs() {
            return new E5M2((byte) MiniFloat.abs(bits & 0xFF, PARAMS));
        }

        public static E5M2 fma(E5M2 a, E5M2 b, E5M2 c) {
            return new E5M2((byte) MiniFloat.fma(a.bits & 0xFF, b.bits & 0xFF, c.bits & 0xFF, PARAMS));
        }

        public static E5M2 min(E5M2 a, E5M2 b) {
            return new E5M2((byte) MiniFloat.min(a.bits & 0xFF, b.bits & 0xFF, PARAMS));
        }

        public static E5M2 max(E5M2 a, E5M2 b) {
            return new E5M2((byte) MiniFloat.max(a.bits & 0xFF, b.bits & 0xFF, PARAMS));
        }

        // Bulk operations
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        public static long byteSize(int count) {
            return count; // 1 byte per element
        }

        // Constants
        public static final E5M2 ZERO = new E5M2((byte) 0);
        public static final E5M2 ONE = fromFloat(1.0f);
        public static final E5M2 NEGATIVE_ONE = fromFloat(-1.0f);
        public static final E5M2 NAN = new E5M2((byte) 0x80); // Sign bit set, rest zero
        public static final E5M2 POSITIVE_INFINITY = fromFloat(Float.POSITIVE_INFINITY);
        public static final E5M2 NEGATIVE_INFINITY = fromFloat(Float.NEGATIVE_INFINITY);
        public static final E5M2 MAX_VALUE = fromFloat((float) PARAMS.maxValue());
        public static final E5M2 MIN_NORMAL = fromFloat((float) PARAMS.minNormal());
        public static final E5M2 MIN_VALUE = fromFloat((float) PARAMS.minSubnormal());

        @Override
        public String toString() {
            if (isNaN()) return "E5M2(NaN)";
            if (isInfinite()) return toFloat() > 0 ? "E5M2(+Inf)" : "E5M2(-Inf)";
            return "E5M2(" + toFloat() + ")";
        }
    }

    // ==================== E4M3 (Binary8p4se) ====================

    /**
     * FP8 E4M3 format (Binary8p4se).
     *
     * <ul>
     *   <li>4-bit exponent, 3-bit mantissa</li>
     *   <li>Range: ±240</li>
     *   <li>Smallest subnormal: 2^-9</li>
     *   <li>Precision: ~0.9 decimal digits</li>
     *   <li>Used for: weights, activations requiring higher precision</li>
     * </ul>
     */
    record E4M3(byte bits) implements Float8 {
        private static final FormatParameters PARAMS = FormatParameters.FP8_E4M3;

        public static E4M3 fromFloat(float value) {
            return new E4M3((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E4M3 fromDouble(double value) {
            return new E4M3((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E4M3 fromBits(int bits) {
            return new E4M3((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0xFF, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isNaN() {
            return MiniFloat.isNaNBits(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isInfinite() {
            return MiniFloat.isInfinityBits(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isZero() {
            return bits == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E4M3 add(E4M3 other) {
            return new E4M3((byte) MiniFloat.add(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3 subtract(E4M3 other) {
            return new E4M3((byte) MiniFloat.subtract(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3 multiply(E4M3 other) {
            return new E4M3((byte) MiniFloat.multiply(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3 divide(E4M3 other) {
            return new E4M3((byte) MiniFloat.divide(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3 negate() {
            return new E4M3((byte) MiniFloat.negate(bits & 0xFF, PARAMS));
        }

        public E4M3 abs() {
            return new E4M3((byte) MiniFloat.abs(bits & 0xFF, PARAMS));
        }

        public static E4M3 fma(E4M3 a, E4M3 b, E4M3 c) {
            return new E4M3((byte) MiniFloat.fma(a.bits & 0xFF, b.bits & 0xFF, c.bits & 0xFF, PARAMS));
        }

        public static E4M3 min(E4M3 a, E4M3 b) {
            return new E4M3((byte) MiniFloat.min(a.bits & 0xFF, b.bits & 0xFF, PARAMS));
        }

        public static E4M3 max(E4M3 a, E4M3 b) {
            return new E4M3((byte) MiniFloat.max(a.bits & 0xFF, b.bits & 0xFF, PARAMS));
        }

        // Bulk operations
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        public static long byteSize(int count) {
            return count;
        }

        // Constants
        public static final E4M3 ZERO = new E4M3((byte) 0);
        public static final E4M3 ONE = fromFloat(1.0f);
        public static final E4M3 NEGATIVE_ONE = fromFloat(-1.0f);
        public static final E4M3 NAN = new E4M3((byte) 0x80);
        public static final E4M3 POSITIVE_INFINITY = fromFloat(Float.POSITIVE_INFINITY);
        public static final E4M3 NEGATIVE_INFINITY = fromFloat(Float.NEGATIVE_INFINITY);
        public static final E4M3 MAX_VALUE = fromFloat((float) PARAMS.maxValue());
        public static final E4M3 MIN_NORMAL = fromFloat((float) PARAMS.minNormal());
        public static final E4M3 MIN_VALUE = fromFloat((float) PARAMS.minSubnormal());

        @Override
        public String toString() {
            if (isNaN()) return "E4M3(NaN)";
            if (isInfinite()) return toFloat() > 0 ? "E4M3(+Inf)" : "E4M3(-Inf)";
            return "E4M3(" + toFloat() + ")";
        }
    }

    // ==================== E4M3FN (Binary8p4sf - Finite-only) ====================

    /**
     * FP8 E4M3FN format (Binary8p4sf).
     *
     * <p>NVIDIA's finite-only variant of E4M3. No infinity or NaN representations,
     * allowing the full range of bit patterns to represent finite values.
     *
     * <ul>
     *   <li>4-bit exponent, 3-bit mantissa</li>
     *   <li>Range: ±448 (larger than E4M3 due to no inf)</li>
     *   <li>No NaN or infinity</li>
     *   <li>Used for: NVIDIA H100/Blackwell inference</li>
     * </ul>
     */
    record E4M3FN(byte bits) implements Float8 {
        private static final FormatParameters PARAMS = FormatParameters.FP8_E4M3FN;

        public static E4M3FN fromFloat(float value) {
            return new E4M3FN((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E4M3FN fromDouble(double value) {
            return new E4M3FN((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E4M3FN fromBits(int bits) {
            return new E4M3FN((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0xFF, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0xFF, PARAMS);
        }

        @Override
        public boolean isNaN() {
            return false; // Finite-only format
        }

        @Override
        public boolean isInfinite() {
            return false; // Finite-only format
        }

        @Override
        public boolean isZero() {
            return bits == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E4M3FN add(E4M3FN other) {
            return new E4M3FN((byte) MiniFloat.add(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3FN subtract(E4M3FN other) {
            return new E4M3FN((byte) MiniFloat.subtract(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3FN multiply(E4M3FN other) {
            return new E4M3FN((byte) MiniFloat.multiply(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3FN divide(E4M3FN other) {
            return new E4M3FN((byte) MiniFloat.divide(bits & 0xFF, other.bits & 0xFF, PARAMS));
        }

        public E4M3FN negate() {
            return new E4M3FN((byte) MiniFloat.negate(bits & 0xFF, PARAMS));
        }

        public E4M3FN abs() {
            return new E4M3FN((byte) MiniFloat.abs(bits & 0xFF, PARAMS));
        }

        public static E4M3FN fma(E4M3FN a, E4M3FN b, E4M3FN c) {
            return new E4M3FN((byte) MiniFloat.fma(a.bits & 0xFF, b.bits & 0xFF, c.bits & 0xFF, PARAMS));
        }

        public static E4M3FN min(E4M3FN a, E4M3FN b) {
            return new E4M3FN((byte) MiniFloat.min(a.bits & 0xFF, b.bits & 0xFF, PARAMS));
        }

        public static E4M3FN max(E4M3FN a, E4M3FN b) {
            return new E4M3FN((byte) MiniFloat.max(a.bits & 0xFF, b.bits & 0xFF, PARAMS));
        }

        // Bulk operations
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        public static long byteSize(int count) {
            return count;
        }

        // Constants
        public static final E4M3FN ZERO = new E4M3FN((byte) 0);
        public static final E4M3FN ONE = fromFloat(1.0f);
        public static final E4M3FN NEGATIVE_ONE = fromFloat(-1.0f);
        public static final E4M3FN MAX_VALUE = fromFloat((float) PARAMS.maxValue());
        public static final E4M3FN MIN_NORMAL = fromFloat((float) PARAMS.minNormal());
        public static final E4M3FN MIN_VALUE = fromFloat((float) PARAMS.minSubnormal());

        @Override
        public String toString() {
            return "E4M3FN(" + toFloat() + ")";
        }
    }

    // ==================== E8M0 (Binary8p1uf - Scale Factor) ====================

    /**
     * FP8 E8M0 format (Binary8p1uf).
     *
     * <p>Exponent-only format used as a scale factor in block-scaled formats
     * like OCP MX. Unsigned, finite-only.
     *
     * <ul>
     *   <li>8-bit exponent, 0-bit mantissa</li>
     *   <li>Represents powers of 2: 2^(code - 127)</li>
     *   <li>Range: 2^-127 to 2^127</li>
     *   <li>Used for: block scale factors in MXFP4/MXFP8</li>
     * </ul>
     */
    record E8M0(byte bits) implements Float8 {
        private static final FormatParameters PARAMS = FormatParameters.FP8_E8M0;
        private static final int BIAS = 127;

        public static E8M0 fromFloat(float value) {
            return new E8M0((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E8M0 fromDouble(double value) {
            return new E8M0((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E8M0 fromBits(int bits) {
            return new E8M0((byte) bits);
        }

        /**
         * Create from a power-of-2 exponent.
         * The value will represent 2^exponent.
         */
        public static E8M0 fromExponent(int exponent) {
            int biased = exponent + BIAS;
            if (biased < 0) biased = 0;
            if (biased > 255) biased = 255;
            return new E8M0((byte) biased);
        }

        @Override
        public float toFloat() {
            return (float) toDouble();
        }

        @Override
        public double toDouble() {
            int biased = bits & 0xFF;
            if (biased == 0) return 0.0; // Special: zero
            return Math.scalb(1.0, biased - BIAS);
        }

        /**
         * Get the unbiased exponent.
         */
        public int exponent() {
            return (bits & 0xFF) - BIAS;
        }

        @Override
        public boolean isNaN() {
            return false; // Finite-only
        }

        @Override
        public boolean isInfinite() {
            return false; // Finite-only
        }

        @Override
        public boolean isZero() {
            return bits == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        /**
         * Multiply two scale factors (add exponents).
         */
        public E8M0 multiply(E8M0 other) {
            int exp = this.exponent() + other.exponent();
            return fromExponent(exp);
        }

        /**
         * Divide two scale factors (subtract exponents).
         */
        public E8M0 divide(E8M0 other) {
            int exp = this.exponent() - other.exponent();
            return fromExponent(exp);
        }

        // Bulk operations
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        public static long byteSize(int count) {
            return count;
        }

        // Constants
        public static final E8M0 ZERO = new E8M0((byte) 0);
        public static final E8M0 ONE = fromExponent(0);  // 2^0 = 1
        public static final E8M0 TWO = fromExponent(1);  // 2^1 = 2
        public static final E8M0 HALF = fromExponent(-1); // 2^-1 = 0.5

        @Override
        public String toString() {
            if (isZero()) return "E8M0(0)";
            return "E8M0(2^" + exponent() + " = " + toFloat() + ")";
        }
    }
}
