package io.surfworks.warpforge.core.formats;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Sealed interface for 4-bit floating-point formats.
 *
 * <p>4-bit formats are highly compressed representations used in modern
 * inference workloads. They typically require block scaling to maintain
 * acceptable accuracy.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link E2M1} - Binary4p2se, 1 mantissa bit (used in NVFP4, MXFP4)</li>
 *   <li>{@link E1M2} - Binary4p3se, 2 mantissa bits (higher precision variant)</li>
 * </ul>
 */
public sealed interface Float4 permits Float4.E2M1, Float4.E1M2 {

    /**
     * Get the raw 4-bit encoding (in lower 4 bits).
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

    // ==================== E2M1 (Binary4p2se) ====================

    /**
     * FP4 E2M1 format (Binary4p2se).
     *
     * <p>The most common 4-bit format, used by NVIDIA NVFP4 and OCP MXFP4.
     *
     * <ul>
     *   <li>1 sign bit, 2-bit exponent, 1-bit mantissa</li>
     *   <li>16 distinct values</li>
     *   <li>Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6, NaN, ±Inf</li>
     *   <li>Used for: NVFP4, MXFP4 block formats</li>
     * </ul>
     *
     * <h3>Value Table (E2M1)</h3>
     * <pre>
     * Code | Sign | Exp | Mant | Value
     * -----|------|-----|------|-------
     * 0000 |  0   |  0  |   0  | +0
     * 0001 |  0   |  0  |   1  | +0.5 (subnormal)
     * 0010 |  0   |  1  |   0  | +1.0
     * 0011 |  0   |  1  |   1  | +1.5
     * 0100 |  0   |  2  |   0  | +2.0
     * 0101 |  0   |  2  |   1  | +3.0
     * 0110 |  0   |  3  |   0  | +4.0
     * 0111 |  0   |  3  |   1  | +6.0 or +Inf (depending on finiteOnly)
     * 1000 |  1   |  0  |   0  | NaN (in P3109)
     * 1001 |  1   |  0  |   1  | -0.5
     * 1010 |  1   |  1  |   0  | -1.0
     * 1011 |  1   |  1  |   1  | -1.5
     * 1100 |  1   |  2  |   0  | -2.0
     * 1101 |  1   |  2  |   1  | -3.0
     * 1110 |  1   |  3  |   0  | -4.0
     * 1111 |  1   |  3  |   1  | -6.0 or -Inf
     * </pre>
     */
    record E2M1(byte bits) implements Float4 {
        private static final FormatParameters PARAMS = FormatParameters.FP4_E2M1;

        public E2M1 {
            // Mask to 4 bits
            bits = (byte) (bits & 0x0F);
        }

        public static E2M1 fromFloat(float value) {
            return new E2M1((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E2M1 fromDouble(double value) {
            return new E2M1((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E2M1 fromBits(int bits) {
            return new E2M1((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0x0F, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0x0F, PARAMS);
        }

        @Override
        public boolean isNaN() {
            return false; // Finite-only format has no NaN
        }

        @Override
        public boolean isInfinite() {
            return false; // Finite-only format has no infinity
        }

        @Override
        public boolean isZero() {
            return (bits & 0x0F) == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E2M1 add(E2M1 other) {
            return new E2M1((byte) MiniFloat.add(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E2M1 subtract(E2M1 other) {
            return new E2M1((byte) MiniFloat.subtract(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E2M1 multiply(E2M1 other) {
            return new E2M1((byte) MiniFloat.multiply(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E2M1 divide(E2M1 other) {
            return new E2M1((byte) MiniFloat.divide(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E2M1 negate() {
            return new E2M1((byte) MiniFloat.negate(bits & 0x0F, PARAMS));
        }

        public E2M1 abs() {
            return new E2M1((byte) MiniFloat.abs(bits & 0x0F, PARAMS));
        }

        public static E2M1 fma(E2M1 a, E2M1 b, E2M1 c) {
            return new E2M1((byte) MiniFloat.fma(a.bits & 0x0F, b.bits & 0x0F, c.bits & 0x0F, PARAMS));
        }

        public static E2M1 min(E2M1 a, E2M1 b) {
            return new E2M1((byte) MiniFloat.min(a.bits & 0x0F, b.bits & 0x0F, PARAMS));
        }

        public static E2M1 max(E2M1 a, E2M1 b) {
            return new E2M1((byte) MiniFloat.max(a.bits & 0x0F, b.bits & 0x0F, PARAMS));
        }

        // Bulk operations (packed 2 values per byte)
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        /**
         * Byte size for count elements (2 elements per byte).
         */
        public static long byteSize(int count) {
            return (count + 1) / 2;
        }

        // Constants
        public static final E2M1 ZERO = new E2M1((byte) 0x0);
        public static final E2M1 HALF = new E2M1((byte) 0x1);          // 0.5
        public static final E2M1 ONE = new E2M1((byte) 0x2);           // 1.0
        public static final E2M1 ONE_HALF = new E2M1((byte) 0x3);      // 1.5
        public static final E2M1 TWO = new E2M1((byte) 0x4);           // 2.0
        public static final E2M1 THREE = new E2M1((byte) 0x5);         // 3.0
        public static final E2M1 FOUR = new E2M1((byte) 0x6);          // 4.0
        public static final E2M1 SIX = new E2M1((byte) 0x7);           // 6.0
        // Note: Code 0x8 is -0 in finite-only format, treated as 0
        public static final E2M1 NEG_HALF = new E2M1((byte) 0x9);      // -0.5
        public static final E2M1 NEG_ONE = new E2M1((byte) 0xA);       // -1.0
        public static final E2M1 NEG_ONE_HALF = new E2M1((byte) 0xB);  // -1.5
        public static final E2M1 NEG_TWO = new E2M1((byte) 0xC);       // -2.0
        public static final E2M1 NEG_THREE = new E2M1((byte) 0xD);     // -3.0
        public static final E2M1 NEG_FOUR = new E2M1((byte) 0xE);      // -4.0
        public static final E2M1 NEG_SIX = new E2M1((byte) 0xF);       // -6.0 (or -Inf)

        @Override
        public String toString() {
            return "E2M1(" + toFloat() + ")";
        }
    }

    // ==================== E1M2 (Binary4p3se) ====================

    /**
     * FP4 E1M2 format (Binary4p3se).
     *
     * <p>Alternative 4-bit format with higher precision but narrower range.
     *
     * <ul>
     *   <li>1 sign bit, 1-bit exponent, 2-bit mantissa</li>
     *   <li>16 distinct values</li>
     *   <li>Representable values: 0, ±0.25, ±0.5, ±0.75, ±1, ±1.25, ±1.5, ±1.75, NaN, ±Inf</li>
     *   <li>Used for: applications requiring finer gradations near 1.0</li>
     * </ul>
     */
    record E1M2(byte bits) implements Float4 {
        private static final FormatParameters PARAMS = FormatParameters.FP4_E1M2;

        public E1M2 {
            bits = (byte) (bits & 0x0F);
        }

        public static E1M2 fromFloat(float value) {
            return new E1M2((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E1M2 fromDouble(double value) {
            return new E1M2((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E1M2 fromBits(int bits) {
            return new E1M2((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0x0F, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0x0F, PARAMS);
        }

        @Override
        public boolean isNaN() {
            return false; // Finite-only format has no NaN
        }

        @Override
        public boolean isInfinite() {
            return false; // Finite-only format has no infinity
        }

        @Override
        public boolean isZero() {
            return (bits & 0x0F) == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E1M2 add(E1M2 other) {
            return new E1M2((byte) MiniFloat.add(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E1M2 subtract(E1M2 other) {
            return new E1M2((byte) MiniFloat.subtract(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E1M2 multiply(E1M2 other) {
            return new E1M2((byte) MiniFloat.multiply(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E1M2 divide(E1M2 other) {
            return new E1M2((byte) MiniFloat.divide(bits & 0x0F, other.bits & 0x0F, PARAMS));
        }

        public E1M2 negate() {
            return new E1M2((byte) MiniFloat.negate(bits & 0x0F, PARAMS));
        }

        public E1M2 abs() {
            return new E1M2((byte) MiniFloat.abs(bits & 0x0F, PARAMS));
        }

        public static E1M2 fma(E1M2 a, E1M2 b, E1M2 c) {
            return new E1M2((byte) MiniFloat.fma(a.bits & 0x0F, b.bits & 0x0F, c.bits & 0x0F, PARAMS));
        }

        public static E1M2 min(E1M2 a, E1M2 b) {
            return new E1M2((byte) MiniFloat.min(a.bits & 0x0F, b.bits & 0x0F, PARAMS));
        }

        public static E1M2 max(E1M2 a, E1M2 b) {
            return new E1M2((byte) MiniFloat.max(a.bits & 0x0F, b.bits & 0x0F, PARAMS));
        }

        // Bulk operations
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        public static long byteSize(int count) {
            return (count + 1) / 2;
        }

        // Constants
        public static final E1M2 ZERO = new E1M2((byte) 0x0);
        public static final E1M2 ONE = fromFloat(1.0f);
        public static final E1M2 NEGATIVE_ONE = fromFloat(-1.0f);
        // Note: No NaN in finite-only format

        @Override
        public String toString() {
            return "E1M2(" + toFloat() + ")";
        }
    }
}
