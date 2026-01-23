package io.surfworks.warpforge.core.formats;

import java.lang.foreign.MemorySegment;

/**
 * Sealed interface for 6-bit floating-point formats.
 *
 * <p>6-bit formats provide a middle ground between FP4 (very compact) and
 * FP8 (better accuracy). They are part of the OCP MX specification.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link E3M2} - Binary6p3se, wider dynamic range</li>
 *   <li>{@link E2M3} - Binary6p4se, higher precision</li>
 * </ul>
 */
public sealed interface Float6 permits Float6.E3M2, Float6.E2M3 {

    /**
     * Get the raw 6-bit encoding (in lower 6 bits).
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

    // ==================== E3M2 (Binary6p3se) ====================

    /**
     * FP6 E3M2 format (Binary6p3se).
     *
     * <p>OCP MXFP6 variant with wider dynamic range.
     *
     * <ul>
     *   <li>1 sign bit, 3-bit exponent, 2-bit mantissa</li>
     *   <li>64 distinct values</li>
     *   <li>Range: ±28.0</li>
     *   <li>Used for: OCP MXFP6 when range is more important</li>
     * </ul>
     */
    record E3M2(byte bits) implements Float6 {
        private static final FormatParameters PARAMS = FormatParameters.FP6_E3M2;

        public E3M2 {
            bits = (byte) (bits & 0x3F);
        }

        public static E3M2 fromFloat(float value) {
            return new E3M2((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E3M2 fromDouble(double value) {
            return new E3M2((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E3M2 fromBits(int bits) {
            return new E3M2((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0x3F, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0x3F, PARAMS);
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
            return (bits & 0x3F) == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E3M2 add(E3M2 other) {
            return new E3M2((byte) MiniFloat.add(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E3M2 subtract(E3M2 other) {
            return new E3M2((byte) MiniFloat.subtract(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E3M2 multiply(E3M2 other) {
            return new E3M2((byte) MiniFloat.multiply(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E3M2 divide(E3M2 other) {
            return new E3M2((byte) MiniFloat.divide(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E3M2 negate() {
            return new E3M2((byte) MiniFloat.negate(bits & 0x3F, PARAMS));
        }

        public E3M2 abs() {
            return new E3M2((byte) MiniFloat.abs(bits & 0x3F, PARAMS));
        }

        public static E3M2 fma(E3M2 a, E3M2 b, E3M2 c) {
            return new E3M2((byte) MiniFloat.fma(a.bits & 0x3F, b.bits & 0x3F, c.bits & 0x3F, PARAMS));
        }

        public static E3M2 min(E3M2 a, E3M2 b) {
            return new E3M2((byte) MiniFloat.min(a.bits & 0x3F, b.bits & 0x3F, PARAMS));
        }

        public static E3M2 max(E3M2 a, E3M2 b) {
            return new E3M2((byte) MiniFloat.max(a.bits & 0x3F, b.bits & 0x3F, PARAMS));
        }

        // Bulk operations (4 values in 3 bytes)
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        /**
         * Byte size for count elements (4 elements in 3 bytes).
         */
        public static long byteSize(int count) {
            return MiniFloat.byteSize(count, PARAMS);
        }

        // Constants
        public static final E3M2 ZERO = new E3M2((byte) 0);
        public static final E3M2 ONE = fromFloat(1.0f);
        public static final E3M2 NEGATIVE_ONE = fromFloat(-1.0f);
        // Note: No NaN in finite-only format
        public static final E3M2 MAX_VALUE = fromFloat((float) PARAMS.maxValue());
        public static final E3M2 MIN_NORMAL = fromFloat((float) PARAMS.minNormal());

        @Override
        public String toString() {
            return "E3M2(" + toFloat() + ")";
        }
    }

    // ==================== E2M3 (Binary6p4se) ====================

    /**
     * FP6 E2M3 format (Binary6p4se).
     *
     * <p>OCP MXFP6 variant with higher precision.
     *
     * <ul>
     *   <li>1 sign bit, 2-bit exponent, 3-bit mantissa</li>
     *   <li>64 distinct values</li>
     *   <li>Range: ±7.5</li>
     *   <li>Precision: better than E3M2</li>
     *   <li>Used for: OCP MXFP6 when precision is more important</li>
     * </ul>
     */
    record E2M3(byte bits) implements Float6 {
        private static final FormatParameters PARAMS = FormatParameters.FP6_E2M3;

        public E2M3 {
            bits = (byte) (bits & 0x3F);
        }

        public static E2M3 fromFloat(float value) {
            return new E2M3((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E2M3 fromDouble(double value) {
            return new E2M3((byte) MiniFloat.encode(value, PARAMS));
        }

        public static E2M3 fromBits(int bits) {
            return new E2M3((byte) bits);
        }

        @Override
        public float toFloat() {
            return MiniFloat.decodeToFloat(bits & 0x3F, PARAMS);
        }

        @Override
        public double toDouble() {
            return MiniFloat.decodeToDouble(bits & 0x3F, PARAMS);
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
            return (bits & 0x3F) == 0;
        }

        @Override
        public FormatParameters parameters() {
            return PARAMS;
        }

        // Arithmetic operations
        public E2M3 add(E2M3 other) {
            return new E2M3((byte) MiniFloat.add(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E2M3 subtract(E2M3 other) {
            return new E2M3((byte) MiniFloat.subtract(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E2M3 multiply(E2M3 other) {
            return new E2M3((byte) MiniFloat.multiply(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E2M3 divide(E2M3 other) {
            return new E2M3((byte) MiniFloat.divide(bits & 0x3F, other.bits & 0x3F, PARAMS));
        }

        public E2M3 negate() {
            return new E2M3((byte) MiniFloat.negate(bits & 0x3F, PARAMS));
        }

        public E2M3 abs() {
            return new E2M3((byte) MiniFloat.abs(bits & 0x3F, PARAMS));
        }

        public static E2M3 fma(E2M3 a, E2M3 b, E2M3 c) {
            return new E2M3((byte) MiniFloat.fma(a.bits & 0x3F, b.bits & 0x3F, c.bits & 0x3F, PARAMS));
        }

        public static E2M3 min(E2M3 a, E2M3 b) {
            return new E2M3((byte) MiniFloat.min(a.bits & 0x3F, b.bits & 0x3F, PARAMS));
        }

        public static E2M3 max(E2M3 a, E2M3 b) {
            return new E2M3((byte) MiniFloat.max(a.bits & 0x3F, b.bits & 0x3F, PARAMS));
        }

        // Bulk operations
        public static void encodeBulk(float[] source, MemorySegment dest) {
            MiniFloat.encodeBulk(source, dest, PARAMS);
        }

        public static void decodeBulk(MemorySegment source, float[] dest) {
            MiniFloat.decodeBulk(source, dest, PARAMS);
        }

        public static long byteSize(int count) {
            return MiniFloat.byteSize(count, PARAMS);
        }

        // Constants
        public static final E2M3 ZERO = new E2M3((byte) 0);
        public static final E2M3 ONE = fromFloat(1.0f);
        public static final E2M3 NEGATIVE_ONE = fromFloat(-1.0f);
        // Note: No NaN in finite-only format
        public static final E2M3 MAX_VALUE = fromFloat((float) PARAMS.maxValue());
        public static final E2M3 MIN_NORMAL = fromFloat((float) PARAMS.minNormal());

        @Override
        public String toString() {
            return "E2M3(" + toFloat() + ")";
        }
    }
}
