package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type for 64-bit floating point (float64/double).
 *
 * <p>Higher precision than F32, but 2x memory and often slower on GPUs.
 * Typically used for:
 * <ul>
 *   <li>Scientific computing requiring high precision</li>
 *   <li>Loss computation to avoid numerical instability</li>
 *   <li>Gradient accumulation</li>
 * </ul>
 *
 * <p>IEEE 754 double precision: 1 sign bit, 11 exponent bits, 52 mantissa bits.
 */
public record F64() implements DTypeTag {

    /**
     * Singleton instance for F64 dtype.
     */
    public static final F64 INSTANCE = new F64();

    @Override
    public ScalarType scalarType() {
        return ScalarType.F64;
    }

    @Override
    public int byteSize() {
        return 8;
    }

    @Override
    public boolean isFloating() {
        return true;
    }

    @Override
    public boolean isInteger() {
        return false;
    }

    @Override
    public String toString() {
        return "F64";
    }
}
