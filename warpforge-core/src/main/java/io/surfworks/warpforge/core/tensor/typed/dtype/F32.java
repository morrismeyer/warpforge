package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type for 32-bit floating point (float32).
 *
 * <p>This is the most commonly used dtype for neural network weights
 * and activations. Provides good balance of precision and performance.
 *
 * <p>IEEE 754 single precision: 1 sign bit, 8 exponent bits, 23 mantissa bits.
 */
public record F32() implements DTypeTag {

    /**
     * Singleton instance for F32 dtype.
     */
    public static final F32 INSTANCE = new F32();

    @Override
    public ScalarType scalarType() {
        return ScalarType.F32;
    }

    @Override
    public int byteSize() {
        return 4;
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
        return "F32";
    }
}
