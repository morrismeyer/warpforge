package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type for 16-bit floating point (float16/half precision).
 *
 * <p>Half the memory of F32 with reduced precision. Commonly used for:
 * <ul>
 *   <li>Mixed precision training (forward pass in F16, gradients in F32)</li>
 *   <li>Inference on memory-constrained devices</li>
 *   <li>Tensor cores on NVIDIA GPUs</li>
 * </ul>
 *
 * <p>IEEE 754 half precision: 1 sign bit, 5 exponent bits, 10 mantissa bits.
 *
 * <p>Note: F16 has limited dynamic range. Values outside approximately
 * [-65504, 65504] overflow, and values smaller than ~6e-8 underflow to zero.
 */
public record F16() implements DTypeTag {

    /**
     * Singleton instance for F16 dtype.
     */
    public static final F16 INSTANCE = new F16();

    @Override
    public ScalarType scalarType() {
        return ScalarType.F16;
    }

    @Override
    public int byteSize() {
        return 2;
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
        return "F16";
    }
}
