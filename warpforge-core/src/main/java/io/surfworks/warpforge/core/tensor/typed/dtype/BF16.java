package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type for bfloat16 (brain floating point).
 *
 * <p>A 16-bit format with the same exponent range as F32 but reduced mantissa.
 * Developed by Google Brain for ML training. Key properties:
 * <ul>
 *   <li>Same dynamic range as F32 (no overflow issues in typical training)</li>
 *   <li>Lower precision than F16 (7 bits vs 10 bits mantissa)</li>
 *   <li>Trivial conversion to/from F32 (truncate/extend mantissa)</li>
 *   <li>Native support on TPUs, NVIDIA Ampere+, AMD CDNA</li>
 * </ul>
 *
 * <p>Format: 1 sign bit, 8 exponent bits, 7 mantissa bits.
 */
public record BF16() implements DTypeTag {

    /**
     * Singleton instance for BF16 dtype.
     */
    public static final BF16 INSTANCE = new BF16();

    @Override
    public ScalarType scalarType() {
        return ScalarType.BF16;
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
        return "BF16";
    }
}
