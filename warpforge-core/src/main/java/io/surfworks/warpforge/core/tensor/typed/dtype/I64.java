package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type for 64-bit signed integer (int64/long).
 *
 * <p>Commonly used for:
 * <ul>
 *   <li>Large index tensors</li>
 *   <li>Timestamps</li>
 *   <li>Hash values</li>
 *   <li>Large vocabulary token IDs</li>
 * </ul>
 *
 * <p>Range: [-9,223,372,036,854,775,808, 9,223,372,036,854,775,807]
 */
public record I64() implements DTypeTag {

    /**
     * Singleton instance for I64 dtype.
     */
    public static final I64 INSTANCE = new I64();

    @Override
    public ScalarType scalarType() {
        return ScalarType.I64;
    }

    @Override
    public int byteSize() {
        return 8;
    }

    @Override
    public boolean isFloating() {
        return false;
    }

    @Override
    public boolean isInteger() {
        return true;
    }

    @Override
    public String toString() {
        return "I64";
    }
}
