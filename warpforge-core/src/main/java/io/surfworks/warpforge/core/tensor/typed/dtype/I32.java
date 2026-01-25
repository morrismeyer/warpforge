package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type for 32-bit signed integer (int32).
 *
 * <p>Commonly used for:
 * <ul>
 *   <li>Indices and index tensors</li>
 *   <li>Class labels</li>
 *   <li>Token IDs in NLP</li>
 *   <li>Sparse tensor coordinates</li>
 * </ul>
 *
 * <p>Range: [-2,147,483,648, 2,147,483,647]
 */
public record I32() implements DTypeTag {

    /**
     * Singleton instance for I32 dtype.
     */
    public static final I32 INSTANCE = new I32();

    @Override
    public ScalarType scalarType() {
        return ScalarType.I32;
    }

    @Override
    public int byteSize() {
        return 4;
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
        return "I32";
    }
}
