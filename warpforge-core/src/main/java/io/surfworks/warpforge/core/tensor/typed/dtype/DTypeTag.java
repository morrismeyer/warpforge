package io.surfworks.warpforge.core.tensor.typed.dtype;

import io.surfworks.warpforge.core.tensor.ScalarType;

/**
 * Phantom type representing tensor element type at compile time.
 *
 * <p>This sealed interface provides type-level differentiation between
 * tensor dtypes, enabling compile-time type checking for dtype compatibility.
 * Each implementation maps to a corresponding {@link ScalarType} enum value.
 *
 * <p>Example usage:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> floatWeights = ...;
 * TypedTensor<Matrix, F64, Cpu> doubleWeights = ...;
 *
 * // This would NOT compile - dtype mismatch:
 * // MatrixOps.matmul(floatWeights, doubleWeights);
 *
 * // Explicit conversion required:
 * TypedTensor<Matrix, F32, Cpu> converted = doubleWeights.cast(F32.INSTANCE);
 * MatrixOps.matmul(floatWeights, converted);  // OK
 * }</pre>
 *
 * @see F32 for 32-bit floating point
 * @see F64 for 64-bit floating point
 * @see F16 for 16-bit floating point (half precision)
 * @see BF16 for bfloat16 (brain floating point)
 * @see I32 for 32-bit signed integer
 * @see I64 for 64-bit signed integer
 */
public sealed interface DTypeTag permits F32, F64, F16, BF16, I32, I64 {

    /**
     * Returns the corresponding runtime {@link ScalarType}.
     *
     * @return the scalar type enum value
     */
    ScalarType scalarType();

    /**
     * Returns the byte size of one element.
     *
     * @return bytes per element
     */
    int byteSize();

    /**
     * Returns true if this is a floating-point type.
     *
     * @return true for F16, BF16, F32, F64
     */
    boolean isFloating();

    /**
     * Returns true if this is an integer type.
     *
     * @return true for I32, I64
     */
    boolean isInteger();
}
