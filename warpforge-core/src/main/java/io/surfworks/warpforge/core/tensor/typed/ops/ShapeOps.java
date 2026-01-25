package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Dynamic;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe shape manipulation operations for tensors.
 *
 * <p>These operations change tensor shapes without modifying the underlying data
 * (where possible) or with minimal data movement. They are fundamental building
 * blocks for transformer models which frequently reshape tensors between
 * different views (e.g., [batch, seq, hidden] to [batch, heads, seq, head_dim]).
 *
 * <p>Example:
 * <pre>{@code
 * // Reshape for multi-head attention
 * TypedTensor<Rank3, F32, Cpu> input = ...;  // [batch, seq, hidden]
 * TypedTensor<Rank4, F32, Cpu> multiHead = ShapeOps.reshape(
 *     input, new Rank4(batch, seq, numHeads, headDim));
 *
 * // Permute to [batch, heads, seq, head_dim]
 * TypedTensor<Rank4, F32, Cpu> permuted = ShapeOps.permute(multiHead, 0, 2, 1, 3);
 * }</pre>
 */
public final class ShapeOps {

    private ShapeOps() {
        // Utility class
    }

    // ==================== Reshape Operations ====================

    /**
     * Reshapes a tensor to a new shape with the same total element count.
     *
     * <p>This is a zero-copy operation when possible - it returns a view with
     * different shape metadata pointing to the same underlying data.
     *
     * @param input the input tensor
     * @param newShape the target shape (must have same element count)
     * @param <S1> input shape type
     * @param <S2> output shape type
     * @param <D> dtype type
     * @param <V> device type
     * @return reshaped tensor
     * @throws IllegalArgumentException if element counts don't match
     */
    public static <S1 extends Shape, S2 extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S2, D, V> reshape(TypedTensor<S1, D, V> input, S2 newShape) {
        long inputCount = input.elementCount();
        long outputCount = newShape.elementCount();

        if (inputCount != outputCount) {
            throw new IllegalArgumentException(
                    "Cannot reshape: element counts differ. Input has " + inputCount +
                    " elements, target shape has " + outputCount + " elements.");
        }

        // Create a new tensor with the same data but different shape
        Tensor result = Tensor.zeros(input.underlying().dtype(), newShape.dimensions());
        copyData(input.underlying().data(), result.data(), inputCount, input.underlying().dtype());

        return TypedTensor.from(result, newShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Reshapes a tensor to a Dynamic shape (for interoperability).
     *
     * @param input the input tensor
     * @param newDimensions the target dimensions
     * @param <S> input shape type
     * @param <D> dtype type
     * @param <V> device type
     * @return reshaped tensor with Dynamic shape
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Dynamic, D, V> reshapeToDynamic(TypedTensor<S, D, V> input, int... newDimensions) {
        Dynamic newShape = new Dynamic(newDimensions);
        return reshape(input, newShape);
    }

    /**
     * Flattens a tensor to a 1D vector.
     *
     * @param input the input tensor
     * @param <S> input shape type
     * @param <D> dtype type
     * @param <V> device type
     * @return flattened vector
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> flatten(TypedTensor<S, D, V> input) {
        int totalElements = (int) input.elementCount();
        return reshape(input, new Vector(totalElements));
    }

    // ==================== Transpose / Permute Operations ====================

    /**
     * Transposes a matrix (2D tensor).
     *
     * <p>For input [M, N], produces output [N, M].
     *
     * @param input the input matrix
     * @param <D> dtype type
     * @param <V> device type
     * @return transposed matrix
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> transpose(TypedTensor<Matrix, D, V> input) {
        // Delegate to MatrixOps which has an optimized implementation
        return MatrixOps.transpose(input);
    }

    /**
     * Permutes the dimensions of a Rank3 tensor.
     *
     * @param input the input tensor [d0, d1, d2]
     * @param perm permutation indices (e.g., {1, 0, 2} swaps first two dims)
     * @param <D> dtype type
     * @param <V> device type
     * @return permuted tensor
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> permute(TypedTensor<Rank3, D, V> input, int... perm) {
        if (perm.length != 3) {
            throw new IllegalArgumentException("Permutation must have 3 elements for Rank3 tensor");
        }
        validatePermutation(perm, 3);

        int[] dims = input.dimensions();
        int[] newDims = new int[]{dims[perm[0]], dims[perm[1]], dims[perm[2]]};
        Rank3 newShape = new Rank3(newDims[0], newDims[1], newDims[2]);

        Tensor result = Tensor.zeros(input.underlying().dtype(), newDims);

        ScalarType dtype = input.underlying().dtype();
        if (dtype == ScalarType.F32) {
            permuteRank3F32(input.underlying().data(), result.data(), dims, perm);
        } else if (dtype == ScalarType.F64) {
            permuteRank3F64(input.underlying().data(), result.data(), dims, perm);
        } else {
            throw new UnsupportedOperationException("permute not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, newShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Permutes the dimensions of a Rank4 tensor.
     *
     * <p>Common use case: [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
     * using permutation {0, 2, 1, 3}.
     *
     * <p>Note: Named permuteRank4 to avoid type erasure clash with Rank3 version.
     *
     * @param input the input tensor [d0, d1, d2, d3]
     * @param perm permutation indices
     * @param <D> dtype type
     * @param <V> device type
     * @return permuted tensor
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank4, D, V> permuteRank4(TypedTensor<Rank4, D, V> input, int... perm) {
        if (perm.length != 4) {
            throw new IllegalArgumentException("Permutation must have 4 elements for Rank4 tensor");
        }
        validatePermutation(perm, 4);

        int[] dims = input.dimensions();
        int[] newDims = new int[]{dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
        Rank4 newShape = new Rank4(newDims[0], newDims[1], newDims[2], newDims[3]);

        Tensor result = Tensor.zeros(input.underlying().dtype(), newDims);

        ScalarType dtype = input.underlying().dtype();
        if (dtype == ScalarType.F32) {
            permuteRank4F32(input.underlying().data(), result.data(), dims, perm);
        } else if (dtype == ScalarType.F64) {
            permuteRank4F64(input.underlying().data(), result.data(), dims, perm);
        } else {
            throw new UnsupportedOperationException("permute not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, newShape, input.dtypeType(), input.deviceType());
    }

    // ==================== Squeeze / Unsqueeze Operations ====================

    /**
     * Removes a dimension of size 1 from a Matrix, producing a Vector.
     *
     * @param input the input matrix (one dimension must be 1)
     * @param axis which axis to squeeze (0 or 1)
     * @param <D> dtype type
     * @param <V> device type
     * @return squeezed vector
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> squeeze(TypedTensor<Matrix, D, V> input, int axis) {
        int[] dims = input.dimensions();
        if (axis < 0 || axis > 1) {
            throw new IllegalArgumentException("Axis must be 0 or 1 for Matrix, got: " + axis);
        }
        if (dims[axis] != 1) {
            throw new IllegalArgumentException(
                    "Can only squeeze axis with size 1, but axis " + axis + " has size " + dims[axis]);
        }

        int newDim = axis == 0 ? dims[1] : dims[0];
        return reshape(input, new Vector(newDim));
    }

    /**
     * Adds a dimension of size 1 to a Vector, producing a Matrix.
     *
     * @param input the input vector
     * @param axis where to add the dimension (0 for row vector, 1 for column vector)
     * @param <D> dtype type
     * @param <V> device type
     * @return unsqueezed matrix
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> unsqueeze(TypedTensor<Vector, D, V> input, int axis) {
        int[] dims = input.dimensions();
        if (axis < 0 || axis > 1) {
            throw new IllegalArgumentException("Axis must be 0 or 1 for Vector->Matrix, got: " + axis);
        }

        Matrix newShape = axis == 0
                ? new Matrix(1, dims[0])   // Row vector [1, N]
                : new Matrix(dims[0], 1);  // Column vector [N, 1]

        return reshape(input, newShape);
    }

    /**
     * Unsqueezes a Matrix to Rank3 by adding a dimension.
     *
     * @param input the input matrix [M, N]
     * @param axis where to add dimension (0, 1, or 2)
     * @param <D> dtype type
     * @param <V> device type
     * @return Rank3 tensor
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> unsqueezeToRank3(TypedTensor<Matrix, D, V> input, int axis) {
        int[] dims = input.dimensions();
        if (axis < 0 || axis > 2) {
            throw new IllegalArgumentException("Axis must be 0, 1, or 2 for Matrix->Rank3, got: " + axis);
        }

        Rank3 newShape = switch (axis) {
            case 0 -> new Rank3(1, dims[0], dims[1]);
            case 1 -> new Rank3(dims[0], 1, dims[1]);
            case 2 -> new Rank3(dims[0], dims[1], 1);
            default -> throw new IllegalStateException("Unexpected axis: " + axis);
        };

        return reshape(input, newShape);
    }

    /**
     * Unsqueezes a Rank3 tensor to Rank4 by adding a dimension.
     *
     * @param input the input Rank3 tensor
     * @param axis where to add dimension (0, 1, 2, or 3)
     * @param <D> dtype type
     * @param <V> device type
     * @return Rank4 tensor
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank4, D, V> unsqueezeToRank4(TypedTensor<Rank3, D, V> input, int axis) {
        int[] dims = input.dimensions();
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be 0, 1, 2, or 3 for Rank3->Rank4, got: " + axis);
        }

        Rank4 newShape = switch (axis) {
            case 0 -> new Rank4(1, dims[0], dims[1], dims[2]);
            case 1 -> new Rank4(dims[0], 1, dims[1], dims[2]);
            case 2 -> new Rank4(dims[0], dims[1], 1, dims[2]);
            case 3 -> new Rank4(dims[0], dims[1], dims[2], 1);
            default -> throw new IllegalStateException("Unexpected axis: " + axis);
        };

        return reshape(input, newShape);
    }

    // ==================== Broadcast Operations ====================

    /**
     * Broadcasts a Vector to a Matrix by repeating along a new dimension.
     *
     * @param input the input vector [N]
     * @param repeatCount how many times to repeat
     * @param axis whether to repeat along rows (0) or columns (1)
     * @param <D> dtype type
     * @param <V> device type
     * @return broadcasted matrix
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> broadcast(TypedTensor<Vector, D, V> input, int repeatCount, int axis) {
        int[] dims = input.dimensions();
        int vecLen = dims[0];

        Matrix newShape = axis == 0
                ? new Matrix(repeatCount, vecLen)  // Repeat as rows
                : new Matrix(vecLen, repeatCount); // Repeat as columns

        Tensor result = Tensor.zeros(input.underlying().dtype(), newShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        if (dtype == ScalarType.F32) {
            broadcastVectorF32(input.underlying().data(), result.data(), vecLen, repeatCount, axis);
        } else if (dtype == ScalarType.F64) {
            broadcastVectorF64(input.underlying().data(), result.data(), vecLen, repeatCount, axis);
        } else {
            throw new UnsupportedOperationException("broadcast not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, newShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Expands a tensor by repeating it along specified dimensions.
     * This is useful for broadcasting in elementwise operations.
     *
     * @param input the input tensor
     * @param targetShape the target shape to expand to
     * @param <D> dtype type
     * @param <V> device type
     * @return expanded tensor
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> expandToRank3(TypedTensor<Matrix, D, V> input, Rank3 targetShape) {
        int[] inputDims = input.dimensions();
        int[] targetDims = targetShape.dimensions();

        // Validate that input can be broadcast to target
        // For [M, N] -> [B, M, N], the last two dims must match
        if (inputDims[0] != targetDims[1] || inputDims[1] != targetDims[2]) {
            throw new IllegalArgumentException(
                    "Cannot expand Matrix " + Arrays.toString(inputDims) +
                    " to Rank3 " + Arrays.toString(targetDims) +
                    ". Last two dimensions must match.");
        }

        int batchSize = targetDims[0];
        Tensor result = Tensor.zeros(input.underlying().dtype(), targetDims);

        ScalarType dtype = input.underlying().dtype();
        long matrixSize = (long) inputDims[0] * inputDims[1];

        // Copy the matrix batchSize times
        MemorySegment src = input.underlying().data();
        MemorySegment dst = result.data();

        if (dtype == ScalarType.F32) {
            for (int b = 0; b < batchSize; b++) {
                long offset = b * matrixSize;
                for (long i = 0; i < matrixSize; i++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, val);
                }
            }
        } else if (dtype == ScalarType.F64) {
            for (int b = 0; b < batchSize; b++) {
                long offset = b * matrixSize;
                for (long i = 0; i < matrixSize; i++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, offset + i, val);
                }
            }
        } else {
            throw new UnsupportedOperationException("expand not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, targetShape, input.dtypeType(), input.deviceType());
    }

    // ==================== Internal Implementation ====================

    private static void validatePermutation(int[] perm, int rank) {
        boolean[] seen = new boolean[rank];
        for (int p : perm) {
            if (p < 0 || p >= rank) {
                throw new IllegalArgumentException(
                        "Permutation index " + p + " out of range for rank " + rank);
            }
            if (seen[p]) {
                throw new IllegalArgumentException(
                        "Duplicate index " + p + " in permutation");
            }
            seen[p] = true;
        }
    }

    private static void copyData(MemorySegment src, MemorySegment dst, long count, ScalarType dtype) {
        if (dtype == ScalarType.F32) {
            for (long i = 0; i < count; i++) {
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, src.getAtIndex(ValueLayout.JAVA_FLOAT, i));
            }
        } else if (dtype == ScalarType.F64) {
            for (long i = 0; i < count; i++) {
                dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, src.getAtIndex(ValueLayout.JAVA_DOUBLE, i));
            }
        } else {
            throw new UnsupportedOperationException("copyData not implemented for dtype: " + dtype);
        }
    }

    private static void permuteRank3F32(MemorySegment src, MemorySegment dst, int[] dims, int[] perm) {
        int d0 = dims[0], d1 = dims[1], d2 = dims[2];
        int[] newDims = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
        int[] invPerm = invertPermutation(perm);

        int nd0 = newDims[0], nd1 = newDims[1], nd2 = newDims[2];
        for (int o0 = 0; o0 < nd0; o0++) {
            for (int o1 = 0; o1 < nd1; o1++) {
                for (int o2 = 0; o2 < nd2; o2++) {
                    int[] outCoords = {o0, o1, o2};
                    int si0 = outCoords[invPerm[0]];
                    int si1 = outCoords[invPerm[1]];
                    int si2 = outCoords[invPerm[2]];

                    long srcIdx = (long) si0 * d1 * d2 + (long) si1 * d2 + si2;
                    long dstIdx = (long) o0 * nd1 * nd2 + (long) o1 * nd2 + o2;

                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcIdx);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstIdx, val);
                }
            }
        }
    }

    private static void permuteRank3F64(MemorySegment src, MemorySegment dst, int[] dims, int[] perm) {
        int d0 = dims[0], d1 = dims[1], d2 = dims[2];
        int[] newDims = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
        int[] invPerm = invertPermutation(perm);

        int nd0 = newDims[0], nd1 = newDims[1], nd2 = newDims[2];
        for (int o0 = 0; o0 < nd0; o0++) {
            for (int o1 = 0; o1 < nd1; o1++) {
                for (int o2 = 0; o2 < nd2; o2++) {
                    int[] outCoords = {o0, o1, o2};
                    int si0 = outCoords[invPerm[0]];
                    int si1 = outCoords[invPerm[1]];
                    int si2 = outCoords[invPerm[2]];

                    long srcIdx = (long) si0 * d1 * d2 + (long) si1 * d2 + si2;
                    long dstIdx = (long) o0 * nd1 * nd2 + (long) o1 * nd2 + o2;

                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcIdx);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstIdx, val);
                }
            }
        }
    }

    private static void permuteRank4F32(MemorySegment src, MemorySegment dst, int[] dims, int[] perm) {
        int d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
        int[] newDims = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
        int[] invPerm = invertPermutation(perm);

        int nd0 = newDims[0], nd1 = newDims[1], nd2 = newDims[2], nd3 = newDims[3];
        for (int o0 = 0; o0 < nd0; o0++) {
            for (int o1 = 0; o1 < nd1; o1++) {
                for (int o2 = 0; o2 < nd2; o2++) {
                    for (int o3 = 0; o3 < nd3; o3++) {
                        int[] outCoords = {o0, o1, o2, o3};
                        int si0 = outCoords[invPerm[0]];
                        int si1 = outCoords[invPerm[1]];
                        int si2 = outCoords[invPerm[2]];
                        int si3 = outCoords[invPerm[3]];

                        long srcIdx = (long) si0 * d1 * d2 * d3 + (long) si1 * d2 * d3 + (long) si2 * d3 + si3;
                        long dstIdx = (long) o0 * nd1 * nd2 * nd3 + (long) o1 * nd2 * nd3 + (long) o2 * nd3 + o3;

                        float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcIdx);
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstIdx, val);
                    }
                }
            }
        }
    }

    private static void permuteRank4F64(MemorySegment src, MemorySegment dst, int[] dims, int[] perm) {
        int d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
        int[] newDims = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
        int[] invPerm = invertPermutation(perm);

        int nd0 = newDims[0], nd1 = newDims[1], nd2 = newDims[2], nd3 = newDims[3];
        for (int o0 = 0; o0 < nd0; o0++) {
            for (int o1 = 0; o1 < nd1; o1++) {
                for (int o2 = 0; o2 < nd2; o2++) {
                    for (int o3 = 0; o3 < nd3; o3++) {
                        int[] outCoords = {o0, o1, o2, o3};
                        int si0 = outCoords[invPerm[0]];
                        int si1 = outCoords[invPerm[1]];
                        int si2 = outCoords[invPerm[2]];
                        int si3 = outCoords[invPerm[3]];

                        long srcIdx = (long) si0 * d1 * d2 * d3 + (long) si1 * d2 * d3 + (long) si2 * d3 + si3;
                        long dstIdx = (long) o0 * nd1 * nd2 * nd3 + (long) o1 * nd2 * nd3 + (long) o2 * nd3 + o3;

                        double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcIdx);
                        dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstIdx, val);
                    }
                }
            }
        }
    }

    private static int[] invertPermutation(int[] perm) {
        int[] inv = new int[perm.length];
        for (int i = 0; i < perm.length; i++) {
            inv[perm[i]] = i;
        }
        return inv;
    }

    private static void broadcastVectorF32(MemorySegment src, MemorySegment dst, int vecLen, int repeatCount, int axis) {
        if (axis == 0) {
            // Repeat as rows: each row is a copy of the vector
            for (int row = 0; row < repeatCount; row++) {
                long rowOffset = (long) row * vecLen;
                for (int col = 0; col < vecLen; col++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, col);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, rowOffset + col, val);
                }
            }
        } else {
            // Repeat as columns: each column is a copy of the vector
            for (int row = 0; row < vecLen; row++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, row);
                long rowOffset = (long) row * repeatCount;
                for (int col = 0; col < repeatCount; col++) {
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, rowOffset + col, val);
                }
            }
        }
    }

    private static void broadcastVectorF64(MemorySegment src, MemorySegment dst, int vecLen, int repeatCount, int axis) {
        if (axis == 0) {
            for (int row = 0; row < repeatCount; row++) {
                long rowOffset = (long) row * vecLen;
                for (int col = 0; col < vecLen; col++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, col);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, rowOffset + col, val);
                }
            }
        } else {
            for (int row = 0; row < vecLen; row++) {
                double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, row);
                long rowOffset = (long) row * repeatCount;
                for (int col = 0; col < repeatCount; col++) {
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, rowOffset + col, val);
                }
            }
        }
    }
}
