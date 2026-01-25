package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe matrix operations with compile-time checking.
 *
 * <p>Matrix operations enforce dtype and device compatibility at compile time.
 * Inner dimension compatibility for matmul is checked at runtime since Java's
 * type system cannot encode dimension values.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(new Matrix(32, 768), F32.INSTANCE, Cpu.INSTANCE);
 * TypedTensor<Matrix, F32, Cpu> b = TypedTensor.zeros(new Matrix(768, 512), F32.INSTANCE, Cpu.INSTANCE);
 *
 * // Compile-time type safety:
 * TypedTensor<Matrix, F32, Cpu> c = MatrixOps.matmul(a, b);  // OK: same dtype and device
 *
 * // These would NOT compile:
 * // MatrixOps.matmul(a, TypedTensor.zeros(new Matrix(768, 512), F64.INSTANCE, Cpu.INSTANCE));  // F64 != F32
 * // MatrixOps.matmul(a, TypedTensor.zeros(new Matrix(768, 512), F32.INSTANCE, Nvidia.DEFAULT)); // device mismatch
 *
 * // Runtime dimension check (inner dims must match):
 * // MatrixOps.matmul(a, TypedTensor.zeros(new Matrix(100, 512), F32.INSTANCE, Cpu.INSTANCE));
 * // throws IllegalArgumentException: inner dimensions don't match (768 vs 100)
 * }</pre>
 */
public final class MatrixOps {

    private MatrixOps() {
        // Utility class
    }

    // ==================== Matrix Multiplication ====================

    /**
     * Matrix multiplication: C = A @ B.
     *
     * <p>For A[M, K] and B[K, N], produces C[M, N].
     * Inner dimensions (K) must match - this is checked at runtime.
     *
     * @param a left matrix [M, K]
     * @param b right matrix [K, N]
     * @param <D> the dtype type (must match for both operands)
     * @param <V> the device type (must match for both operands)
     * @return result matrix [M, N]
     * @throws IllegalArgumentException if inner dimensions don't match
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> matmul(TypedTensor<Matrix, D, V> a, TypedTensor<Matrix, D, V> b) {
        int[] shapeA = a.dimensions();
        int[] shapeB = b.dimensions();

        int M = shapeA[0];
        int K1 = shapeA[1];
        int K2 = shapeB[0];
        int N = shapeB[1];

        // Runtime check for inner dimension compatibility
        if (K1 != K2) {
            throw new IllegalArgumentException(
                    "Matrix inner dimensions don't match: [" + M + ", " + K1 + "] @ [" + K2 + ", " + N + "]. " +
                    "Left matrix has " + K1 + " columns, right matrix has " + K2 + " rows.");
        }

        Matrix resultShape = new Matrix(M, N);
        Tensor result = Tensor.zeros(a.underlying().dtype(), M, N);

        ScalarType dtype = a.underlying().dtype();
        if (dtype == ScalarType.F32) {
            matmulF32(a.underlying().data(), b.underlying().data(), result.data(), M, K1, N);
        } else if (dtype == ScalarType.F64) {
            matmulF64(a.underlying().data(), b.underlying().data(), result.data(), M, K1, N);
        } else {
            throw new UnsupportedOperationException("matmul not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, resultShape, a.dtypeType(), a.deviceType());
    }

    /**
     * Batched matrix multiplication for 3D tensors.
     *
     * <p>For A[B, M, K] and B[B, K, N], produces C[B, M, N].
     * Batch dimension and inner dimensions must match.
     *
     * <p>Note: This method accepts Matrix-typed tensors but operates on their
     * 3D equivalent interpretation. Use reshape to convert Rank3 to batched matrices.
     *
     * @param a left batch of matrices
     * @param b right batch of matrices
     * @param batchSize the batch dimension
     * @param <D> the dtype type
     * @param <V> the device type
     * @return result batch of matrices
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> batchedMatmul(
            TypedTensor<Matrix, D, V> a,
            TypedTensor<Matrix, D, V> b,
            int batchSize) {
        // For simplicity, this is a placeholder that shows the type signature.
        // Full batched matmul implementation would require Rank3 tensor support.
        throw new UnsupportedOperationException(
                "Batched matmul requires Rank3 tensor support. Use reshape and loop for now.");
    }

    // ==================== Matrix-Vector Operations ====================

    /**
     * Matrix-vector multiplication: y = A @ x.
     *
     * <p>For A[M, N] and x[N], produces y[M].
     *
     * @param matrix the matrix [M, N]
     * @param vector the vector [N]
     * @param <D> the dtype type
     * @param <V> the device type
     * @return result vector [M]
     * @throws IllegalArgumentException if dimensions don't match
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> matvec(TypedTensor<Matrix, D, V> matrix, TypedTensor<Vector, D, V> vector) {
        int[] matShape = matrix.dimensions();
        int[] vecShape = vector.dimensions();

        int M = matShape[0];
        int N = matShape[1];
        int vecLen = vecShape[0];

        if (N != vecLen) {
            throw new IllegalArgumentException(
                    "Matrix columns (" + N + ") must match vector length (" + vecLen + ")");
        }

        Vector resultShape = new Vector(M);
        Tensor result = Tensor.zeros(matrix.underlying().dtype(), M);

        ScalarType dtype = matrix.underlying().dtype();
        if (dtype == ScalarType.F32) {
            matvecF32(matrix.underlying().data(), vector.underlying().data(), result.data(), M, N);
        } else if (dtype == ScalarType.F64) {
            matvecF64(matrix.underlying().data(), vector.underlying().data(), result.data(), M, N);
        } else {
            throw new UnsupportedOperationException("matvec not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, resultShape, matrix.dtypeType(), matrix.deviceType());
    }

    /**
     * Vector-matrix multiplication: y = x @ A.
     *
     * <p>For x[M] and A[M, N], produces y[N].
     * Equivalent to transpose(A) @ x.
     *
     * @param vector the row vector [M]
     * @param matrix the matrix [M, N]
     * @param <D> the dtype type
     * @param <V> the device type
     * @return result vector [N]
     * @throws IllegalArgumentException if dimensions don't match
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> vecmat(TypedTensor<Vector, D, V> vector, TypedTensor<Matrix, D, V> matrix) {
        int[] vecShape = vector.dimensions();
        int[] matShape = matrix.dimensions();

        int vecLen = vecShape[0];
        int M = matShape[0];
        int N = matShape[1];

        if (vecLen != M) {
            throw new IllegalArgumentException(
                    "Vector length (" + vecLen + ") must match matrix rows (" + M + ")");
        }

        Vector resultShape = new Vector(N);
        Tensor result = Tensor.zeros(matrix.underlying().dtype(), N);

        ScalarType dtype = matrix.underlying().dtype();
        if (dtype == ScalarType.F32) {
            vecmatF32(vector.underlying().data(), matrix.underlying().data(), result.data(), M, N);
        } else if (dtype == ScalarType.F64) {
            vecmatF64(vector.underlying().data(), matrix.underlying().data(), result.data(), M, N);
        } else {
            throw new UnsupportedOperationException("vecmat not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, resultShape, matrix.dtypeType(), matrix.deviceType());
    }

    /**
     * Vector dot product: c = a · b.
     *
     * @param a first vector
     * @param b second vector
     * @param <V> the device type
     * @return the scalar dot product
     * @throws IllegalArgumentException if vector lengths don't match
     */
    public static <V extends DeviceTag>
    float dot(TypedTensor<Vector, F32, V> a, TypedTensor<Vector, F32, V> b) {
        int lenA = a.dimensions()[0];
        int lenB = b.dimensions()[0];

        if (lenA != lenB) {
            throw new IllegalArgumentException(
                    "Vector lengths must match for dot product: " + lenA + " vs " + lenB);
        }

        MemorySegment dataA = a.underlying().data();
        MemorySegment dataB = b.underlying().data();
        float sum = 0.0f;

        for (int i = 0; i < lenA; i++) {
            float va = dataA.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = dataB.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            sum += va * vb;
        }

        return sum;
    }

    /**
     * Vector dot product (double precision).
     */
    public static <V extends DeviceTag>
    double dotF64(TypedTensor<Vector, F64, V> a, TypedTensor<Vector, F64, V> b) {
        int lenA = a.dimensions()[0];
        int lenB = b.dimensions()[0];

        if (lenA != lenB) {
            throw new IllegalArgumentException(
                    "Vector lengths must match for dot product: " + lenA + " vs " + lenB);
        }

        MemorySegment dataA = a.underlying().data();
        MemorySegment dataB = b.underlying().data();
        double sum = 0.0;

        for (int i = 0; i < lenA; i++) {
            double va = dataA.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double vb = dataB.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            sum += va * vb;
        }

        return sum;
    }

    // ==================== Matrix Transformations ====================

    /**
     * Transposes a matrix: B = A^T.
     *
     * <p>For A[M, N], produces B[N, M].
     *
     * @param a the input matrix
     * @param <D> the dtype type
     * @param <V> the device type
     * @return the transposed matrix
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> transpose(TypedTensor<Matrix, D, V> a) {
        int[] shape = a.dimensions();
        int M = shape[0];
        int N = shape[1];

        Matrix resultShape = new Matrix(N, M);
        Tensor result = Tensor.zeros(a.underlying().dtype(), N, M);

        ScalarType dtype = a.underlying().dtype();
        if (dtype == ScalarType.F32) {
            transposeF32(a.underlying().data(), result.data(), M, N);
        } else if (dtype == ScalarType.F64) {
            transposeF64(a.underlying().data(), result.data(), M, N);
        } else {
            throw new UnsupportedOperationException("transpose not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, resultShape, a.dtypeType(), a.deviceType());
    }

    /**
     * Outer product of two vectors: C = a ⊗ b.
     *
     * <p>For a[M] and b[N], produces C[M, N] where C[i,j] = a[i] * b[j].
     *
     * @param a the column vector
     * @param b the row vector
     * @param <D> the dtype type
     * @param <V> the device type
     * @return the outer product matrix
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> outer(TypedTensor<Vector, D, V> a, TypedTensor<Vector, D, V> b) {
        int M = a.dimensions()[0];
        int N = b.dimensions()[0];

        Matrix resultShape = new Matrix(M, N);
        Tensor result = Tensor.zeros(a.underlying().dtype(), M, N);

        ScalarType dtype = a.underlying().dtype();
        if (dtype == ScalarType.F32) {
            outerF32(a.underlying().data(), b.underlying().data(), result.data(), M, N);
        } else if (dtype == ScalarType.F64) {
            outerF64(a.underlying().data(), b.underlying().data(), result.data(), M, N);
        } else {
            throw new UnsupportedOperationException("outer not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, resultShape, a.dtypeType(), a.deviceType());
    }

    // ==================== Internal Implementation ====================

    // F32 implementations
    private static void matmulF32(MemorySegment a, MemorySegment b, MemorySegment c, int M, int K, int N) {
        // Basic O(M*K*N) matrix multiplication
        // For production, this should use BLAS or vectorized operations
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float aVal = a.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * K + k);
                    float bVal = b.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * N + j);
                    sum += aVal * bVal;
                }
                c.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j, sum);
            }
        }
    }

    private static void matvecF32(MemorySegment mat, MemorySegment vec, MemorySegment out, int M, int N) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float mVal = mat.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                float vVal = vec.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                sum += mVal * vVal;
            }
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, sum);
        }
    }

    private static void vecmatF32(MemorySegment vec, MemorySegment mat, MemorySegment out, int M, int N) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                float vVal = vec.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float mVal = mat.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                sum += vVal * mVal;
            }
            out.setAtIndex(ValueLayout.JAVA_FLOAT, j, sum);
        }
    }

    private static void transposeF32(MemorySegment in, MemorySegment out, int M, int N) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float val = in.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                out.setAtIndex(ValueLayout.JAVA_FLOAT, (long) j * M + i, val);
            }
        }
    }

    private static void outerF32(MemorySegment a, MemorySegment b, MemorySegment out, int M, int N) {
        for (int i = 0; i < M; i++) {
            float aVal = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            for (int j = 0; j < N; j++) {
                float bVal = b.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                out.setAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j, aVal * bVal);
            }
        }
    }

    // F64 implementations
    private static void matmulF64(MemorySegment a, MemorySegment b, MemorySegment c, int M, int K, int N) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    double aVal = a.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * K + k);
                    double bVal = b.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) k * N + j);
                    sum += aVal * bVal;
                }
                c.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * N + j, sum);
            }
        }
    }

    private static void matvecF64(MemorySegment mat, MemorySegment vec, MemorySegment out, int M, int N) {
        for (int i = 0; i < M; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                double mVal = mat.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * N + j);
                double vVal = vec.getAtIndex(ValueLayout.JAVA_DOUBLE, j);
                sum += mVal * vVal;
            }
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, sum);
        }
    }

    private static void vecmatF64(MemorySegment vec, MemorySegment mat, MemorySegment out, int M, int N) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int i = 0; i < M; i++) {
                double vVal = vec.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                double mVal = mat.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * N + j);
                sum += vVal * mVal;
            }
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, j, sum);
        }
    }

    private static void transposeF64(MemorySegment in, MemorySegment out, int M, int N) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double val = in.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * N + j);
                out.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) j * M + i, val);
            }
        }
    }

    private static void outerF64(MemorySegment a, MemorySegment b, MemorySegment out, int M, int N) {
        for (int i = 0; i < M; i++) {
            double aVal = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            for (int j = 0; j < N; j++) {
                double bVal = b.getAtIndex(ValueLayout.JAVA_DOUBLE, j);
                out.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) i * N + j, aVal * bVal);
            }
        }
    }
}
