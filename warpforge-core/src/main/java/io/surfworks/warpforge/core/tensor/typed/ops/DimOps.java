package io.surfworks.warpforge.core.tensor.typed.ops;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dim.Dim;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.shape.DimMatrix;
import io.surfworks.warpforge.core.tensor.typed.shape.DimRank3;
import io.surfworks.warpforge.core.tensor.typed.shape.DimRank4;
import io.surfworks.warpforge.core.tensor.typed.shape.DimVector;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;

/**
 * Dimension-typed operations with compile-time shape checking.
 *
 * <p>These operations use dimension type parameters to enforce shape
 * compatibility at compile time. The inner dimension K in matmul is
 * checked by the Java compiler, not at runtime.
 *
 * <h2>Matmul Example</h2>
 * <pre>{@code
 * interface B extends Dim {}  // Batch
 * interface H extends Dim {}  // Hidden
 * interface V extends Dim {}  // Vocab
 *
 * TypedTensor<DimMatrix<B, H>, F32, Cpu> hidden = ...;   // [batch, hidden]
 * TypedTensor<DimMatrix<H, V>, F32, Cpu> classifier = ...;   // [hidden, vocab]
 *
 * // Compile-time safe: H matches in both
 * var logits = DimOps.matmul(hidden, classifier);  // DimMatrix<B, V>
 *
 * // COMPILE ERROR: B vs H mismatch in inner dimension
 * // DimOps.matmul(hidden, TypedTensor<DimMatrix<B, V>, ...>);
 * }</pre>
 *
 * <h2>Attention Example</h2>
 * <pre>{@code
 * interface Batch extends Dim {}
 * interface NumHeads extends Dim {}
 * interface SeqLen extends Dim {}
 * interface HeadDim extends Dim {}
 *
 * TypedTensor<DimRank4<Batch, NumHeads, SeqLen, HeadDim>, F32, Cpu> Q = ...;
 * TypedTensor<DimRank4<Batch, NumHeads, HeadDim, SeqLen>, F32, Cpu> K_T = ...;
 *
 * // Q @ K^T produces attention scores with HeadDim matched at compile time
 * var scores = DimOps.batchedMatmulRank4(Q, K_T);  // [Batch, NumHeads, SeqLen, SeqLen]
 * }</pre>
 *
 * <h2>Design Principle</h2>
 * <p>DimOps provides compile-time shape checking while delegating actual
 * computation to existing implementations. Runtime dimension checks still
 * happen as defense-in-depth, but the primary goal is catching shape
 * mismatches at compile time when consistent dimension markers are used.
 *
 * @see io.surfworks.warpforge.core.tensor.typed.dim.Dim
 * @see io.surfworks.warpforge.core.tensor.typed.dim.Semantic
 * @see io.surfworks.warpforge.core.tensor.typed.dim.Numeric
 */
public final class DimOps {

    private DimOps() {
        // Utility class - no instantiation
    }

    // ==================== Matrix Multiplication ====================

    /**
     * Matrix multiplication with compile-time dimension checking.
     *
     * <p>For A[M, K] and B[K, N], produces C[M, N].
     * The K dimension MUST be the same type parameter in both inputs,
     * ensuring inner dimension compatibility at compile time.
     *
     * @param a left matrix with shape [M, K]
     * @param b right matrix with shape [K, N]
     * @param <M> row dimension of A (and result)
     * @param <K> inner dimension (must match between A's cols and B's rows)
     * @param <N> column dimension of B (and result)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result matrix with shape [M, N]
     */
    public static <M extends Dim, K extends Dim, N extends Dim, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<DimMatrix<M, N>, D, V> matmul(
            TypedTensor<DimMatrix<M, K>, D, V> a,
            TypedTensor<DimMatrix<K, N>, D, V> b) {

        // Delegate to existing MatrixOps implementation
        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> aMatrix =
                TypedTensor.from(
                        a.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Matrix(
                                a.shapeType().rows(), a.shapeType().cols()),
                        a.dtypeType(),
                        a.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> bMatrix =
                TypedTensor.from(
                        b.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Matrix(
                                b.shapeType().rows(), b.shapeType().cols()),
                        b.dtypeType(),
                        b.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> result =
                MatrixOps.matmul(aMatrix, bMatrix);

        DimMatrix<M, N> resultShape = new DimMatrix<>(
                a.shapeType().rows(),
                b.shapeType().cols());

        return TypedTensor.from(result.underlying(), resultShape, a.dtypeType(), a.deviceType());
    }

    /**
     * Batched matrix multiplication for Rank3 tensors.
     *
     * <p>For A[B, M, K] and B[B, K, N], produces C[B, M, N].
     * Both batch dimension (B) and inner dimension (K) must match.
     *
     * @param a left batch of matrices [B, M, K]
     * @param b right batch of matrices [B, K, N]
     * @param <B> batch dimension (must match)
     * @param <M> row dimension
     * @param <K> inner dimension (must match)
     * @param <N> column dimension
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result batch of matrices [B, M, N]
     */
    public static <B extends Dim, M extends Dim, K extends Dim, N extends Dim,
            D extends DTypeTag, V extends DeviceTag>
    TypedTensor<DimRank3<B, M, N>, D, V> batchedMatmul(
            TypedTensor<DimRank3<B, M, K>, D, V> a,
            TypedTensor<DimRank3<B, K, N>, D, V> b) {

        // Delegate to existing MatrixOps implementation
        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Rank3, D, V> aRank3 =
                TypedTensor.from(
                        a.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Rank3(
                                a.shapeType().dim0(), a.shapeType().dim1(), a.shapeType().dim2()),
                        a.dtypeType(),
                        a.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Rank3, D, V> bRank3 =
                TypedTensor.from(
                        b.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Rank3(
                                b.shapeType().dim0(), b.shapeType().dim1(), b.shapeType().dim2()),
                        b.dtypeType(),
                        b.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Rank3, D, V> result =
                MatrixOps.batchedMatmul(aRank3, bRank3);

        DimRank3<B, M, N> resultShape = new DimRank3<>(
                a.shapeType().dim0(),
                a.shapeType().dim1(),
                b.shapeType().dim2());

        return TypedTensor.from(result.underlying(), resultShape, a.dtypeType(), a.deviceType());
    }

    /**
     * Batched matrix multiplication for Rank4 tensors (multi-head attention).
     *
     * <p>For A[B, H, M, K] and B[B, H, K, N], produces C[B, H, M, N].
     * This is specifically designed for multi-head attention where:
     * <ul>
     *   <li>B is batch size
     *   <li>H is number of attention heads
     *   <li>M is sequence length (query)
     *   <li>K is head dimension
     *   <li>N is sequence length (key/value)
     * </ul>
     *
     * @param a left tensor [B, H, M, K] - typically Q after reshape
     * @param b right tensor [B, H, K, N] - typically K^T after reshape
     * @param <B> batch dimension (must match)
     * @param <H> heads dimension (must match)
     * @param <M> row dimension
     * @param <K> inner dimension (must match)
     * @param <N> column dimension
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result tensor [B, H, M, N] - attention scores
     */
    public static <B extends Dim, H extends Dim, M extends Dim, K extends Dim, N extends Dim,
            D extends DTypeTag, V extends DeviceTag>
    TypedTensor<DimRank4<B, H, M, N>, D, V> batchedMatmulRank4(
            TypedTensor<DimRank4<B, H, M, K>, D, V> a,
            TypedTensor<DimRank4<B, H, K, N>, D, V> b) {

        // Delegate to existing MatrixOps implementation
        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Rank4, D, V> aRank4 =
                TypedTensor.from(
                        a.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Rank4(
                                a.shapeType().dim0(), a.shapeType().dim1(),
                                a.shapeType().dim2(), a.shapeType().dim3()),
                        a.dtypeType(),
                        a.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Rank4, D, V> bRank4 =
                TypedTensor.from(
                        b.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Rank4(
                                b.shapeType().dim0(), b.shapeType().dim1(),
                                b.shapeType().dim2(), b.shapeType().dim3()),
                        b.dtypeType(),
                        b.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Rank4, D, V> result =
                MatrixOps.batchedMatmulRank4(aRank4, bRank4);

        DimRank4<B, H, M, N> resultShape = new DimRank4<>(
                a.shapeType().dim0(),
                a.shapeType().dim1(),
                a.shapeType().dim2(),
                b.shapeType().dim3());

        return TypedTensor.from(result.underlying(), resultShape, a.dtypeType(), a.deviceType());
    }

    // ==================== Matrix-Vector Operations ====================

    /**
     * Matrix-vector multiplication: y = A @ x.
     *
     * <p>For A[M, N] and x[N], produces y[M].
     * The N dimension must match between matrix columns and vector length.
     *
     * @param matrix the matrix [M, N]
     * @param vector the vector [N]
     * @param <M> row dimension of matrix (and result length)
     * @param <N> column dimension of matrix (must match vector length)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result vector [M]
     */
    public static <M extends Dim, N extends Dim, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<DimVector<M>, D, V> matvec(
            TypedTensor<DimMatrix<M, N>, D, V> matrix,
            TypedTensor<DimVector<N>, D, V> vector) {

        // Delegate to existing MatrixOps implementation
        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> matrixTyped =
                TypedTensor.from(
                        matrix.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Matrix(
                                matrix.shapeType().rows(), matrix.shapeType().cols()),
                        matrix.dtypeType(),
                        matrix.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Vector, D, V> vectorTyped =
                TypedTensor.from(
                        vector.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Vector(
                                vector.shapeType().length()),
                        vector.dtypeType(),
                        vector.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Vector, D, V> result =
                MatrixOps.matvec(matrixTyped, vectorTyped);

        DimVector<M> resultShape = new DimVector<>(matrix.shapeType().rows());

        return TypedTensor.from(result.underlying(), resultShape, matrix.dtypeType(), matrix.deviceType());
    }

    /**
     * Vector-matrix multiplication: y = x @ A.
     *
     * <p>For x[M] and A[M, N], produces y[N].
     * The M dimension must match between vector length and matrix rows.
     *
     * @param vector the row vector [M]
     * @param matrix the matrix [M, N]
     * @param <M> vector length (must match matrix rows)
     * @param <N> column dimension of matrix (and result length)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result vector [N]
     */
    public static <M extends Dim, N extends Dim, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<DimVector<N>, D, V> vecmat(
            TypedTensor<DimVector<M>, D, V> vector,
            TypedTensor<DimMatrix<M, N>, D, V> matrix) {

        // Delegate to existing MatrixOps implementation
        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Vector, D, V> vectorTyped =
                TypedTensor.from(
                        vector.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Vector(
                                vector.shapeType().length()),
                        vector.dtypeType(),
                        vector.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> matrixTyped =
                TypedTensor.from(
                        matrix.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Matrix(
                                matrix.shapeType().rows(), matrix.shapeType().cols()),
                        matrix.dtypeType(),
                        matrix.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Vector, D, V> result =
                MatrixOps.vecmat(vectorTyped, matrixTyped);

        DimVector<N> resultShape = new DimVector<>(matrix.shapeType().cols());

        return TypedTensor.from(result.underlying(), resultShape, matrix.dtypeType(), matrix.deviceType());
    }

    // ==================== Transpose Operations ====================

    /**
     * Transposes a matrix, swapping dimension types.
     *
     * <p>For A[R, C], produces A^T[C, R].
     * The dimension types are swapped to maintain type safety.
     *
     * @param a the input matrix
     * @param <R> row dimension type
     * @param <C> column dimension type
     * @param <D> dtype
     * @param <V> device
     * @return the transposed matrix with swapped dimension types
     */
    public static <R extends Dim, C extends Dim, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<DimMatrix<C, R>, D, V> transpose(TypedTensor<DimMatrix<R, C>, D, V> a) {

        // Delegate to existing MatrixOps implementation
        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> matrixTyped =
                TypedTensor.from(
                        a.underlying(),
                        new io.surfworks.warpforge.core.tensor.typed.shape.Matrix(
                                a.shapeType().rows(), a.shapeType().cols()),
                        a.dtypeType(),
                        a.deviceType());

        TypedTensor<io.surfworks.warpforge.core.tensor.typed.shape.Matrix, D, V> result =
                MatrixOps.transpose(matrixTyped);

        DimMatrix<C, R> resultShape = new DimMatrix<>(
                a.shapeType().cols(),
                a.shapeType().rows());

        return TypedTensor.from(result.underlying(), resultShape, a.dtypeType(), a.deviceType());
    }

    // ==================== Element-wise Operations ====================

    /**
     * Element-wise addition of two tensors with the same shape type.
     *
     * <p>Both tensors must have identical shape, dtype, and device types.
     *
     * @param a first tensor
     * @param b second tensor
     * @param <S> shape type (must match exactly)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result tensor with same shape
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> add(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        Tensor result = TypedOps.add(a, b).underlying();
        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Element-wise subtraction of two tensors with the same shape type.
     *
     * @param a first tensor
     * @param b second tensor (subtracted from a)
     * @param <S> shape type (must match exactly)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result tensor with same shape (a - b)
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> sub(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        Tensor result = TypedOps.sub(a, b).underlying();
        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Element-wise multiplication (Hadamard product) of two tensors.
     *
     * @param a first tensor
     * @param b second tensor
     * @param <S> shape type (must match exactly)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result tensor with same shape (a * b element-wise)
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> mul(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        Tensor result = TypedOps.mul(a, b).underlying();
        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Element-wise division of two tensors.
     *
     * @param a numerator tensor
     * @param b denominator tensor
     * @param <S> shape type (must match exactly)
     * @param <D> dtype (must match)
     * @param <V> device (must match)
     * @return result tensor with same shape (a / b element-wise)
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> div(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        Tensor result = TypedOps.div(a, b).underlying();
        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Scales a F32 tensor by a scalar value.
     *
     * @param a the tensor to scale
     * @param scalar the scalar multiplier
     * @param <S> shape type
     * @param <V> device
     * @return scaled tensor with same shape
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, io.surfworks.warpforge.core.tensor.typed.dtype.F32, V> scale(
            TypedTensor<S, io.surfworks.warpforge.core.tensor.typed.dtype.F32, V> a, float scalar) {
        TypedTensor<S, io.surfworks.warpforge.core.tensor.typed.dtype.F32, V> result = TypedOps.scale(a, scalar);
        return TypedTensor.from(result.underlying(), a.shapeType(),
                io.surfworks.warpforge.core.tensor.typed.dtype.F32.INSTANCE, a.deviceType());
    }

    /**
     * Scales a F64 tensor by a scalar value.
     *
     * @param a the tensor to scale
     * @param scalar the scalar multiplier
     * @param <S> shape type
     * @param <V> device
     * @return scaled tensor with same shape
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, io.surfworks.warpforge.core.tensor.typed.dtype.F64, V> scale(
            TypedTensor<S, io.surfworks.warpforge.core.tensor.typed.dtype.F64, V> a, double scalar) {
        TypedTensor<S, io.surfworks.warpforge.core.tensor.typed.dtype.F64, V> result = TypedOps.scale(a, scalar);
        return TypedTensor.from(result.underlying(), a.shapeType(),
                io.surfworks.warpforge.core.tensor.typed.dtype.F64.INSTANCE, a.deviceType());
    }
}
