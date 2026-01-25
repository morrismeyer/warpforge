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
 * <h2>Quick Reference: Shape Rules</h2>
 * <pre>
 * +------------------+----------------------------------+------------------+
 * | Operation        | Input Shapes                     | Output Shape     |
 * +------------------+----------------------------------+------------------+
 * | matmul(A, B)     | A[M, K] @ B[K, N]                | C[M, N]          |
 * | batchedMatmul    | A[B, M, K] @ B[B, K, N]          | C[B, M, N]       |
 * | batchedMatmulR4  | A[B, H, M, K] @ B[B, H, K, N]    | C[B, H, M, N]    |
 * | matvec(M, v)     | M[M, N] @ v[N]                   | y[M]             |
 * | vecmat(v, M)     | v[M] @ M[M, N]                   | y[N]             |
 * | transpose(A)     | A[R, C]                          | A^T[C, R]        |
 * | add/sub/mul/div  | A[S] op B[S]                     | C[S]             |
 * +------------------+----------------------------------+------------------+
 * </pre>
 *
 * <h2>Matmul Shape Diagram</h2>
 * <pre>
 *      A[M, K]          B[K, N]           C[M, N]
 *    +--------+       +--------+       +--------+
 *    |        |       |        |       |        |
 *  M |   A    | K   K |   B    | N = M |   C    | N
 *    |        |       |        |       |        |
 *    +--------+       +--------+       +--------+
 *           \___________/
 *           Must be same!
 * </pre>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // 1. Define dimension markers for your model
 * interface Batch extends Dim {}
 * interface Hidden extends Dim {}
 * interface Vocab extends Dim {}
 *
 * // 2. Create tensors with dimension types
 * TypedTensor<DimMatrix<Batch, Hidden>, F32, Cpu> hidden = ...;     // [batch, hidden]
 * TypedTensor<DimMatrix<Hidden, Vocab>, F32, Cpu> classifier = ...; // [hidden, vocab]
 *
 * // 3. Matmul - compiler checks that Hidden matches!
 * var logits = DimOps.matmul(hidden, classifier);  // DimMatrix<Batch, Vocab>
 * }</pre>
 *
 * <h2>Attention Pattern</h2>
 * <pre>{@code
 * // Multi-head attention with compile-time dimension checking
 * interface Batch extends Dim {}
 * interface NumHeads extends Dim {}
 * interface SeqLen extends Dim {}
 * interface HeadDim extends Dim {}
 *
 * // Q: [B, H, S, D], K^T: [B, H, D, S] -> scores: [B, H, S, S]
 * TypedTensor<DimRank4<Batch, NumHeads, SeqLen, HeadDim>, F32, Cpu> Q = ...;
 * TypedTensor<DimRank4<Batch, NumHeads, HeadDim, SeqLen>, F32, Cpu> K_T = ...;
 *
 * // HeadDim must match between Q's last dim and K_T's second-to-last dim
 * var scores = DimOps.batchedMatmulRank4(Q, K_T);
 * }</pre>
 *
 * <h2>Troubleshooting Compile Errors</h2>
 *
 * <p><b>Error: "no suitable method found for matmul(...)"</b>
 * <br>This means dimension types don't match. Check:
 * <ul>
 *   <li>Inner dimensions: A's column type must equal B's row type</li>
 *   <li>Dtype: Both tensors must have same dtype (F32, F64, etc.)</li>
 *   <li>Device: Both tensors must be on same device (Cpu, Nvidia, etc.)</li>
 * </ul>
 *
 * <p><b>Example of dimension mismatch:</b>
 * <pre>{@code
 * DimMatrix<Batch, Hidden> @ DimMatrix<Vocab, Output>
 *                  ^                    ^
 *                  Hidden != Vocab  --> COMPILE ERROR
 * }</pre>
 *
 * <p><b>Fix:</b> Ensure the inner dimension type is the same:
 * <pre>{@code
 * DimMatrix<Batch, Hidden> @ DimMatrix<Hidden, Output>  // OK!
 *                  ^                    ^
 *                  Hidden == Hidden  --> Compiles
 * }</pre>
 *
 * <h2>IntelliJ Live Templates</h2>
 * <p>Type these abbreviations and press Tab:
 * <ul>
 *   <li>{@code dimmat} - Create a DimMatrix tensor</li>
 *   <li>{@code dimvec} - Create a DimVector tensor</li>
 *   <li>{@code dimr3} / {@code dimr4} - Create rank-3/4 tensors</li>
 *   <li>{@code matmul} - Matrix multiplication with shape comment</li>
 *   <li>{@code attention} - Full attention pattern template</li>
 *   <li>{@code dimops-imports} - Import all DimOps classes</li>
 * </ul>
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
     * <h3>Shape Transformation</h3>
     * <pre>
     *   A[M, K]  @  B[K, N]  =  C[M, N]
     *        ↑________↑
     *        Must match!
     * </pre>
     *
     * <h3>Example</h3>
     * <pre>{@code
     * interface Batch extends Dim {}
     * interface Hidden extends Dim {}
     * interface Vocab extends Dim {}
     *
     * var input = TypedTensor.<DimMatrix<Batch, Hidden>, F32, Cpu>zeros(...);
     * var weights = TypedTensor.<DimMatrix<Hidden, Vocab>, F32, Cpu>zeros(...);
     *
     * // Compiles: Hidden matches Hidden
     * var output = DimOps.matmul(input, weights);  // DimMatrix<Batch, Vocab>
     * }</pre>
     *
     * <h3>Common Errors</h3>
     * <p><b>"no suitable method found"</b> - Check these three things:
     * <ol>
     *   <li><b>Inner dimension mismatch:</b> A's column type ≠ B's row type
     *       <br>{@code DimMatrix<M, K>} requires {@code DimMatrix<K, N>}, not {@code DimMatrix<P, N>}</li>
     *   <li><b>Dtype mismatch:</b> Mixing F32 and F64 tensors</li>
     *   <li><b>Device mismatch:</b> Mixing Cpu and Nvidia tensors</li>
     * </ol>
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
     * <h3>Shape Transformation</h3>
     * <pre>
     *   A[B, M, K]  @  B[B, K, N]  =  C[B, M, N]
     *     ↑              ↑              ↑
     *     └──────────────┴── Batch must match
     *           ↑_________↑
     *           Inner must match
     * </pre>
     *
     * <h3>Example</h3>
     * <pre>{@code
     * interface Batch extends Dim {}
     * interface SeqLen extends Dim {}
     * interface Hidden extends Dim {}
     *
     * var a = TypedTensor.<DimRank3<Batch, SeqLen, Hidden>, F32, Cpu>zeros(...);
     * var b = TypedTensor.<DimRank3<Batch, Hidden, SeqLen>, F32, Cpu>zeros(...);
     *
     * var c = DimOps.batchedMatmul(a, b);  // DimRank3<Batch, SeqLen, SeqLen>
     * }</pre>
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
     * <h3>Shape Transformation</h3>
     * <pre>
     *   Q[B, H, S, D]  @  K^T[B, H, D, S]  =  scores[B, H, S, S]
     *     ↑  ↑              ↑  ↑                  ↑  ↑
     *     └──┴──────────────┴──┴── Batch and Heads must match
     *              ↑_____________↑
     *              HeadDim must match
     * </pre>
     *
     * <h3>Multi-Head Attention Pattern</h3>
     * <pre>{@code
     * interface Batch extends Dim {}
     * interface NumHeads extends Dim {}
     * interface SeqLen extends Dim {}
     * interface HeadDim extends Dim {}
     *
     * // Q: [batch, heads, seq_len, head_dim]
     * var Q = TypedTensor.<DimRank4<Batch, NumHeads, SeqLen, HeadDim>, F32, Cpu>zeros(...);
     *
     * // K^T: [batch, heads, head_dim, seq_len] (transposed for matmul)
     * var K_T = TypedTensor.<DimRank4<Batch, NumHeads, HeadDim, SeqLen>, F32, Cpu>zeros(...);
     *
     * // V: [batch, heads, seq_len, head_dim]
     * var V = TypedTensor.<DimRank4<Batch, NumHeads, SeqLen, HeadDim>, F32, Cpu>zeros(...);
     *
     * // Attention scores: Q @ K^T -> [batch, heads, seq_len, seq_len]
     * var scores = DimOps.batchedMatmulRank4(Q, K_T);
     *
     * // After softmax: scores @ V -> [batch, heads, seq_len, head_dim]
     * // var output = DimOps.batchedMatmulRank4(softmax(scores), V);
     * }</pre>
     *
     * <h3>Dimension Semantics</h3>
     * <ul>
     *   <li><b>B</b> - Batch size (number of sequences)</li>
     *   <li><b>H</b> - Number of attention heads</li>
     *   <li><b>M</b> - Query sequence length</li>
     *   <li><b>K</b> - Head dimension (hidden_dim / num_heads)</li>
     *   <li><b>N</b> - Key/Value sequence length</li>
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
     * <h3>Shape Transformation</h3>
     * <pre>
     *   A[M, N]  @  x[N]  =  y[M]
     *       ↑________↑
     *       Must match!
     * </pre>
     *
     * <h3>Example: Linear Layer Forward</h3>
     * <pre>{@code
     * interface Features extends Dim {}
     * interface Output extends Dim {}
     *
     * var weights = TypedTensor.<DimMatrix<Output, Features>, F32, Cpu>zeros(...);
     * var input = TypedTensor.<DimVector<Features>, F32, Cpu>zeros(...);
     *
     * // Features dimension must match
     * var output = DimOps.matvec(weights, input);  // DimVector<Output>
     * }</pre>
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
     * <h3>Shape Transformation</h3>
     * <pre>
     *   x[M]  @  A[M, N]  =  y[N]
     *     ↑______↑
     *     Must match!
     * </pre>
     *
     * <h3>Example</h3>
     * <pre>{@code
     * interface Hidden extends Dim {}
     * interface Output extends Dim {}
     *
     * var hidden = TypedTensor.<DimVector<Hidden>, F32, Cpu>zeros(...);
     * var weights = TypedTensor.<DimMatrix<Hidden, Output>, F32, Cpu>zeros(...);
     *
     * var output = DimOps.vecmat(hidden, weights);  // DimVector<Output>
     * }</pre>
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
     * <h3>Shape Transformation</h3>
     * <pre>
     *   A[R, C]  -->  A^T[C, R]
     *
     *   Before:          After:
     *   +------+         +---+
     *   |      | C       |   |
     * R |  A   |    -->  | A | R
     *   |      |       C |   |
     *   +------+         +---+
     * </pre>
     *
     * <h3>Example: Preparing K for Attention</h3>
     * <pre>{@code
     * interface SeqLen extends Dim {}
     * interface HeadDim extends Dim {}
     *
     * // K: [SeqLen, HeadDim]
     * var K = TypedTensor.<DimMatrix<SeqLen, HeadDim>, F32, Cpu>zeros(...);
     *
     * // K^T: [HeadDim, SeqLen] - ready for Q @ K^T
     * var K_T = DimOps.transpose(K);  // DimMatrix<HeadDim, SeqLen>
     * }</pre>
     *
     * <h3>Type Safety</h3>
     * <p>Note that transpose swaps the dimension types. If you have
     * {@code DimMatrix<M, N>}, transpose gives you {@code DimMatrix<N, M>}.
     * This is intentional - it maintains type safety through the operation.
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
