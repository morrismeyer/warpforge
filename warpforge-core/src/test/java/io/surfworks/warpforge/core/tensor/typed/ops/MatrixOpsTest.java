package io.surfworks.warpforge.core.tensor.typed.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for MatrixOps matrix operations.
 */
@DisplayName("MatrixOps")
class MatrixOpsTest {

    private static final float EPSILON = 1e-5f;
    private static final double EPSILON_D = 1e-10;

    @Nested
    @DisplayName("Matrix Multiplication")
    class MatmulTests {

        @Test
        @DisplayName("matmul computes correct result")
        void matmulComputesCorrectResult() {
            // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
            // B = [[7, 8, 9], [10, 11, 12]]  (2x3)
            // C = A @ B = [[27, 30, 33], [61, 68, 75], [95, 106, 117]]  (3x3)
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(3, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{7, 8, 9, 10, 11, 12}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> c = MatrixOps.matmul(a, b)) {

                assertEquals(3, c.shapeType().rows());
                assertEquals(3, c.shapeType().cols());

                float[] expected = {27, 30, 33, 61, 68, 75, 95, 106, 117};
                assertArrayEquals(expected, c.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("matmul handles square matrices")
        void matmulHandlesSquareMatrices() {
            // 2x2 identity @ 2x2 identity = 2x2 identity
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 0, 0, 1}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{1, 0, 0, 1}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> c = MatrixOps.matmul(a, b)) {

                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(new float[]{1, 0, 0, 1}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("matmul rejects inner dimension mismatch")
        void matmulRejectsInnerDimensionMismatch() {
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.zeros(
                    new Matrix(5, 6), F32.INSTANCE, Cpu.INSTANCE)) {

                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.matmul(a, b));
                assertTrue(ex.getMessage().contains("inner dimensions"));
            }
        }

        @Test
        @DisplayName("matmul works with F64")
        void matmulWorksWithF64() {
            try (TypedTensor<Matrix, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    new double[]{1, 2, 3, 4}, new Matrix(2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F64, Cpu> b = TypedTensor.fromDoubleArray(
                    new double[]{5, 6, 7, 8}, new Matrix(2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F64, Cpu> c = MatrixOps.matmul(a, b)) {

                // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
                double[] expected = {19, 22, 43, 50};
                assertArrayEquals(expected, c.underlying().toDoubleArray(), EPSILON_D);
            }
        }

        @Test
        @DisplayName("matmul produces correct output dimensions")
        void matmulProducesCorrectOutputDimensions() {
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(
                    new Matrix(10, 20), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.zeros(
                    new Matrix(20, 30), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> c = MatrixOps.matmul(a, b)) {

                assertEquals(10, c.shapeType().rows());
                assertEquals(30, c.shapeType().cols());
            }
        }
    }

    @Nested
    @DisplayName("Matrix-Vector Operations")
    class MatvecTests {

        @Test
        @DisplayName("matvec computes matrix-vector product")
        void matvecComputesMatrixVectorProduct() {
            // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
            // x = [1, 2, 3]
            // y = A @ x = [14, 32]
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = MatrixOps.matvec(mat, vec)) {

                assertEquals(2, result.shapeType().length());
                assertArrayEquals(new float[]{14, 32}, result.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("matvec rejects dimension mismatch")
        void matvecRejectsDimensionMismatch() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.matvec(mat, vec));
            }
        }

        @Test
        @DisplayName("vecmat computes vector-matrix product")
        void vecmatComputesVectorMatrixProduct() {
            // x = [1, 2]
            // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
            // y = x @ A = [9, 12, 15]
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = MatrixOps.vecmat(vec, mat)) {

                assertEquals(3, result.shapeType().length());
                assertArrayEquals(new float[]{9, 12, 15}, result.underlying().toFloatArray(), EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Dot Product")
    class DotTests {

        @Test
        @DisplayName("dot computes inner product")
        void dotComputesInnerProduct() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{4, 5, 6}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
                float result = MatrixOps.dot(a, b);
                assertEquals(32.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("dotF64 works with F64")
        void dotF64WorksWithF64() {
            try (TypedTensor<Vector, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    new double[]{1.0, 2.0, 3.0}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> b = TypedTensor.fromDoubleArray(
                    new double[]{4.0, 5.0, 6.0}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE)) {

                double result = MatrixOps.dotF64(a, b);
                assertEquals(32.0, result, EPSILON_D);
            }
        }

        @Test
        @DisplayName("dot rejects length mismatch")
        void dotRejectsLengthMismatch() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.zeros(
                    new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.dot(a, b));
            }
        }
    }

    @Nested
    @DisplayName("Transpose")
    class TransposeTests {

        @Test
        @DisplayName("transpose swaps dimensions")
        void transposeSwapsDimensions() {
            // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
            // A^T = [[1, 4], [2, 5], [3, 6]]  (3x2)
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> at = MatrixOps.transpose(a)) {

                assertEquals(3, at.shapeType().rows());
                assertEquals(2, at.shapeType().cols());

                float[] expected = {1, 4, 2, 5, 3, 6};
                assertArrayEquals(expected, at.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("double transpose returns original")
        void doubleTransposeReturnsOriginal() {
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> att = MatrixOps.transpose(MatrixOps.transpose(a))) {

                assertArrayEquals(new int[]{2, 3}, att.dimensions());
                assertArrayEquals(a.underlying().toFloatArray(),
                                  att.underlying().toFloatArray(), EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Outer Product")
    class OuterTests {

        @Test
        @DisplayName("outer computes outer product")
        void outerComputesOuterProduct() {
            // a = [1, 2, 3]
            // b = [4, 5]
            // a ⊗ b = [[4, 5], [8, 10], [12, 15]]  (3x2)
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{4, 5}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> result = MatrixOps.outer(a, b)) {

                assertEquals(3, result.shapeType().rows());
                assertEquals(2, result.shapeType().cols());

                float[] expected = {4, 5, 8, 10, 12, 15};
                assertArrayEquals(expected, result.underlying().toFloatArray(), EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Type Safety")
    class TypeSafety {

        @Test
        @DisplayName("matmul returns Matrix shape type")
        void matmulReturnsMatrixShapeType() {
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.zeros(
                    new Matrix(4, 5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> c = MatrixOps.matmul(a, b)) {

                // Return type is TypedTensor<Matrix, F32, Cpu> - verified by compilation
                Matrix resultShape = c.shapeType();
                assertEquals(3, resultShape.rows());
                assertEquals(5, resultShape.cols());
            }
        }

        @Test
        @DisplayName("matvec returns Vector shape type")
        void matvecReturnsVectorShapeType() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = TypedTensor.zeros(
                    new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = MatrixOps.matvec(mat, vec)) {

                // Return type is TypedTensor<Vector, F32, Cpu> - verified by compilation
                Vector resultShape = result.shapeType();
                assertEquals(3, resultShape.length());
            }
        }

        // Note: The following would not compile if uncommented:
        //
        // @Test
        // void matmulDtypeMismatchDoesNotCompile() {
        //     TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
        //     TypedTensor<Matrix, F64, Cpu> b = TypedTensor.zeros(new Matrix(4, 5), F64.INSTANCE, Cpu.INSTANCE);
        //     MatrixOps.matmul(a, b);  // ERROR: F32 != F64
        // }
    }

    private static void assertTrue(boolean condition) {
        org.junit.jupiter.api.Assertions.assertTrue(condition);
    }

    @Nested
    @DisplayName("Batched Matrix Multiplication (Rank3)")
    class BatchedMatmulTests {

        @Test
        @DisplayName("batchedMatmul computes correct result for each batch")
        void batchedMatmulComputesCorrectResultForEachBatch() {
            // Batch of 2 matrices: A[2, 2, 3], B[2, 3, 2]
            // Batch 0: [[1,2,3], [4,5,6]] @ [[7,8], [9,10], [11,12]]
            // Batch 1: [[1,0,0], [0,1,0]] @ [[1,2], [3,4], [5,6]]
            float[] aData = {
                    1, 2, 3, 4, 5, 6,     // Batch 0: 2x3
                    1, 0, 0, 0, 1, 0      // Batch 1: 2x3
            };
            float[] bData = {
                    7, 8, 9, 10, 11, 12,  // Batch 0: 3x2
                    1, 2, 3, 4, 5, 6      // Batch 1: 3x2
            };

            try (TypedTensor<Rank3, F32, Cpu> a = TypedTensor.fromFloatArray(
                    aData, new Rank3(2, 2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> b = TypedTensor.fromFloatArray(
                    bData, new Rank3(2, 3, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> c = MatrixOps.batchedMatmul(a, b)) {

                assertArrayEquals(new int[]{2, 2, 2}, c.dimensions());

                float[] result = c.underlying().toFloatArray();

                // Batch 0: [[1,2,3], [4,5,6]] @ [[7,8], [9,10], [11,12]]
                // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
                // = [[58, 64], [139, 154]]
                assertEquals(58.0f, result[0], EPSILON);
                assertEquals(64.0f, result[1], EPSILON);
                assertEquals(139.0f, result[2], EPSILON);
                assertEquals(154.0f, result[3], EPSILON);

                // Batch 1: identity-like @ [[1,2], [3,4], [5,6]]
                // = [[1,2], [3,4]]
                assertEquals(1.0f, result[4], EPSILON);
                assertEquals(2.0f, result[5], EPSILON);
                assertEquals(3.0f, result[6], EPSILON);
                assertEquals(4.0f, result[7], EPSILON);
            }
        }

        @Test
        @DisplayName("batchedMatmul handles single batch")
        void batchedMatmulHandlesSingleBatch() {
            float[] aData = {1, 2, 3, 4, 5, 6};  // [1, 2, 3]
            float[] bData = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // [1, 3, 3] identity

            try (TypedTensor<Rank3, F32, Cpu> a = TypedTensor.fromFloatArray(
                    aData, new Rank3(1, 2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> b = TypedTensor.fromFloatArray(
                    bData, new Rank3(1, 3, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> c = MatrixOps.batchedMatmul(a, b)) {

                assertArrayEquals(new int[]{1, 2, 3}, c.dimensions());

                // Multiply by identity should return original
                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(aData, result, EPSILON);
            }
        }

        @Test
        @DisplayName("batchedMatmul rejects batch dimension mismatch")
        void batchedMatmulRejectsBatchDimensionMismatch() {
            try (TypedTensor<Rank3, F32, Cpu> a = TypedTensor.zeros(
                    new Rank3(3, 4, 5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> b = TypedTensor.zeros(
                    new Rank3(2, 5, 6), F32.INSTANCE, Cpu.INSTANCE)) {

                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.batchedMatmul(a, b));
                assertTrue(ex.getMessage().contains("Batch"));
            }
        }

        @Test
        @DisplayName("batchedMatmul rejects inner dimension mismatch")
        void batchedMatmulRejectsInnerDimensionMismatch() {
            try (TypedTensor<Rank3, F32, Cpu> a = TypedTensor.zeros(
                    new Rank3(2, 4, 5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> b = TypedTensor.zeros(
                    new Rank3(2, 6, 7), F32.INSTANCE, Cpu.INSTANCE)) {

                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.batchedMatmul(a, b));
                assertTrue(ex.getMessage().contains("Inner"));
            }
        }

        @Test
        @DisplayName("batchedMatmul works with F64")
        void batchedMatmulWorksWithF64() {
            double[] aData = {1, 2, 3, 4};  // [1, 2, 2]
            double[] bData = {1, 0, 0, 1};  // [1, 2, 2] identity

            try (TypedTensor<Rank3, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    aData, new Rank3(1, 2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F64, Cpu> b = TypedTensor.fromDoubleArray(
                    bData, new Rank3(1, 2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F64, Cpu> c = MatrixOps.batchedMatmul(a, b)) {

                assertArrayEquals(aData, c.underlying().toDoubleArray(), EPSILON_D);
            }
        }

        @Test
        @DisplayName("batchedMatmul produces correct output dimensions")
        void batchedMatmulProducesCorrectOutputDimensions() {
            try (TypedTensor<Rank3, F32, Cpu> a = TypedTensor.zeros(
                    new Rank3(8, 16, 32), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> b = TypedTensor.zeros(
                    new Rank3(8, 32, 64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> c = MatrixOps.batchedMatmul(a, b)) {

                Rank3 shape = c.shapeType();
                assertEquals(8, shape.dim0());
                assertEquals(16, shape.dim1());
                assertEquals(64, shape.dim2());
            }
        }

        @Test
        @DisplayName("batchedMatmul batches are independent")
        void batchedMatmulBatchesAreIndependent() {
            // Different scaling in each batch
            float[] aData = {
                    2, 0, 0, 2,   // Batch 0: 2*I
                    3, 0, 0, 3    // Batch 1: 3*I
            };
            float[] bData = {
                    1, 2, 3, 4,   // Batch 0
                    1, 2, 3, 4    // Batch 1 (same as batch 0)
            };

            try (TypedTensor<Rank3, F32, Cpu> a = TypedTensor.fromFloatArray(
                    aData, new Rank3(2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> b = TypedTensor.fromFloatArray(
                    bData, new Rank3(2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> c = MatrixOps.batchedMatmul(a, b)) {

                float[] result = c.underlying().toFloatArray();

                // Batch 0: 2*I @ [[1,2],[3,4]] = [[2,4],[6,8]]
                assertEquals(2.0f, result[0], EPSILON);
                assertEquals(4.0f, result[1], EPSILON);
                assertEquals(6.0f, result[2], EPSILON);
                assertEquals(8.0f, result[3], EPSILON);

                // Batch 1: 3*I @ [[1,2],[3,4]] = [[3,6],[9,12]]
                assertEquals(3.0f, result[4], EPSILON);
                assertEquals(6.0f, result[5], EPSILON);
                assertEquals(9.0f, result[6], EPSILON);
                assertEquals(12.0f, result[7], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Batched Matrix Multiplication (Rank4 - Multi-Head Attention)")
    class BatchedMatmulRank4Tests {

        @Test
        @DisplayName("batchedMatmulRank4 computes Q @ K^T pattern")
        void batchedMatmulRank4ComputesQKPattern() {
            // Simulate attention: Q[batch, heads, seq, head_dim] @ K^T[batch, heads, head_dim, seq]
            // Simplified: [1, 2, 2, 3] @ [1, 2, 3, 2]
            float[] qData = new float[12];  // 1 * 2 * 2 * 3 = 12
            float[] kData = new float[12];  // 1 * 2 * 3 * 2 = 12

            // Layout for Q [1, 2, 2, 3]: [batch, head, row, col]
            // Head 0, Q: [[1,0,0], [0,1,0]] - selects first two rows of K^T
            // Indices: [0,0,0,0]=0, [0,0,0,1]=1, [0,0,0,2]=2, [0,0,1,0]=3, [0,0,1,1]=4, [0,0,1,2]=5
            qData[0] = 1;  // Q[0,0,0,0] = 1
            qData[4] = 1;  // Q[0,0,1,1] = 1

            // Head 1, Q: [[1,1,0], [0,0,1]] - combines first two rows, selects third
            // Indices: [0,1,0,0]=6, [0,1,0,1]=7, [0,1,0,2]=8, [0,1,1,0]=9, [0,1,1,1]=10, [0,1,1,2]=11
            qData[6] = 1;   // Q[0,1,0,0] = 1
            qData[7] = 1;   // Q[0,1,0,1] = 1
            qData[11] = 1;  // Q[0,1,1,2] = 1

            // K^T for both heads: [[1,2], [3,4], [5,6]] with shape [1, 2, 3, 2]
            // Head 0: indices 0-5, Head 1: indices 6-11
            kData[0] = 1; kData[1] = 2;
            kData[2] = 3; kData[3] = 4;
            kData[4] = 5; kData[5] = 6;
            kData[6] = 1; kData[7] = 2;
            kData[8] = 3; kData[9] = 4;
            kData[10] = 5; kData[11] = 6;

            try (TypedTensor<Rank4, F32, Cpu> q = TypedTensor.fromFloatArray(
                    qData, new Rank4(1, 2, 2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> k = TypedTensor.fromFloatArray(
                    kData, new Rank4(1, 2, 3, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> attn = MatrixOps.batchedMatmulRank4(q, k)) {

                assertArrayEquals(new int[]{1, 2, 2, 2}, attn.dimensions());

                float[] result = attn.underlying().toFloatArray();

                // Head 0: [[1,0,0], [0,1,0]] @ [[1,2], [3,4], [5,6]] = [[1,2], [3,4]]
                assertEquals(1.0f, result[0], EPSILON);
                assertEquals(2.0f, result[1], EPSILON);
                assertEquals(3.0f, result[2], EPSILON);
                assertEquals(4.0f, result[3], EPSILON);

                // Head 1: [[1,1,0], [0,0,1]] @ [[1,2], [3,4], [5,6]] = [[4,6], [5,6]]
                assertEquals(4.0f, result[4], EPSILON);
                assertEquals(6.0f, result[5], EPSILON);
                assertEquals(5.0f, result[6], EPSILON);
                assertEquals(6.0f, result[7], EPSILON);
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 handles multiple batches and heads")
        void batchedMatmulRank4HandlesMultipleBatchesAndHeads() {
            // [2, 2, 2, 2] @ [2, 2, 2, 2] = [2, 2, 2, 2]
            // Using identity matrices scaled differently per batch/head

            float[] aData = new float[16];
            float[] bData = new float[16];

            // Fill with identities scaled by (batch + 1) * (head + 1)
            for (int b = 0; b < 2; b++) {
                for (int h = 0; h < 2; h++) {
                    float scale = (b + 1) * (h + 1);
                    int offset = (b * 2 + h) * 4;
                    aData[offset] = scale;
                    aData[offset + 3] = scale;
                    bData[offset] = 1;
                    bData[offset + 1] = 2;
                    bData[offset + 2] = 3;
                    bData[offset + 3] = 4;
                }
            }

            try (TypedTensor<Rank4, F32, Cpu> a = TypedTensor.fromFloatArray(
                    aData, new Rank4(2, 2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> b = TypedTensor.fromFloatArray(
                    bData, new Rank4(2, 2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> c = MatrixOps.batchedMatmulRank4(a, b)) {

                float[] result = c.underlying().toFloatArray();

                // Each scaled identity times [[1,2],[3,4]] = scale * [[1,2],[3,4]]
                // Batch 0, Head 0: scale = 1
                assertEquals(1.0f, result[0], EPSILON);
                assertEquals(2.0f, result[1], EPSILON);

                // Batch 0, Head 1: scale = 2
                assertEquals(2.0f, result[4], EPSILON);
                assertEquals(4.0f, result[5], EPSILON);

                // Batch 1, Head 0: scale = 2
                assertEquals(2.0f, result[8], EPSILON);
                assertEquals(4.0f, result[9], EPSILON);

                // Batch 1, Head 1: scale = 4
                assertEquals(4.0f, result[12], EPSILON);
                assertEquals(8.0f, result[13], EPSILON);
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 rejects batch mismatch")
        void batchedMatmulRank4RejectsBatchMismatch() {
            try (TypedTensor<Rank4, F32, Cpu> a = TypedTensor.zeros(
                    new Rank4(3, 4, 5, 6), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> b = TypedTensor.zeros(
                    new Rank4(2, 4, 6, 7), F32.INSTANCE, Cpu.INSTANCE)) {

                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.batchedMatmulRank4(a, b));
                assertTrue(ex.getMessage().contains("Batch"));
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 rejects head mismatch")
        void batchedMatmulRank4RejectsHeadMismatch() {
            try (TypedTensor<Rank4, F32, Cpu> a = TypedTensor.zeros(
                    new Rank4(2, 8, 5, 6), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> b = TypedTensor.zeros(
                    new Rank4(2, 4, 6, 7), F32.INSTANCE, Cpu.INSTANCE)) {

                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.batchedMatmulRank4(a, b));
                assertTrue(ex.getMessage().contains("Head"));
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 rejects inner dimension mismatch")
        void batchedMatmulRank4RejectsInnerDimensionMismatch() {
            try (TypedTensor<Rank4, F32, Cpu> a = TypedTensor.zeros(
                    new Rank4(2, 4, 5, 64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> b = TypedTensor.zeros(
                    new Rank4(2, 4, 32, 7), F32.INSTANCE, Cpu.INSTANCE)) {

                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> MatrixOps.batchedMatmulRank4(a, b));
                assertTrue(ex.getMessage().contains("Inner"));
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 works with F64")
        void batchedMatmulRank4WorksWithF64() {
            double[] aData = {1, 0, 0, 1, 2, 0, 0, 2};  // [1, 2, 2, 2] - scaled identities
            double[] bData = {1, 2, 3, 4, 1, 2, 3, 4};  // [1, 2, 2, 2]

            try (TypedTensor<Rank4, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    aData, new Rank4(1, 2, 2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F64, Cpu> b = TypedTensor.fromDoubleArray(
                    bData, new Rank4(1, 2, 2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F64, Cpu> c = MatrixOps.batchedMatmulRank4(a, b)) {

                double[] result = c.underlying().toDoubleArray();

                // Head 0: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
                assertEquals(1.0, result[0], EPSILON_D);
                assertEquals(2.0, result[1], EPSILON_D);

                // Head 1: 2I @ [[1,2],[3,4]] = [[2,4],[6,8]]
                assertEquals(2.0, result[4], EPSILON_D);
                assertEquals(4.0, result[5], EPSILON_D);
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 produces correct output dimensions")
        void batchedMatmulRank4ProducesCorrectOutputDimensions() {
            // Typical transformer attention dimensions
            int batch = 4;
            int heads = 8;
            int seqLen = 128;
            int headDim = 64;

            try (TypedTensor<Rank4, F32, Cpu> q = TypedTensor.zeros(
                    new Rank4(batch, heads, seqLen, headDim), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> k = TypedTensor.zeros(
                    new Rank4(batch, heads, headDim, seqLen), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> attn = MatrixOps.batchedMatmulRank4(q, k)) {

                // Q @ K^T produces attention scores [batch, heads, seq, seq]
                Rank4 shape = attn.shapeType();
                assertEquals(batch, shape.dim0());
                assertEquals(heads, shape.dim1());
                assertEquals(seqLen, shape.dim2());
                assertEquals(seqLen, shape.dim3());
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 attention scores @ V pattern")
        void batchedMatmulRank4AttentionScoresVPattern() {
            // Second matmul in attention: attn_scores @ V
            // attn_scores: [batch, heads, q_len, k_len]
            // V: [batch, heads, k_len, head_dim]
            // Output: [batch, heads, q_len, head_dim]

            // Use simple case: attn_scores selects specific value rows
            // attn_scores = [[1,0], [0,1]] (identity, selects row 0 then row 1)
            float[] attnData = {1, 0, 0, 1, 1, 0, 0, 1};  // [1, 2, 2, 2]
            float[] vData = {1, 2, 3, 4, 5, 6, 7, 8};     // [1, 2, 2, 2] different values per head

            try (TypedTensor<Rank4, F32, Cpu> attn = TypedTensor.fromFloatArray(
                    attnData, new Rank4(1, 2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> v = TypedTensor.fromFloatArray(
                    vData, new Rank4(1, 2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> output = MatrixOps.batchedMatmulRank4(attn, v)) {

                float[] result = output.underlying().toFloatArray();

                // Head 0: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
                assertArrayEquals(new float[]{1, 2, 3, 4},
                        new float[]{result[0], result[1], result[2], result[3]}, EPSILON);

                // Head 1: I @ [[5,6],[7,8]] = [[5,6],[7,8]]
                assertArrayEquals(new float[]{5, 6, 7, 8},
                        new float[]{result[4], result[5], result[6], result[7]}, EPSILON);
            }
        }
    }
}
