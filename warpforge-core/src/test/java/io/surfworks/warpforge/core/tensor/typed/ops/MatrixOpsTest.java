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
}
