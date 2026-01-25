package io.surfworks.warpforge.core.tensor.typed.dim;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.ops.DimOps;
import io.surfworks.warpforge.core.tensor.typed.shape.DimMatrix;
import io.surfworks.warpforge.core.tensor.typed.shape.DimRank3;
import io.surfworks.warpforge.core.tensor.typed.shape.DimRank4;
import io.surfworks.warpforge.core.tensor.typed.shape.DimVector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Tests for dimension-typed tensor operations.
 *
 * <p>These tests verify that DimOps correctly:
 * <ul>
 *   <li>Produces correct output shapes
 *   <li>Computes correct values (delegating to underlying ops)
 *   <li>Preserves dimension type information
 * </ul>
 *
 * <p>Note: Compile-time type checking cannot be tested at runtime.
 * See {@link CompileTimeShapeTest} for documentation of what should NOT compile.
 */
@DisplayName("DimOps - Dimension-Typed Operations")
class DimOpsTest {

    // Define test dimensions
    interface M extends Dim {}
    interface K extends Dim {}
    interface N extends Dim {}
    interface Batch extends Dim {}
    interface Heads extends Dim {}

    @Nested
    @DisplayName("Matrix Multiplication")
    class MatmulTests {

        @Test
        @DisplayName("matmul produces correct output shape")
        void matmulProducesCorrectShape() {
            // A[3, 4] @ B[4, 5] = C[3, 5]
            try (var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                    new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 var b = TypedTensor.<DimMatrix<K, N>, F32, Cpu>zeros(
                         new DimMatrix<>(4, 5), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimMatrix<M, N>, F32, Cpu> c = DimOps.matmul(a, b);

                assertNotNull(c);
                assertArrayEquals(new int[]{3, 5}, c.dimensions());
                assertEquals(2, c.rank());

                c.close();
            }
        }

        @Test
        @DisplayName("matmul computes correct values")
        void matmulComputesCorrectValues() {
            // Create a simple 2x2 identity test
            // [1, 0]   [2, 3]   [2, 3]
            // [0, 1] @ [4, 5] = [4, 5]
            float[] aData = {1, 0, 0, 1};
            float[] bData = {2, 3, 4, 5};

            try (var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>fromFloatArray(
                    aData, new DimMatrix<>(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 var b = TypedTensor.<DimMatrix<K, N>, F32, Cpu>fromFloatArray(
                         bData, new DimMatrix<>(2, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimMatrix<M, N>, F32, Cpu> c = DimOps.matmul(a, b);

                java.lang.foreign.MemorySegment data = c.underlying().data();

                // Result should equal B (identity property)
                assertEquals(2.0f, data.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0), 0.001f);
                assertEquals(3.0f, data.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1), 0.001f);
                assertEquals(4.0f, data.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 2), 0.001f);
                assertEquals(5.0f, data.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 3), 0.001f);

                c.close();
            }
        }

        @Test
        @DisplayName("matmul with larger matrices")
        void matmulLargerMatrices() {
            // [32, 768] @ [768, 512] = [32, 512]
            try (var a = TypedTensor.<DimMatrix<Batch, K>, F32, Cpu>zeros(
                    new DimMatrix<>(32, 768), F32.INSTANCE, Cpu.INSTANCE);
                 var b = TypedTensor.<DimMatrix<K, N>, F32, Cpu>zeros(
                         new DimMatrix<>(768, 512), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimMatrix<Batch, N>, F32, Cpu> c = DimOps.matmul(a, b);

                assertArrayEquals(new int[]{32, 512}, c.dimensions());
                assertEquals(32 * 512, c.elementCount());

                c.close();
            }
        }
    }

    @Nested
    @DisplayName("Batched Matrix Multiplication")
    class BatchedMatmulTests {

        @Test
        @DisplayName("batchedMatmul Rank3 produces correct shape")
        void batchedMatmulRank3CorrectShape() {
            // [8, 3, 4] @ [8, 4, 5] = [8, 3, 5]
            try (var a = TypedTensor.<DimRank3<Batch, M, K>, F32, Cpu>zeros(
                    new DimRank3<>(8, 3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 var b = TypedTensor.<DimRank3<Batch, K, N>, F32, Cpu>zeros(
                         new DimRank3<>(8, 4, 5), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimRank3<Batch, M, N>, F32, Cpu> c = DimOps.batchedMatmul(a, b);

                assertArrayEquals(new int[]{8, 3, 5}, c.dimensions());
                assertEquals(3, c.rank());

                c.close();
            }
        }

        @Test
        @DisplayName("batchedMatmulRank4 for attention produces correct shape")
        void batchedMatmulRank4AttentionCorrectShape() {
            // Q: [2, 12, 128, 64] @ K^T: [2, 12, 64, 128] = scores: [2, 12, 128, 128]
            // Batch=2, Heads=12, SeqLen=128, HeadDim=64

            interface SeqLen extends Dim {}
            interface HeadDim extends Dim {}

            try (var q = TypedTensor.<DimRank4<Batch, Heads, SeqLen, HeadDim>, F32, Cpu>zeros(
                    new DimRank4<>(2, 12, 128, 64), F32.INSTANCE, Cpu.INSTANCE);
                 var kT = TypedTensor.<DimRank4<Batch, Heads, HeadDim, SeqLen>, F32, Cpu>zeros(
                         new DimRank4<>(2, 12, 64, 128), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimRank4<Batch, Heads, SeqLen, SeqLen>, F32, Cpu> scores =
                        DimOps.batchedMatmulRank4(q, kT);

                assertArrayEquals(new int[]{2, 12, 128, 128}, scores.dimensions());
                assertEquals(4, scores.rank());

                scores.close();
            }
        }
    }

    @Nested
    @DisplayName("Matrix-Vector Operations")
    class MatvecTests {

        @Test
        @DisplayName("matvec produces correct shape")
        void matvecCorrectShape() {
            // A[3, 4] @ x[4] = y[3]
            try (var matrix = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                    new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 var vector = TypedTensor.<DimVector<N>, F32, Cpu>zeros(
                         new DimVector<>(4), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimVector<M>, F32, Cpu> result = DimOps.matvec(matrix, vector);

                assertArrayEquals(new int[]{3}, result.dimensions());
                assertEquals(1, result.rank());

                result.close();
            }
        }

        @Test
        @DisplayName("vecmat produces correct shape")
        void vecmatCorrectShape() {
            // x[3] @ A[3, 4] = y[4]
            try (var vector = TypedTensor.<DimVector<M>, F32, Cpu>zeros(
                    new DimVector<>(3), F32.INSTANCE, Cpu.INSTANCE);
                 var matrix = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                         new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimVector<N>, F32, Cpu> result = DimOps.vecmat(vector, matrix);

                assertArrayEquals(new int[]{4}, result.dimensions());
                assertEquals(1, result.rank());

                result.close();
            }
        }
    }

    @Nested
    @DisplayName("Transpose")
    class TransposeTests {

        @Test
        @DisplayName("transpose swaps dimensions")
        void transposeSwapsDimensions() {
            // [3, 5] -> [5, 3]
            try (var matrix = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                    new DimMatrix<>(3, 5), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimMatrix<N, M>, F32, Cpu> transposed = DimOps.transpose(matrix);

                assertArrayEquals(new int[]{5, 3}, transposed.dimensions());
                assertEquals(2, transposed.rank());

                transposed.close();
            }
        }

        @Test
        @DisplayName("double transpose returns to original shape")
        void doubleTransposeReturnsOriginal() {
            try (var original = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                    new DimMatrix<>(3, 5), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimMatrix<N, M>, F32, Cpu> transposed = DimOps.transpose(original);
                TypedTensor<DimMatrix<M, N>, F32, Cpu> doubleTransposed = DimOps.transpose(transposed);

                assertArrayEquals(new int[]{3, 5}, doubleTransposed.dimensions());

                transposed.close();
                doubleTransposed.close();
            }
        }
    }

    @Nested
    @DisplayName("Element-wise Operations")
    class ElementwiseTests {

        @Test
        @DisplayName("add preserves shape type")
        void addPreservesShapeType() {
            try (var a = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                    new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 var b = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                         new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimMatrix<M, N>, F32, Cpu> result = DimOps.add(a, b);

                assertArrayEquals(new int[]{3, 4}, result.dimensions());
                result.close();
            }
        }

        @Test
        @DisplayName("scale preserves shape type")
        void scalePreservesShapeType() {
            try (var a = TypedTensor.<DimVector<M>, F32, Cpu>zeros(
                    new DimVector<>(10), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<DimVector<M>, F32, Cpu> result = DimOps.scale(a, 2.0f);

                assertArrayEquals(new int[]{10}, result.dimensions());
                result.close();
            }
        }
    }

    @Nested
    @DisplayName("Shape Type Information")
    class ShapeTypeInfoTests {

        @Test
        @DisplayName("DimMatrix shape provides transposed method")
        void dimMatrixTransposedMethod() {
            DimMatrix<M, N> original = new DimMatrix<>(3, 5);
            DimMatrix<N, M> transposed = original.transposed();

            assertEquals(5, transposed.rows());
            assertEquals(3, transposed.cols());
        }

        @Test
        @DisplayName("DimRank4 provides transposeLastTwo method")
        void dimRank4TransposeLastTwo() {
            DimRank4<Batch, Heads, M, N> original = new DimRank4<>(2, 8, 10, 20);
            DimRank4<Batch, Heads, N, M> transposed = original.transposeLastTwo();

            assertEquals(2, transposed.dim0());
            assertEquals(8, transposed.dim1());
            assertEquals(20, transposed.dim2());
            assertEquals(10, transposed.dim3());
        }
    }
}
