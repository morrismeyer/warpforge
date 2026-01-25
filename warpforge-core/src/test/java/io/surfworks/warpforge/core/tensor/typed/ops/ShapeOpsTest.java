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
import io.surfworks.warpforge.core.tensor.typed.shape.Dynamic;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for ShapeOps shape manipulation operations.
 */
@DisplayName("ShapeOps")
class ShapeOpsTest {

    private static final float EPSILON = 1e-5f;
    private static final double EPSILON_D = 1e-10;

    @Nested
    @DisplayName("Reshape Operations")
    class ReshapeTests {

        @Test
        @DisplayName("reshape vector to matrix preserves data")
        void reshapeVectorToMatrixPreservesData() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Vector(6), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> mat = ShapeOps.reshape(vec, new Matrix(2, 3))) {

                assertArrayEquals(new int[]{2, 3}, mat.dimensions());
                assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, mat.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("reshape matrix to vector preserves data")
        void reshapeMatrixToVectorPreservesData() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = ShapeOps.reshape(mat, new Vector(6))) {

                assertArrayEquals(new int[]{6}, vec.dimensions());
                assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, vec.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("reshape matrix to rank3 preserves data")
        void reshapeMatrixToRank3PreservesData() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                    new Matrix(4, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> rank3 = ShapeOps.reshape(mat, new Rank3(2, 2, 3))) {

                assertArrayEquals(new int[]{2, 2, 3}, rank3.dimensions());
                assertEquals(12, rank3.elementCount());
            }
        }

        @Test
        @DisplayName("reshape rejects element count mismatch")
        void reshapeRejectsElementCountMismatch() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ShapeOps.reshape(vec, new Matrix(3, 4)));
            }
        }

        @Test
        @DisplayName("reshape works with F64")
        void reshapeWorksWithF64() {
            try (TypedTensor<Vector, F64, Cpu> vec = TypedTensor.fromDoubleArray(
                    new double[]{1.0, 2.0, 3.0, 4.0}, new Vector(4), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F64, Cpu> mat = ShapeOps.reshape(vec, new Matrix(2, 2))) {

                assertArrayEquals(new int[]{2, 2}, mat.dimensions());
                assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0}, mat.underlying().toDoubleArray(), EPSILON_D);
            }
        }

        @Test
        @DisplayName("reshapeToDynamic creates dynamic shape")
        void reshapeToDynamicCreatesDynamicShape() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Vector(6), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Dynamic, F32, Cpu> dyn = ShapeOps.reshapeToDynamic(vec, 2, 3)) {

                assertArrayEquals(new int[]{2, 3}, dyn.dimensions());
            }
        }
    }

    @Nested
    @DisplayName("Flatten Operations")
    class FlattenTests {

        @Test
        @DisplayName("flatten matrix to vector")
        void flattenMatrixToVector() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = ShapeOps.flatten(mat)) {

                assertEquals(6, vec.shapeType().length());
                assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, vec.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("flatten rank3 to vector")
        void flattenRank3ToVector() {
            try (TypedTensor<Rank3, F32, Cpu> rank3 = TypedTensor.zeros(
                    new Rank3(2, 3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = ShapeOps.flatten(rank3)) {

                assertEquals(24, vec.shapeType().length());
            }
        }
    }

    @Nested
    @DisplayName("Permute Operations")
    class PermuteTests {

        @Test
        @DisplayName("permute rank3 with identity permutation")
        void permuteRank3WithIdentity() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8},
                    new Rank3(2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = ShapeOps.permute(input, 0, 1, 2)) {

                assertArrayEquals(new int[]{2, 2, 2}, output.dimensions());
                assertArrayEquals(input.underlying().toFloatArray(),
                        output.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("permute rank3 swaps first two dimensions")
        void permuteRank3SwapsFirstTwoDimensions() {
            // Input [2, 3, 4] -> Output [3, 2, 4] with perm [1, 0, 2]
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                // Set some values to verify permutation
                input.underlying().data().setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,
                        0 * 3 * 4 + 1 * 4 + 2, 42.0f);

                try (TypedTensor<Rank3, F32, Cpu> output = ShapeOps.permute(input, 1, 0, 2)) {
                    assertArrayEquals(new int[]{3, 2, 4}, output.dimensions());

                    // Value at input[0, 1, 2] should be at output[1, 0, 2]
                    float val = output.underlying().data().getAtIndex(
                            java.lang.foreign.ValueLayout.JAVA_FLOAT,
                            1 * 2 * 4 + 0 * 4 + 2);
                    assertEquals(42.0f, val, EPSILON);
                }
            }
        }

        @Test
        @DisplayName("permute rank4 for attention reshape")
        void permuteRank4ForAttentionReshape() {
            // Common attention permutation: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
            // Permutation: [0, 2, 1, 3]
            try (TypedTensor<Rank4, F32, Cpu> input = TypedTensor.zeros(
                    new Rank4(2, 8, 4, 16), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> output = ShapeOps.permuteRank4(input, 0, 2, 1, 3)) {

                assertArrayEquals(new int[]{2, 4, 8, 16}, output.dimensions());
            }
        }

        @Test
        @DisplayName("permute rejects invalid permutation length")
        void permuteRejectsInvalidPermutationLength() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ShapeOps.permute(input, 0, 1));  // Only 2 elements instead of 3
            }
        }

        @Test
        @DisplayName("permute rejects duplicate indices")
        void permuteRejectsDuplicateIndices() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ShapeOps.permute(input, 0, 0, 2));  // Duplicate 0
            }
        }

        @Test
        @DisplayName("permute rejects out of range indices")
        void permuteRejectsOutOfRangeIndices() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ShapeOps.permute(input, 0, 1, 3));  // 3 is out of range
            }
        }

        @Test
        @DisplayName("permute works with F64")
        void permuteWorksWithF64() {
            try (TypedTensor<Rank3, F64, Cpu> input = TypedTensor.fromDoubleArray(
                    new double[]{1, 2, 3, 4, 5, 6, 7, 8},
                    new Rank3(2, 2, 2), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F64, Cpu> output = ShapeOps.permute(input, 2, 1, 0)) {

                assertArrayEquals(new int[]{2, 2, 2}, output.dimensions());
            }
        }
    }

    @Nested
    @DisplayName("Squeeze Operations")
    class SqueezeTests {

        @Test
        @DisplayName("squeeze removes dimension of size 1 (axis 0)")
        void squeezeRemovesDimensionAxis0() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Matrix(1, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = ShapeOps.squeeze(mat, 0)) {

                assertEquals(3, vec.shapeType().length());
                assertArrayEquals(new float[]{1, 2, 3}, vec.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("squeeze removes dimension of size 1 (axis 1)")
        void squeezeRemovesDimensionAxis1() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Matrix(3, 1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vec = ShapeOps.squeeze(mat, 1)) {

                assertEquals(3, vec.shapeType().length());
                assertArrayEquals(new float[]{1, 2, 3}, vec.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("squeeze rejects non-singleton dimension")
        void squeezeRejectsNonSingletonDimension() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.zeros(
                    new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ShapeOps.squeeze(mat, 0));
            }
        }
    }

    @Nested
    @DisplayName("Unsqueeze Operations")
    class UnsqueezeTests {

        @Test
        @DisplayName("unsqueeze vector to row matrix")
        void unsqueezeVectorToRowMatrix() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> mat = ShapeOps.unsqueeze(vec, 0)) {

                assertArrayEquals(new int[]{1, 3}, mat.dimensions());
                assertArrayEquals(new float[]{1, 2, 3}, mat.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("unsqueeze vector to column matrix")
        void unsqueezeVectorToColumnMatrix() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> mat = ShapeOps.unsqueeze(vec, 1)) {

                assertArrayEquals(new int[]{3, 1}, mat.dimensions());
            }
        }

        @Test
        @DisplayName("unsqueezeToRank3 adds batch dimension")
        void unsqueezeToRank3AddsBatchDimension() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.zeros(
                    new Matrix(4, 5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> rank3 = ShapeOps.unsqueezeToRank3(mat, 0)) {

                assertArrayEquals(new int[]{1, 4, 5}, rank3.dimensions());
            }
        }

        @Test
        @DisplayName("unsqueezeToRank4 adds head dimension")
        void unsqueezeToRank4AddsHeadDimension() {
            try (TypedTensor<Rank3, F32, Cpu> rank3 = TypedTensor.zeros(
                    new Rank3(2, 8, 64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> rank4 = ShapeOps.unsqueezeToRank4(rank3, 1)) {

                assertArrayEquals(new int[]{2, 1, 8, 64}, rank4.dimensions());
            }
        }
    }

    @Nested
    @DisplayName("Broadcast Operations")
    class BroadcastTests {

        @Test
        @DisplayName("broadcast vector as rows")
        void broadcastVectorAsRows() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> mat = ShapeOps.broadcast(vec, 4, 0)) {

                assertArrayEquals(new int[]{4, 3}, mat.dimensions());

                // Each row should be [1, 2, 3]
                float[] data = mat.underlying().toFloatArray();
                for (int r = 0; r < 4; r++) {
                    assertEquals(1.0f, data[r * 3], EPSILON);
                    assertEquals(2.0f, data[r * 3 + 1], EPSILON);
                    assertEquals(3.0f, data[r * 3 + 2], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("broadcast vector as columns")
        void broadcastVectorAsColumns() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> mat = ShapeOps.broadcast(vec, 4, 1)) {

                assertArrayEquals(new int[]{3, 4}, mat.dimensions());

                // Each column should be [1, 2, 3]
                float[] data = mat.underlying().toFloatArray();
                for (int c = 0; c < 4; c++) {
                    assertEquals(1.0f, data[0 * 4 + c], EPSILON);
                    assertEquals(2.0f, data[1 * 4 + c], EPSILON);
                    assertEquals(3.0f, data[2 * 4 + c], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("expandToRank3 expands matrix to batch")
        void expandToRank3ExpandsMatrixToBatch() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> rank3 = ShapeOps.expandToRank3(mat, new Rank3(4, 2, 3))) {

                assertArrayEquals(new int[]{4, 2, 3}, rank3.dimensions());

                // Each batch should have the same values
                float[] data = rank3.underlying().toFloatArray();
                for (int b = 0; b < 4; b++) {
                    int offset = b * 6;
                    assertEquals(1.0f, data[offset], EPSILON);
                    assertEquals(2.0f, data[offset + 1], EPSILON);
                    assertEquals(3.0f, data[offset + 2], EPSILON);
                    assertEquals(4.0f, data[offset + 3], EPSILON);
                    assertEquals(5.0f, data[offset + 4], EPSILON);
                    assertEquals(6.0f, data[offset + 5], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("expandToRank3 rejects shape mismatch")
        void expandToRank3RejectsShapeMismatch() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.zeros(
                    new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ShapeOps.expandToRank3(mat, new Rank3(4, 3, 2)));  // 3, 2 doesn't match 2, 3
            }
        }
    }

    @Nested
    @DisplayName("Round-trip Operations")
    class RoundTripTests {

        @Test
        @DisplayName("reshape then reshape back preserves data")
        void reshapeRoundTrip() {
            try (TypedTensor<Matrix, F32, Cpu> original = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> reshaped = ShapeOps.reshape(original, new Rank3(2, 2, 3));
                 TypedTensor<Matrix, F32, Cpu> restored = ShapeOps.reshape(reshaped, new Matrix(3, 4))) {

                assertArrayEquals(original.underlying().toFloatArray(),
                        restored.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("squeeze then unsqueeze preserves data")
        void squeezeUnsqueezeRoundTrip() {
            try (TypedTensor<Matrix, F32, Cpu> original = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Matrix(1, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> squeezed = ShapeOps.squeeze(original, 0);
                 TypedTensor<Matrix, F32, Cpu> unsqueezed = ShapeOps.unsqueeze(squeezed, 0)) {

                assertArrayEquals(original.dimensions(), unsqueezed.dimensions());
                assertArrayEquals(original.underlying().toFloatArray(),
                        unsqueezed.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("double permute returns to original")
        void doublePermuteRoundTrip() {
            try (TypedTensor<Rank3, F32, Cpu> original = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8},
                    new Rank3(2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> permuted = ShapeOps.permute(original, 2, 0, 1);
                 TypedTensor<Rank3, F32, Cpu> restored = ShapeOps.permute(permuted, 1, 2, 0)) {

                assertArrayEquals(original.dimensions(), restored.dimensions());
                assertArrayEquals(original.underlying().toFloatArray(),
                        restored.underlying().toFloatArray(), EPSILON);
            }
        }
    }
}
