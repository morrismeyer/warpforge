package io.surfworks.warpforge.core.tensor.typed.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
 * Tests for ActivationOps activation function operations.
 */
@DisplayName("ActivationOps")
class ActivationOpsTest {

    private static final float EPSILON = 1e-4f;
    private static final double EPSILON_D = 1e-8;

    @Nested
    @DisplayName("Softmax Operations")
    class SoftmaxTests {

        @Test
        @DisplayName("softmax on vector sums to 1")
        void softmaxOnVectorSumsToOne() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.softmax(input)) {

                float[] data = output.underlying().toFloatArray();

                // Sum should be 1
                float sum = 0;
                for (float v : data) {
                    sum += v;
                }
                assertEquals(1.0f, sum, EPSILON);

                // All values should be positive
                for (float v : data) {
                    assertTrue(v > 0);
                }

                // Larger input -> larger output
                assertTrue(data[2] > data[1]);
                assertTrue(data[1] > data[0]);
            }
        }

        @Test
        @DisplayName("softmax is stable for large values")
        void softmaxIsStableForLargeValues() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1000, 1001, 1002}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.softmax(input)) {

                float[] data = output.underlying().toFloatArray();

                // Should not be NaN or Inf
                for (float v : data) {
                    assertTrue(Float.isFinite(v), "Softmax should be numerically stable");
                }

                // Sum should still be 1
                float sum = 0;
                for (float v : data) {
                    sum += v;
                }
                assertEquals(1.0f, sum, EPSILON);
            }
        }

        @Test
        @DisplayName("softmax on uniform input produces uniform output")
        void softmaxOnUniformInputProducesUniformOutput() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{5, 5, 5, 5}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.softmax(input)) {

                float[] data = output.underlying().toFloatArray();

                // Each value should be 0.25
                for (float v : data) {
                    assertEquals(0.25f, v, EPSILON);
                }
            }
        }

        @Test
        @DisplayName("softmaxRows applies to each row independently")
        void softmaxRowsAppliesToEachRowIndependently() {
            try (TypedTensor<Matrix, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> output = ActivationOps.softmaxRows(input)) {

                float[] data = output.underlying().toFloatArray();

                // Each row should sum to 1
                float row0Sum = data[0] + data[1] + data[2];
                float row1Sum = data[3] + data[4] + data[5];

                assertEquals(1.0f, row0Sum, EPSILON);
                assertEquals(1.0f, row1Sum, EPSILON);
            }
        }

        @Test
        @DisplayName("softmaxRank3 applies along last dimension")
        void softmaxRank3AppliesAlongLastDimension() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8},
                    new Rank3(2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = ActivationOps.softmaxRank3(input)) {

                float[] data = output.underlying().toFloatArray();

                // Each [hidden] slice should sum to 1
                for (int i = 0; i < 4; i++) {
                    float sum = data[i * 2] + data[i * 2 + 1];
                    assertEquals(1.0f, sum, EPSILON);
                }
            }
        }

        @Test
        @DisplayName("softmaxRank4 for attention scores")
        void softmaxRank4ForAttentionScores() {
            // Shape: [batch=1, heads=2, seq_q=2, seq_k=3]
            float[] input = new float[12];
            for (int i = 0; i < 12; i++) {
                input[i] = i;
            }

            try (TypedTensor<Rank4, F32, Cpu> scores = TypedTensor.fromFloatArray(
                    input, new Rank4(1, 2, 2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> probs = ActivationOps.softmaxRank4(scores)) {

                assertArrayEquals(new int[]{1, 2, 2, 3}, probs.dimensions());

                float[] data = probs.underlying().toFloatArray();

                // Each [seq_k] slice should sum to 1
                for (int i = 0; i < 4; i++) {
                    float sum = data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2];
                    assertEquals(1.0f, sum, EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("GELU Operations")
    class GeluTests {

        @Test
        @DisplayName("gelu at zero equals zero")
        void geluAtZeroEqualsZero() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{0}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.gelu(input)) {

                assertEquals(0, output.underlying().toFloatArray()[0], EPSILON);
            }
        }

        @Test
        @DisplayName("gelu is approximately x for large positive x")
        void geluIsApproximatelyXForLargePositiveX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{5.0f}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.gelu(input)) {

                // For large positive x, GELU(x) ≈ x
                float result = output.underlying().toFloatArray()[0];
                assertEquals(5.0f, result, 0.01f);
            }
        }

        @Test
        @DisplayName("gelu is approximately 0 for large negative x")
        void geluIsApproximatelyZeroForLargeNegativeX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-5.0f}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.gelu(input)) {

                // For large negative x, GELU(x) ≈ 0
                float result = output.underlying().toFloatArray()[0];
                assertEquals(0, result, 0.01f);
            }
        }

        @Test
        @DisplayName("gelu preserves shape")
        void geluPreservesShape() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 4, 8), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = ActivationOps.gelu(input)) {

                assertArrayEquals(new int[]{2, 4, 8}, output.dimensions());
            }
        }

        @Test
        @DisplayName("gelu matches expected values")
        void geluMatchesExpectedValues() {
            // Known GELU values from PyTorch
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-2, -1, 0, 1, 2}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.gelu(input)) {

                float[] result = output.underlying().toFloatArray();

                // Approximate expected values
                assertEquals(-0.0454f, result[0], 0.01f);  // GELU(-2)
                assertEquals(-0.1588f, result[1], 0.01f);  // GELU(-1)
                assertEquals(0, result[2], EPSILON);       // GELU(0)
                assertEquals(0.8412f, result[3], 0.01f);   // GELU(1)
                assertEquals(1.9546f, result[4], 0.01f);   // GELU(2)
            }
        }

        @Test
        @DisplayName("geluF64 works with double precision")
        void geluF64WorksWithDoublePrecision() {
            try (TypedTensor<Vector, F64, Cpu> input = TypedTensor.fromDoubleArray(
                    new double[]{-1, 0, 1}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> output = ActivationOps.geluF64(input)) {

                double[] result = output.underlying().toDoubleArray();
                assertEquals(0, result[1], EPSILON_D);  // GELU(0) = 0
            }
        }
    }

    @Nested
    @DisplayName("SiLU Operations")
    class SiluTests {

        @Test
        @DisplayName("silu at zero equals zero")
        void siluAtZeroEqualsZero() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{0}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.silu(input)) {

                assertEquals(0, output.underlying().toFloatArray()[0], EPSILON);
            }
        }

        @Test
        @DisplayName("silu is approximately x for large positive x")
        void siluIsApproximatelyXForLargePositiveX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{10.0f}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.silu(input)) {

                // For large positive x, sigmoid(x) ≈ 1, so SiLU(x) ≈ x
                float result = output.underlying().toFloatArray()[0];
                assertEquals(10.0f, result, 0.01f);
            }
        }

        @Test
        @DisplayName("silu is approximately 0 for large negative x")
        void siluIsApproximatelyZeroForLargeNegativeX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-10.0f}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.silu(input)) {

                // For large negative x, sigmoid(x) ≈ 0, so SiLU(x) ≈ 0
                float result = output.underlying().toFloatArray()[0];
                assertEquals(0, result, 0.01f);
            }
        }

        @Test
        @DisplayName("silu matches expected values")
        void siluMatchesExpectedValues() {
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-1, 0, 1}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.silu(input)) {

                float[] result = output.underlying().toFloatArray();

                // SiLU(-1) = -1 * sigmoid(-1) = -1 / (1 + e^1) ≈ -0.2689
                assertEquals(-0.2689f, result[0], 0.01f);

                // SiLU(0) = 0
                assertEquals(0, result[1], EPSILON);

                // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
                assertEquals(0.7311f, result[2], 0.01f);
            }
        }
    }

    @Nested
    @DisplayName("ReLU Operations")
    class ReluTests {

        @Test
        @DisplayName("relu zeros negative values")
        void reluZerosNegativeValues() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-2, -1, 0, 1, 2}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.relu(input)) {

                assertArrayEquals(new float[]{0, 0, 0, 1, 2},
                        output.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("relu preserves positive values")
        void reluPreservesPositiveValues() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{0.5f, 1.0f, 100.0f}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.relu(input)) {

                assertArrayEquals(new float[]{0.5f, 1.0f, 100.0f},
                        output.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("relu preserves shape")
        void reluPreservesShape() {
            try (TypedTensor<Matrix, F32, Cpu> input = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> output = ActivationOps.relu(input)) {

                assertArrayEquals(new int[]{3, 4}, output.dimensions());
            }
        }
    }

    @Nested
    @DisplayName("Sigmoid Operations")
    class SigmoidTests {

        @Test
        @DisplayName("sigmoid at zero equals 0.5")
        void sigmoidAtZeroEqualsHalf() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{0}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.sigmoid(input)) {

                assertEquals(0.5f, output.underlying().toFloatArray()[0], EPSILON);
            }
        }

        @Test
        @DisplayName("sigmoid approaches 1 for large positive x")
        void sigmoidApproachesOneForLargePositiveX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{10}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.sigmoid(input)) {

                float result = output.underlying().toFloatArray()[0];
                assertTrue(result > 0.99f);
                assertTrue(result <= 1.0f);
            }
        }

        @Test
        @DisplayName("sigmoid approaches 0 for large negative x")
        void sigmoidApproachesZeroForLargeNegativeX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-10}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.sigmoid(input)) {

                float result = output.underlying().toFloatArray()[0];
                assertTrue(result < 0.01f);
                assertTrue(result >= 0.0f);
            }
        }

        @Test
        @DisplayName("sigmoid output is between 0 and 1")
        void sigmoidOutputIsBetweenZeroAndOne() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-100, -10, -1, 0, 1, 10, 100},
                    new Vector(7), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.sigmoid(input)) {

                for (float v : output.underlying().toFloatArray()) {
                    assertTrue(v >= 0.0f && v <= 1.0f,
                            "Sigmoid output should be in [0, 1], got " + v);
                }
            }
        }
    }

    @Nested
    @DisplayName("Tanh Operations")
    class TanhTests {

        @Test
        @DisplayName("tanh at zero equals zero")
        void tanhAtZeroEqualsZero() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{0}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.tanh(input)) {

                assertEquals(0, output.underlying().toFloatArray()[0], EPSILON);
            }
        }

        @Test
        @DisplayName("tanh is antisymmetric")
        void tanhIsAntisymmetric() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-1, 1}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.tanh(input)) {

                float[] data = output.underlying().toFloatArray();
                assertEquals(-data[0], data[1], EPSILON);
            }
        }

        @Test
        @DisplayName("tanh output is between -1 and 1")
        void tanhOutputIsBetweenMinusOneAndOne() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{-100, -10, -1, 0, 1, 10, 100},
                    new Vector(7), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.tanh(input)) {

                for (float v : output.underlying().toFloatArray()) {
                    assertTrue(v >= -1.0f && v <= 1.0f,
                            "Tanh output should be in [-1, 1], got " + v);
                }
            }
        }

        @Test
        @DisplayName("tanh approaches 1 for large positive x")
        void tanhApproachesOneForLargePositiveX() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{10}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = ActivationOps.tanh(input)) {

                float result = output.underlying().toFloatArray()[0];
                assertTrue(result > 0.99f);
            }
        }
    }

    @Nested
    @DisplayName("Shape Preservation")
    class ShapePreservationTests {

        @Test
        @DisplayName("all activations preserve rank3 shape")
        void allActivationsPreserveRank3Shape() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 4, 8), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Rank3, F32, Cpu> gelu = ActivationOps.gelu(input);
                     TypedTensor<Rank3, F32, Cpu> silu = ActivationOps.silu(input);
                     TypedTensor<Rank3, F32, Cpu> relu = ActivationOps.relu(input);
                     TypedTensor<Rank3, F32, Cpu> sigmoid = ActivationOps.sigmoid(input);
                     TypedTensor<Rank3, F32, Cpu> tanh = ActivationOps.tanh(input)) {

                    int[] expectedShape = new int[]{2, 4, 8};
                    assertArrayEquals(expectedShape, gelu.dimensions());
                    assertArrayEquals(expectedShape, silu.dimensions());
                    assertArrayEquals(expectedShape, relu.dimensions());
                    assertArrayEquals(expectedShape, sigmoid.dimensions());
                    assertArrayEquals(expectedShape, tanh.dimensions());
                }
            }
        }

        @Test
        @DisplayName("all activations preserve matrix shape")
        void allActivationsPreserveMatrixShape() {
            try (TypedTensor<Matrix, F32, Cpu> input = TypedTensor.zeros(
                    new Matrix(5, 10), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Matrix, F32, Cpu> gelu = ActivationOps.gelu(input);
                     TypedTensor<Matrix, F32, Cpu> silu = ActivationOps.silu(input);
                     TypedTensor<Matrix, F32, Cpu> relu = ActivationOps.relu(input);
                     TypedTensor<Matrix, F32, Cpu> sigmoid = ActivationOps.sigmoid(input);
                     TypedTensor<Matrix, F32, Cpu> tanh = ActivationOps.tanh(input)) {

                    int[] expectedShape = new int[]{5, 10};
                    assertArrayEquals(expectedShape, gelu.dimensions());
                    assertArrayEquals(expectedShape, silu.dimensions());
                    assertArrayEquals(expectedShape, relu.dimensions());
                    assertArrayEquals(expectedShape, sigmoid.dimensions());
                    assertArrayEquals(expectedShape, tanh.dimensions());
                }
            }
        }
    }
}
