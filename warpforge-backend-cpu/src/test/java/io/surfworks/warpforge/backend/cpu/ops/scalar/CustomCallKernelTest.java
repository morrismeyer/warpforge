package io.surfworks.warpforge.backend.cpu.ops.scalar;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScalarType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * Tests for CustomCallKernel transformer operation handlers.
 */
@DisplayName("CustomCallKernel")
class CustomCallKernelTest {

    private static final float EPSILON = 1e-4f;
    private CustomCallKernel kernel;

    @BeforeEach
    void setUp() {
        kernel = new CustomCallKernel();
    }

    private CustomCallOp createCustomCall(String target, int... outputShape) {
        List<Integer> shapeList = Arrays.stream(outputShape).boxed().collect(Collectors.toList());
        TensorType resultType = new TensorType(shapeList, ScalarType.F32);
        Value result = new Value("%result", resultType);
        return new CustomCallOp(result, target, List.of(), resultType);
    }

    @Nested
    @DisplayName("LayerNorm")
    class LayerNormTests {

        @Test
        @DisplayName("layer_norm normalizes to zero mean and unit variance")
        void layerNormNormalizesToZeroMeanUnitVariance() {
            // Input: [2, 4] tensor
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8};
            try (Tensor input = Tensor.fromFloatArray(data, 2, 4)) {
                CustomCallOp op = createCustomCall("layer_norm", new int[]{2, 4});

                List<Tensor> results = kernel.execute(op, List.of(input));
                assertEquals(1, results.size());

                try (Tensor output = results.get(0)) {
                    assertArrayEquals(new int[]{2, 4}, output.shape());

                    float[] result = output.toFloatArray();

                    // Check each row has approximately zero mean
                    for (int row = 0; row < 2; row++) {
                        float sum = 0;
                        for (int col = 0; col < 4; col++) {
                            sum += result[row * 4 + col];
                        }
                        float mean = sum / 4;
                        assertEquals(0.0f, mean, EPSILON, "Row " + row + " mean should be ~0");
                    }

                    // Check each row has approximately unit variance
                    for (int row = 0; row < 2; row++) {
                        float varSum = 0;
                        for (int col = 0; col < 4; col++) {
                            float val = result[row * 4 + col];
                            varSum += val * val;
                        }
                        float variance = varSum / 4;
                        assertEquals(1.0f, variance, EPSILON, "Row " + row + " variance should be ~1");
                    }
                }
            }
        }

        @Test
        @DisplayName("layer_norm with weight and bias")
        void layerNormWithWeightAndBias() {
            float[] data = {0, 1, 2, 3};  // Will normalize to [-1.34, -0.45, 0.45, 1.34] approx
            float[] weight = {2, 2, 2, 2};  // Scale by 2
            float[] bias = {1, 1, 1, 1};    // Shift by 1

            try (Tensor input = Tensor.fromFloatArray(data, 1, 4);
                 Tensor gamma = Tensor.fromFloatArray(weight, 4);
                 Tensor beta = Tensor.fromFloatArray(bias, 4)) {

                CustomCallOp op = createCustomCall("layer_norm", new int[]{1, 4});

                List<Tensor> results = kernel.execute(op, List.of(input, gamma, beta));

                try (Tensor output = results.get(0)) {
                    float[] result = output.toFloatArray();

                    // Mean of normalized should be ~0, after scale and shift the mean should be ~1
                    float sum = 0;
                    for (float v : result) sum += v;
                    float mean = sum / 4;
                    assertEquals(1.0f, mean, EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("Softmax")
    class SoftmaxTests {

        @Test
        @DisplayName("softmax sums to 1")
        void softmaxSumsToOne() {
            float[] data = {1, 2, 3, 4};
            try (Tensor input = Tensor.fromFloatArray(data, 1, 4)) {
                CustomCallOp op = createCustomCall("softmax", new int[]{1, 4});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    float[] result = output.toFloatArray();

                    float sum = 0;
                    for (float v : result) sum += v;
                    assertEquals(1.0f, sum, EPSILON);
                }
            }
        }

        @Test
        @DisplayName("softmax is monotonic")
        void softmaxIsMonotonic() {
            float[] data = {1, 2, 3, 4};
            try (Tensor input = Tensor.fromFloatArray(data, 4)) {
                CustomCallOp op = createCustomCall("softmax", new int[]{4});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    float[] result = output.toFloatArray();

                    // Softmax should preserve relative ordering
                    assertTrue(result[0] < result[1]);
                    assertTrue(result[1] < result[2]);
                    assertTrue(result[2] < result[3]);
                }
            }
        }

        @Test
        @DisplayName("softmax is numerically stable with large values")
        void softmaxIsNumericallyStable() {
            // Large values that would overflow without max subtraction
            float[] data = {1000, 1001, 1002, 1003};
            try (Tensor input = Tensor.fromFloatArray(data, 4)) {
                CustomCallOp op = createCustomCall("softmax", new int[]{4});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    float[] result = output.toFloatArray();

                    // Should not produce NaN or Inf
                    for (float v : result) {
                        assertTrue(Float.isFinite(v), "Softmax should be finite");
                        assertTrue(v >= 0 && v <= 1, "Softmax should be in [0,1]");
                    }

                    float sum = 0;
                    for (float v : result) sum += v;
                    assertEquals(1.0f, sum, EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("GELU")
    class GeluTests {

        @Test
        @DisplayName("gelu at zero is zero")
        void geluAtZeroIsZero() {
            float[] data = {0};
            try (Tensor input = Tensor.fromFloatArray(data, 1)) {
                CustomCallOp op = createCustomCall("gelu", new int[]{1});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    assertEquals(0.0f, output.toFloatArray()[0], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("gelu approaches x for large positive x")
        void geluApproachesXForLargePositive() {
            float[] data = {5.0f};
            try (Tensor input = Tensor.fromFloatArray(data, 1)) {
                CustomCallOp op = createCustomCall("gelu", new int[]{1});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    // GELU(5) ≈ 5 (approaches identity for large positive)
                    assertTrue(output.toFloatArray()[0] > 4.9f);
                }
            }
        }

        @Test
        @DisplayName("gelu approaches 0 for large negative x")
        void geluApproachesZeroForLargeNegative() {
            float[] data = {-5.0f};
            try (Tensor input = Tensor.fromFloatArray(data, 1)) {
                CustomCallOp op = createCustomCall("gelu", new int[]{1});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    // GELU(-5) ≈ 0
                    assertTrue(Math.abs(output.toFloatArray()[0]) < 0.01f);
                }
            }
        }
    }

    @Nested
    @DisplayName("SiLU")
    class SiluTests {

        @Test
        @DisplayName("silu at zero is zero")
        void siluAtZeroIsZero() {
            float[] data = {0};
            try (Tensor input = Tensor.fromFloatArray(data, 1)) {
                CustomCallOp op = createCustomCall("silu", new int[]{1});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    assertEquals(0.0f, output.toFloatArray()[0], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("silu is x * sigmoid(x)")
        void siluIsXTimesSigmoid() {
            float[] data = {1.0f, 2.0f, -1.0f};
            try (Tensor input = Tensor.fromFloatArray(data, 3)) {
                CustomCallOp op = createCustomCall("silu", new int[]{3});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    float[] result = output.toFloatArray();

                    for (int i = 0; i < 3; i++) {
                        float x = data[i];
                        float expected = x / (1 + (float) Math.exp(-x));
                        assertEquals(expected, result[i], EPSILON);
                    }
                }
            }
        }
    }

    @Nested
    @DisplayName("RMSNorm")
    class RmsNormTests {

        @Test
        @DisplayName("rms_norm normalizes by RMS")
        void rmsNormNormalizesByRms() {
            float[] data = {3, 4};  // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.54
            try (Tensor input = Tensor.fromFloatArray(data, 1, 2)) {
                CustomCallOp op = createCustomCall("rms_norm", new int[]{1, 2});

                List<Tensor> results = kernel.execute(op, List.of(input));

                try (Tensor output = results.get(0)) {
                    float[] result = output.toFloatArray();

                    // After RMS normalization, RMS should be ~1
                    float sumSq = 0;
                    for (float v : result) sumSq += v * v;
                    float rms = (float) Math.sqrt(sumSq / 2);
                    assertEquals(1.0f, rms, EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("BatchNorm")
    class BatchNormTests {

        @Test
        @DisplayName("batch_norm normalizes with running stats")
        void batchNormNormalizesWithRunningStats() {
            // Input: [1, 2, 2, 2] - batch=1, channels=2, spatial=2x2
            float[] data = {
                1, 2, 3, 4,   // Channel 0
                5, 6, 7, 8    // Channel 1
            };
            float[] weight = {1, 1};
            float[] bias = {0, 0};
            float[] mean = {2.5f, 6.5f};  // Mean per channel
            float[] var = {1.25f, 1.25f}; // Variance per channel

            try (Tensor input = Tensor.fromFloatArray(data, 1, 2, 2, 2);
                 Tensor gamma = Tensor.fromFloatArray(weight, 2);
                 Tensor beta = Tensor.fromFloatArray(bias, 2);
                 Tensor runningMean = Tensor.fromFloatArray(mean, 2);
                 Tensor runningVar = Tensor.fromFloatArray(var, 2)) {

                CustomCallOp op = createCustomCall("batch_norm", new int[]{1, 2, 2, 2});

                List<Tensor> results = kernel.execute(op,
                    List.of(input, gamma, beta, runningMean, runningVar));

                try (Tensor output = results.get(0)) {
                    assertArrayEquals(new int[]{1, 2, 2, 2}, output.shape());

                    // Normalized values should be centered around 0
                    float[] result = output.toFloatArray();
                    float sum = 0;
                    for (float v : result) sum += v;
                    // Mean should be close to 0 since we're centering
                    assertTrue(Math.abs(sum / 8) < 0.5f);
                }
            }
        }
    }
}
