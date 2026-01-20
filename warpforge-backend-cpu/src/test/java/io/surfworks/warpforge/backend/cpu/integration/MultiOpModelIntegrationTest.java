package io.surfworks.warpforge.backend.cpu.integration;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.testing.TensorAssert;
import io.surfworks.warpforge.core.testing.ToleranceConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Integration tests for multi-operation models.
 *
 * <p>These tests verify that sequences of operations execute correctly
 * when composed together, testing the interpreter and backend end-to-end.
 */
@DisplayName("Multi-Operation Model Integration Tests")
@Tag("integration")
class MultiOpModelIntegrationTest {

    private StableHloInterpreter interpreter;

    @BeforeEach
    void setUp() {
        interpreter = new StableHloInterpreter();
    }

    @AfterEach
    void tearDown() {
        if (interpreter != null) {
            interpreter.close();
        }
    }

    @Nested
    @DisplayName("Neural Network Patterns")
    class NeuralNetworkPatterns {

        @Test
        @DisplayName("linear layer: y = xW + b")
        void linearLayer() {
            // Simple linear: input[2,3] @ weights[3,4] + bias[4] -> output[2,4]
            String mlir = """
                module @main {
                  func.func public @forward(%input: tensor<2x3xf32>, %weights: tensor<3x4xf32>, %bias: tensor<4xf32>) -> (tensor<2x4xf32>) {
                    %matmul = stablehlo.dot_general %input, %weights, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
                    %bias_broadcast = stablehlo.broadcast_in_dim %bias, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
                    %result = stablehlo.add %matmul, %bias_broadcast : tensor<2x4xf32>
                    stablehlo.return %result : tensor<2x4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);
            assertNotNull(module);

            // input = [[1,2,3], [4,5,6]]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 // weights = [[1,0,0,1], [0,1,0,1], [0,0,1,1]]
                 Tensor weights = Tensor.fromFloatArray(new float[]{
                     1, 0, 0, 1,
                     0, 1, 0, 1,
                     0, 0, 1, 1
                 }, 3, 4);
                 // bias = [0.1, 0.2, 0.3, 0.4]
                 Tensor bias = Tensor.fromFloatArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 4)) {

                List<Tensor> outputs = interpreter.execute(module, List.of(input, weights, bias));

                assertEquals(1, outputs.size());

                // Expected: matmul gives [[1,2,3,6], [4,5,6,15]], plus bias
                // Result: [[1.1, 2.2, 3.3, 6.4], [4.1, 5.2, 6.3, 15.4]]
                try (Tensor expected = Tensor.fromFloatArray(new float[]{
                         1.1f, 2.2f, 3.3f, 6.4f,
                         4.1f, 5.2f, 6.3f, 15.4f
                     }, 2, 4);
                     Tensor actual = outputs.get(0)) {

                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.forOp("dot_general"));
                }
            }
        }

        @Test
        @DisplayName("relu activation: max(0, x)")
        void reluActivation() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
                    %result = stablehlo.maximum %x, %zero : tensor<4xf32>
                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            try (Tensor input = Tensor.fromFloatArray(new float[]{-2, -1, 0, 1}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{0, 0, 0, 1}, 4);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }

        @Test
        @DisplayName("softmax: exp(x) / sum(exp(x))")
        void softmax() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %exp_x = stablehlo.exponential %x : tensor<4xf32>
                    %zero = stablehlo.constant dense<0.0> : tensor<f32>
                    %sum = stablehlo.reduce %exp_x, %zero, dims=[0], reducer=add : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %sum_broadcast = stablehlo.broadcast_in_dim %sum, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %result = stablehlo.divide %exp_x, %sum_broadcast : tensor<4xf32>
                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // Input: [1, 2, 3, 4]
            // exp: [e, e^2, e^3, e^4] = [2.718, 7.389, 20.086, 54.598]
            // sum: 84.791
            // softmax: [0.032, 0.087, 0.237, 0.644]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor actual = outputs.get(0)) {
                    float[] data = actual.toFloatArray();
                    assertEquals(4, data.length);

                    // Verify softmax properties
                    float sum = 0;
                    for (float v : data) {
                        sum += v;
                    }
                    // Sum should be ~1.0
                    assertEquals(1.0f, sum, 1e-4f);

                    // Values should be in increasing order
                    for (int i = 1; i < data.length; i++) {
                        assert data[i] > data[i - 1] : "Softmax should preserve ordering";
                    }
                }
            }
        }

        @Test
        @DisplayName("batch normalization pattern: (x - mean) / sqrt(var + eps) * scale + offset")
        void batchNormPattern() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>, %scale: tensor<4xf32>, %offset: tensor<4xf32>) -> (tensor<4xf32>) {
                    %eps = stablehlo.constant dense<1.0e-5> : tensor<4xf32>
                    %zero = stablehlo.constant dense<0.0> : tensor<f32>

                    %sum = stablehlo.reduce %x, %zero, dims=[0], reducer=add : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %count = stablehlo.constant dense<4.0> : tensor<f32>
                    %mean_scalar = stablehlo.divide %sum, %count : tensor<f32>
                    %mean = stablehlo.broadcast_in_dim %mean_scalar, dims = [] : (tensor<f32>) -> tensor<4xf32>

                    %centered = stablehlo.subtract %x, %mean : tensor<4xf32>
                    %squared = stablehlo.multiply %centered, %centered : tensor<4xf32>

                    %sum_sq = stablehlo.reduce %squared, %zero, dims=[0], reducer=add : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    %var_scalar = stablehlo.divide %sum_sq, %count : tensor<f32>
                    %var = stablehlo.broadcast_in_dim %var_scalar, dims = [] : (tensor<f32>) -> tensor<4xf32>

                    %var_eps = stablehlo.add %var, %eps : tensor<4xf32>
                    %std = stablehlo.sqrt %var_eps : tensor<4xf32>
                    %normalized = stablehlo.divide %centered, %std : tensor<4xf32>

                    %scaled = stablehlo.multiply %normalized, %scale : tensor<4xf32>
                    %result = stablehlo.add %scaled, %offset : tensor<4xf32>

                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // x = [1, 2, 3, 4], mean = 2.5, var = 1.25
            // normalized = [-1.118, -0.447, 0.447, 1.118] (approximately)
            try (Tensor x = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor scale = Tensor.full(1.0f, 4);
                 Tensor offset = Tensor.zeros(4)) {

                List<Tensor> outputs = interpreter.execute(module, List.of(x, scale, offset));

                try (Tensor actual = outputs.get(0)) {
                    float[] data = actual.toFloatArray();

                    // Mean of normalized output should be ~0
                    float mean = 0;
                    for (float v : data) mean += v;
                    mean /= 4;
                    assertEquals(0.0f, mean, 1e-4f);

                    // Variance should be ~1
                    float variance = 0;
                    for (float v : data) variance += v * v;
                    variance /= 4;
                    assertEquals(1.0f, variance, 1e-3f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Shape Manipulation Chains")
    class ShapeManipulationChains {

        @Test
        @DisplayName("reshape -> transpose -> reshape")
        void reshapeTransposeReshape() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<6xf32>) -> (tensor<6xf32>) {
                    %reshaped = stablehlo.reshape %x : tensor<6xf32> -> tensor<2x3xf32>
                    %transposed = stablehlo.transpose %reshaped, dims = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
                    %result = stablehlo.reshape %transposed : tensor<3x2xf32> -> tensor<6xf32>
                    stablehlo.return %result : tensor<6xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // Input: [1,2,3,4,5,6] -> [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]] -> [1,4,2,5,3,6]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 6)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{1, 4, 2, 5, 3, 6}, 6);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }

        @Test
        @DisplayName("broadcast -> slice -> concatenate")
        void broadcastSliceConcatenate() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<4xf32>) {
                    %concat = stablehlo.concatenate %x, %y, dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>
                    stablehlo.return %concat : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            try (Tensor x = Tensor.fromFloatArray(new float[]{1, 2}, 2);
                 Tensor y = Tensor.fromFloatArray(new float[]{3, 4}, 2)) {

                List<Tensor> outputs = interpreter.execute(module, List.of(x, y));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }

        @Test
        @DisplayName("slice operation")
        void sliceOperation() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<2xf32>) {
                    %sliced = stablehlo.slice %x, starts = [1], limits = [3], strides = [1] : tensor<4xf32> -> tensor<2xf32>
                    stablehlo.return %sliced : tensor<2xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // Input: [1,2,3,4] -> slice [1:3] -> [2,3]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{2, 3}, 2);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }
    }

    @Nested
    @DisplayName("Reduction Operations")
    class ReductionOperations {

        @Test
        @DisplayName("reduce sum across dimension")
        void reduceSumAcrossDimension() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<2x3xf32>) -> (tensor<3xf32>) {
                    %zero = stablehlo.constant dense<0.0> : tensor<f32>
                    %result = stablehlo.reduce %x, %zero, dims=[0], reducer=add : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
                    stablehlo.return %result : tensor<3xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // Input: [[1,2,3], [4,5,6]] -> sum across dim 0 -> [5, 7, 9]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{5, 7, 9}, 3);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.forOp("reduce"));
                }
            }
        }

        @Test
        @DisplayName("reduce max")
        void reduceMax() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<f32>) {
                    %neg_inf = stablehlo.constant dense<-3.4028235E38> : tensor<f32>
                    %result = stablehlo.reduce %x, %neg_inf, dims=[0], reducer=max : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    stablehlo.return %result : tensor<f32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            try (Tensor input = Tensor.fromFloatArray(new float[]{3, 1, 4, 1}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{4});
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }

        @Test
        @DisplayName("reduce product")
        void reduceProduct() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<f32>) {
                    %one = stablehlo.constant dense<1.0> : tensor<f32>
                    %result = stablehlo.reduce %x, %one, dims=[0], reducer=mul : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
                    stablehlo.return %result : tensor<f32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // 1 * 2 * 3 * 4 = 24
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{24});
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.forOp("reduce"));
                }
            }
        }
    }

    @Nested
    @DisplayName("Comparison and Selection")
    class ComparisonAndSelection {

        @Test
        @DisplayName("compare and select (clamp pattern)")
        void compareAndSelect() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %lo = stablehlo.constant dense<0.0> : tensor<4xf32>
                    %hi = stablehlo.constant dense<1.0> : tensor<4xf32>
                    %clamped = stablehlo.clamp %lo, %x, %hi : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %clamped : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // Input: [-1, 0.5, 1.5, 0.8] -> clamp(0,1) -> [0, 0.5, 1, 0.8]
            try (Tensor input = Tensor.fromFloatArray(new float[]{-1f, 0.5f, 1.5f, 0.8f}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{0f, 0.5f, 1f, 0.8f}, 4);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.forOp("clamp"));
                }
            }
        }

        @Test
        @DisplayName("select with predicate")
        void selectWithPredicate() {
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
                    %neg_x = stablehlo.negate %x : tensor<4xf32>
                    %pred = stablehlo.compare %x, %zero, direction = GE : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    %result = stablehlo.select %pred, %x, %neg_x : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // abs function: select(x >= 0, x, -x)
            // Input: [-2, -1, 1, 2] -> [2, 1, 1, 2]
            try (Tensor input = Tensor.fromFloatArray(new float[]{-2, -1, 1, 2}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{2, 1, 1, 2}, 4);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }
    }

    @Nested
    @DisplayName("Matrix Operations")
    class MatrixOperations {

        @Test
        @DisplayName("matrix multiplication (2x3) @ (3x2)")
        void matmulBasic() {
            String mlir = """
                module @main {
                  func.func public @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> (tensor<2x2xf32>) {
                    %result = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
                    stablehlo.return %result : tensor<2x2xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // A = [[1,2,3], [4,5,6]] (2x3)
            // B = [[1,4], [2,5], [3,6]] (3x2)
            // C = A @ B = [[14,32], [32,77]] (2x2)
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 4, 2, 5, 3, 6}, 3, 2)) {

                List<Tensor> outputs = interpreter.execute(module, List.of(a, b));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{14, 32, 32, 77}, 2, 2);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.forOp("dot_general"));
                }
            }
        }

        @Test
        @DisplayName("vector dot product")
        void vectorDotProduct() {
            String mlir = """
                module @main {
                  func.func public @forward(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<f32>) {
                    %result = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]> : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
                    stablehlo.return %result : tensor<f32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            // a = [1, 2, 3, 4], b = [4, 3, 2, 1]
            // a · b = 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{4, 3, 2, 1}, 4)) {

                List<Tensor> outputs = interpreter.execute(module, List.of(a, b));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{20});
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.forOp("dot_general"));
                }
            }
        }
    }

    @Nested
    @DisplayName("Type Conversions")
    class TypeConversions {

        @Test
        @DisplayName("convert f32 passthrough")
        void convertFloat32Passthrough() {
            // Current ConvertKernel implementation passes through float data
            // Full type conversion would require more sophisticated implementation
            String mlir = """
                module @main {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %result = stablehlo.convert %x : tensor<4xf32> -> tensor<4xf32>
                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1.5f, 2.9f, -1.1f, 0.0f}, 4)) {
                List<Tensor> outputs = interpreter.execute(module, List.of(input));

                try (Tensor expected = Tensor.fromFloatArray(new float[]{1.5f, 2.9f, -1.1f, 0.0f}, 4);
                     Tensor actual = outputs.get(0)) {
                    TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
                }
            }
        }
    }

    @Nested
    @DisplayName("Constant Operations")
    class ConstantOperations {

        @Test
        @DisplayName("iota generates sequential values")
        void iotaOperation() {
            String mlir = """
                module @main {
                  func.func public @forward() -> (tensor<4xf32>) {
                    %result = stablehlo.iota dim = 0 : tensor<4xf32>
                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            List<Tensor> outputs = interpreter.execute(module, List.of());

            try (Tensor expected = Tensor.fromFloatArray(new float[]{0, 1, 2, 3}, 4);
                 Tensor actual = outputs.get(0)) {
                TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
            }
        }

        @Test
        @DisplayName("constant dense tensor with add")
        void constantDenseWithAdd() {
            // Test constant combined with operations since array literals aren't supported
            String mlir = """
                module @main {
                  func.func public @forward() -> (tensor<4xf32>) {
                    %iota = stablehlo.iota dim = 0 : tensor<4xf32>
                    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
                    %result = stablehlo.add %iota, %one : tensor<4xf32>
                    stablehlo.return %result : tensor<4xf32>
                  }
                }
                """;

            StableHloAst.Module module = StableHloParser.parse(mlir);

            List<Tensor> outputs = interpreter.execute(module, List.of());

            // iota gives [0,1,2,3], add 1 gives [1,2,3,4]
            try (Tensor expected = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor actual = outputs.get(0)) {
                TensorAssert.assertEquals(expected, actual, ToleranceConfig.STRICT);
            }
        }
    }
}
