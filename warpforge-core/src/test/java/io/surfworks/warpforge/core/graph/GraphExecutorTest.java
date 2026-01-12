package io.surfworks.warpforge.core.graph;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.*;
import io.surfworks.warpforge.backend.cpu.CpuBackend;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GraphExecutorTest {

    private CpuBackend backend;
    private GraphExecutor executor;
    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;

    // Helper to create TensorType
    private static TensorType tensor(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), F32);
    }

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
        executor = new GraphExecutor(backend);
    }

    @AfterEach
    void tearDown() {
        backend.close();
    }

    @Nested
    @DisplayName("Identity Function")
    class IdentityTests {

        @Test
        void executeIdentity() {
            // func @identity(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            //   return %arg0 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> outputs = executor.execute(graph, input);

                assertEquals(1, outputs.size());
                // Identity returns the same tensor reference
                assertSame(input, outputs.getFirst());
            }
        }
    }

    @Nested
    @DisplayName("Unary Operations")
    class UnaryOpTests {

        @Test
        void executeNegate() {
            // func @negate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            //   %0 = stablehlo.negate %arg0 : tensor<4xf32>
            //   return %0 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value result = new Value("0", t4f32);

            Function func = new Function(
                "negate",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(
                    new NegateOp(result, arg0, t4f32),
                    new ReturnOp(List.of(result))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, -2, 3, -4}, 4)) {
                List<Tensor> outputs = executor.execute(graph, input);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {-1, 2, -3, 4};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void executeAbs() {
            // func @abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            //   %0 = stablehlo.abs %arg0 : tensor<4xf32>
            //   return %0 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value result = new Value("0", t4f32);

            Function func = new Function(
                "abs",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(
                    new AbsOp(result, arg0, t4f32),
                    new ReturnOp(List.of(result))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor input = Tensor.fromFloatArray(new float[]{-1, 2, -3, 4}, 4)) {
                List<Tensor> outputs = executor.execute(graph, input);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Binary Operations")
    class BinaryOpTests {

        @Test
        void executeAdd() {
            // func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            //   %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            //   return %0 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value arg1 = new Value("arg1", t4f32);
            Value result = new Value("0", t4f32);

            Function func = new Function(
                "add",
                List.of(
                    new Argument("arg0", t4f32),
                    new Argument("arg1", t4f32)
                ),
                List.of(t4f32),
                List.of(
                    new AddOp(result, arg0, arg1, t4f32),
                    new ReturnOp(List.of(result))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 4)) {

                List<Tensor> outputs = executor.execute(graph, a, b);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {6, 8, 10, 12};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void executeMultiply() {
            // func @mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            //   %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
            //   return %0 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value arg1 = new Value("arg1", t4f32);
            Value result = new Value("0", t4f32);

            Function func = new Function(
                "mul",
                List.of(
                    new Argument("arg0", t4f32),
                    new Argument("arg1", t4f32)
                ),
                List.of(t4f32),
                List.of(
                    new MultiplyOp(result, arg0, arg1, t4f32),
                    new ReturnOp(List.of(result))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{2, 3, 4, 5}, 4)) {

                List<Tensor> outputs = executor.execute(graph, a, b);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {2, 6, 12, 20};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Chained Operations")
    class ChainedOpTests {

        @Test
        void executeNegateAbs() {
            // func @chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            //   %0 = stablehlo.negate %arg0 : tensor<4xf32>
            //   %1 = stablehlo.abs %0 : tensor<4xf32>
            //   return %1 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value v0 = new Value("0", t4f32);
            Value v1 = new Value("1", t4f32);

            Function func = new Function(
                "chain",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(
                    new NegateOp(v0, arg0, t4f32),
                    new AbsOp(v1, v0, t4f32),
                    new ReturnOp(List.of(v1))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // negate([1, -2, 3, -4]) = [-1, 2, -3, 4]
            // abs([-1, 2, -3, 4]) = [1, 2, 3, 4]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, -2, 3, -4}, 4)) {
                List<Tensor> outputs = executor.execute(graph, input);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Diamond Pattern")
    class DiamondPatternTests {

        @Test
        void executeDiamond() {
            // func @diamond(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            //   %0 = stablehlo.negate %arg0 : tensor<4xf32>
            //   %1 = stablehlo.abs %arg0 : tensor<4xf32>
            //   %2 = stablehlo.add %0, %1 : tensor<4xf32>
            //   return %2 : tensor<4xf32>
            // }
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value v0 = new Value("0", t4f32);
            Value v1 = new Value("1", t4f32);
            Value v2 = new Value("2", t4f32);

            Function func = new Function(
                "diamond",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(
                    new NegateOp(v0, arg0, t4f32),
                    new AbsOp(v1, arg0, t4f32),
                    new AddOp(v2, v0, v1, t4f32),
                    new ReturnOp(List.of(v2))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Input: [-2, 3, -4, 5]
            // negate([-2, 3, -4, 5]) = [2, -3, 4, -5]
            // abs([-2, 3, -4, 5]) = [2, 3, 4, 5]
            // add([2, -3, 4, -5], [2, 3, 4, 5]) = [4, 0, 8, 0]
            try (Tensor input = Tensor.fromFloatArray(new float[]{-2, 3, -4, 5}, 4)) {
                List<Tensor> outputs = executor.execute(graph, input);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {4, 0, 8, 0};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Error Handling")
    class ErrorHandlingTests {

        @Test
        void wrongInputCountThrows() {
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);
            Value arg1 = new Value("arg1", t4f32);
            Value result = new Value("0", t4f32);

            Function func = new Function(
                "add",
                List.of(
                    new Argument("arg0", t4f32),
                    new Argument("arg1", t4f32)
                ),
                List.of(t4f32),
                List.of(
                    new AddOp(result, arg0, arg1, t4f32),
                    new ReturnOp(List.of(result))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                // Only 1 input when 2 are expected
                assertThrows(IllegalArgumentException.class, () ->
                    executor.execute(graph, a));
            }
        }

        @Test
        void wrongInputShapeThrows() {
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Shape [8] when [4] is expected
            try (Tensor wrongShape = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 8)) {
                assertThrows(IllegalArgumentException.class, () ->
                    executor.execute(graph, wrongShape));
            }
        }
    }

    @Nested
    @DisplayName("Multi-dimensional Tensors")
    class MultiDimensionalTests {

        @Test
        void execute2dTensorAdd() {
            // func @add2d(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
            //   %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
            //   return %0 : tensor<2x3xf32>
            // }
            TensorType t2x3f32 = tensor(2, 3);
            Value arg0 = new Value("arg0", t2x3f32);
            Value arg1 = new Value("arg1", t2x3f32);
            Value result = new Value("0", t2x3f32);

            Function func = new Function(
                "add2d",
                List.of(
                    new Argument("arg0", t2x3f32),
                    new Argument("arg1", t2x3f32)
                ),
                List.of(t2x3f32),
                List.of(
                    new AddOp(result, arg0, arg1, t2x3f32),
                    new ReturnOp(List.of(result))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{6, 5, 4, 3, 2, 1}, 2, 3)) {

                List<Tensor> outputs = executor.execute(graph, a, b);

                assertEquals(1, outputs.size());
                try (Tensor output = outputs.getFirst()) {
                    float[] expected = {7, 7, 7, 7, 7, 7};
                    assertArrayEquals(expected, output.toFloatArray(), 1e-6f);
                    assertArrayEquals(new int[]{2, 3}, output.shape());
                }
            }
        }
    }
}
