package io.surfworks.warpforge.core.tensor;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NegateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReturnOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.warpforge.backend.cpu.CpuBackend;
import io.surfworks.warpforge.core.graph.ExecutableGraph;
import io.surfworks.warpforge.core.graph.GraphCompiler;
import io.surfworks.warpforge.core.graph.GraphExecutor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Comprehensive tests for GraphExecutor shape validation.
 * These tests document expected behavior for shape mismatch scenarios
 * at graph execution boundaries.
 */
@DisplayName("GraphExecutor Shape Validation")
class GraphExecutorShapeTest {

    private CpuBackend backend;
    private GraphExecutor executor;
    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;

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
    @DisplayName("Input Count Validation")
    class InputCountValidation {

        @Test
        @DisplayName("Throws when providing too few inputs")
        void tooFewInputs() {
            // Graph expects 2 inputs
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
                // Only 1 input when 2 expected
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, a));
                assertTrue(ex.getMessage().contains("2"),
                    "Error message should mention expected count");
                assertTrue(ex.getMessage().contains("1"),
                    "Error message should mention actual count");
            }
        }

        @Test
        @DisplayName("Throws when providing too many inputs")
        void tooManyInputs() {
            // Graph expects 1 input
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

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 4)) {
                // 2 inputs when 1 expected
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, a, b));
                assertTrue(ex.getMessage().contains("1"),
                    "Error message should mention expected count");
                assertTrue(ex.getMessage().contains("2"),
                    "Error message should mention actual count");
            }
        }

        @Test
        @DisplayName("Throws when providing zero inputs to non-zero-input graph")
        void zeroInputsForNonZeroGraph() {
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

            var ex = assertThrows(IllegalArgumentException.class,
                () -> executor.execute(graph, List.of()));
            assertTrue(ex.getMessage().contains("1"),
                "Error message should mention expected count");
            assertTrue(ex.getMessage().contains("0"),
                "Error message should mention zero inputs provided");
        }

        @Test
        @DisplayName("Succeeds with correct input count")
        void correctInputCount() {
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
                assertDoesNotThrow(() -> executor.execute(graph, a, b));
            }
        }
    }

    @Nested
    @DisplayName("Input Shape Validation - Wrong Dimensions")
    class InputShapeWrongDimensions {

        @Test
        @DisplayName("Throws when input has wrong rank (fewer dimensions)")
        void wrongRankFewerDimensions() {
            // Graph expects [2, 3] tensor
            TensorType t2x3f32 = tensor(2, 3);
            Value arg0 = new Value("arg0", t2x3f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t2x3f32)),
                List.of(t2x3f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Provide 1D tensor [6] instead of 2D [2, 3]
            try (Tensor wrongRank = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 6)) {
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongRank));
                assertTrue(ex.getMessage().contains("[6]") || ex.getMessage().contains("6"),
                    "Error message should include actual shape");
                assertTrue(ex.getMessage().contains("[2, 3]"),
                    "Error message should include expected shape");
            }
        }

        @Test
        @DisplayName("Throws when input has wrong rank (more dimensions)")
        void wrongRankMoreDimensions() {
            // Graph expects [6] tensor
            TensorType t6f32 = tensor(6);
            Value arg0 = new Value("arg0", t6f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t6f32)),
                List.of(t6f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Provide 2D tensor [2, 3] instead of 1D [6]
            try (Tensor wrongRank = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongRank));
            }
        }

        @Test
        @DisplayName("Throws when input has wrong rank (3D vs 2D)")
        void wrongRank3Dvs2D() {
            // Graph expects [2, 3] tensor
            TensorType t2x3f32 = tensor(2, 3);
            Value arg0 = new Value("arg0", t2x3f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t2x3f32)),
                List.of(t2x3f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Provide 3D tensor [1, 2, 3] instead of 2D [2, 3]
            try (Tensor wrongRank = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 1, 2, 3)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongRank));
            }
        }
    }

    @Nested
    @DisplayName("Input Shape Validation - Right Rank, Wrong Sizes")
    class InputShapeWrongSizes {

        @Test
        @DisplayName("Throws when first dimension size is wrong")
        void wrongFirstDimensionSize() {
            // Graph expects [2, 3] tensor
            TensorType t2x3f32 = tensor(2, 3);
            Value arg0 = new Value("arg0", t2x3f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t2x3f32)),
                List.of(t2x3f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Provide [3, 3] instead of [2, 3]
            try (Tensor wrongSize = Tensor.fromFloatArray(new float[9], 3, 3)) {
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongSize));
                assertTrue(ex.getMessage().contains("Input 0"),
                    "Error message should indicate which input failed");
            }
        }

        @Test
        @DisplayName("Throws when second dimension size is wrong")
        void wrongSecondDimensionSize() {
            // Graph expects [2, 3] tensor
            TensorType t2x3f32 = tensor(2, 3);
            Value arg0 = new Value("arg0", t2x3f32);

            Function func = new Function(
                "identity",
                List.of(new Argument("arg0", t2x3f32)),
                List.of(t2x3f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Provide [2, 4] instead of [2, 3]
            try (Tensor wrongSize = Tensor.fromFloatArray(new float[8], 2, 4)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongSize));
            }
        }

        @Test
        @DisplayName("Throws when dimension size is larger than expected")
        void dimensionSizeTooLarge() {
            // Graph expects [4] tensor
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

            // Provide [8] instead of [4]
            try (Tensor wrongSize = Tensor.fromFloatArray(new float[8], 8)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongSize));
            }
        }

        @Test
        @DisplayName("Throws when dimension size is smaller than expected")
        void dimensionSizeTooSmall() {
            // Graph expects [4] tensor
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

            // Provide [2] instead of [4]
            try (Tensor wrongSize = Tensor.fromFloatArray(new float[2], 2)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongSize));
            }
        }
    }

    @Nested
    @DisplayName("Multi-Input Shape Validation")
    class MultiInputShapeValidation {

        @Test
        @DisplayName("Throws when second input has wrong shape")
        void secondInputWrongShape() {
            // Graph expects two [4] tensors
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

            // First input correct, second input wrong shape
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 6)) {
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, a, b));
                assertTrue(ex.getMessage().contains("Input 1"),
                    "Error message should indicate input 1 (second input) failed");
            }
        }

        @Test
        @DisplayName("Throws when first input of many has wrong shape")
        void firstInputWrongShapeInMultiInput() {
            // Graph expects two [4] tensors
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

            // First input wrong shape, second input correct
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 6);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, a, b));
                assertTrue(ex.getMessage().contains("Input 0"),
                    "Error message should indicate input 0 (first input) failed");
            }
        }

        @Test
        @DisplayName("Validates all inputs have correct shapes")
        void allInputsCorrectShapes() {
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
            }
        }
    }

    @Nested
    @DisplayName("Mixed Shape Requirements")
    class MixedShapeRequirements {

        @Test
        @DisplayName("Graph with inputs of different shapes validates each correctly")
        void differentInputShapes() {
            // Graph expects [4] and [2, 3] tensors
            TensorType t4f32 = tensor(4);
            TensorType t2x3f32 = tensor(2, 3);

            Value arg0 = new Value("arg0", t4f32);
            Value arg1 = new Value("arg1", t2x3f32);
            Value v0 = new Value("0", t4f32);
            Value v1 = new Value("1", t2x3f32);

            Function func = new Function(
                "mixed",
                List.of(
                    new Argument("arg0", t4f32),
                    new Argument("arg1", t2x3f32)
                ),
                List.of(t4f32, t2x3f32),
                List.of(
                    new NegateOp(v0, arg0, t4f32),
                    new NegateOp(v1, arg1, t2x3f32),
                    new ReturnOp(List.of(v0, v1))
                ),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Correct shapes
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                List<Tensor> outputs = executor.execute(graph, a, b);
                assertEquals(2, outputs.size());
            }

            // Swapped shapes should fail
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, a, b));
            }
        }
    }

    @Nested
    @DisplayName("Multi-Dimensional Shape Validation")
    class MultiDimensionalValidation {

        @Test
        @DisplayName("3D tensor shape validation")
        void threeDimensionalValidation() {
            TensorType t2x3x4f32 = tensor(2, 3, 4);
            Value arg0 = new Value("arg0", t2x3x4f32);

            Function func = new Function(
                "identity3d",
                List.of(new Argument("arg0", t2x3x4f32)),
                List.of(t2x3x4f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Correct 3D shape
            try (Tensor correct = Tensor.zeros(2, 3, 4)) {
                assertDoesNotThrow(() -> executor.execute(graph, correct));
            }

            // Wrong 3D shape (same total elements, different shape)
            try (Tensor wrong = Tensor.zeros(4, 3, 2)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrong));
            }

            // 2D with same total elements
            try (Tensor wrong2d = Tensor.zeros(6, 4)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrong2d));
            }
        }

        @Test
        @DisplayName("4D tensor shape validation (batch, channel, height, width)")
        void fourDimensionalValidation() {
            TensorType t8x3x32x32f32 = tensor(8, 3, 32, 32); // [N, C, H, W]
            Value arg0 = new Value("arg0", t8x3x32x32f32);

            Function func = new Function(
                "conv_input",
                List.of(new Argument("arg0", t8x3x32x32f32)),
                List.of(t8x3x32x32f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            // Correct shape
            try (Tensor correct = Tensor.zeros(8, 3, 32, 32)) {
                assertDoesNotThrow(() -> executor.execute(graph, correct));
            }

            // Wrong batch size
            try (Tensor wrongBatch = Tensor.zeros(16, 3, 32, 32)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongBatch));
            }

            // Wrong channel count
            try (Tensor wrongChannels = Tensor.zeros(8, 1, 32, 32)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongChannels));
            }

            // Wrong spatial dimensions
            try (Tensor wrongSpatial = Tensor.zeros(8, 3, 64, 64)) {
                assertThrows(IllegalArgumentException.class,
                    () -> executor.execute(graph, wrongSpatial));
            }
        }
    }
}
