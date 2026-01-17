package io.surfworks.warpforge.core.graph;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AbsOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NegateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReturnOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.warpforge.core.tensor.ScalarType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GraphCompilerTest {

    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;

    // Helper to create TensorType with integer shape
    private static TensorType tensor(int... dims) {
        return new TensorType(
            java.util.Arrays.stream(dims).boxed().toList(),
            F32
        );
    }

    @Nested
    @DisplayName("Simple Functions")
    class SimpleFunctionTests {

        @Test
        void compileIdentityFunction() {
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

            assertEquals("identity", graph.name());
            assertEquals(1, graph.inputCount());
            assertEquals(1, graph.outputCount());
            assertEquals(1, graph.tensorCount()); // Only the input tensor
            assertEquals(1, graph.nodes().size()); // Only return op
            assertTrue(graph.nodes().getFirst().isReturn());
        }

        @Test
        void compileUnaryNegation() {
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

            assertEquals("negate", graph.name());
            assertEquals(1, graph.inputCount());
            assertEquals(1, graph.outputCount());
            assertEquals(2, graph.tensorCount()); // arg0 and %0
            assertEquals(2, graph.nodes().size()); // negate + return

            GraphNode negateNode = graph.nodes().get(0);
            assertEquals("stablehlo.negate", negateNode.opName());
            assertArrayEquals(new int[]{0}, negateNode.inputIndices()); // arg0
            assertArrayEquals(new int[]{1}, negateNode.outputIndices()); // %0
        }

        @Test
        void compileBinaryAdd() {
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

            assertEquals(2, graph.inputCount());
            assertEquals(1, graph.outputCount());
            assertEquals(3, graph.tensorCount()); // arg0, arg1, %0

            GraphNode addNode = graph.nodes().get(0);
            assertArrayEquals(new int[]{0, 1}, addNode.inputIndices());
            assertArrayEquals(new int[]{2}, addNode.outputIndices());
        }
    }

    @Nested
    @DisplayName("Multi-Operation Graphs")
    class MultiOperationTests {

        @Test
        void compileChainedOperations() {
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

            assertEquals(3, graph.tensorCount()); // arg0, %0, %1
            assertEquals(3, graph.nodes().size()); // negate, abs, return

            // Verify chain: arg0 -> negate -> %0 -> abs -> %1
            GraphNode negateNode = graph.nodes().get(0);
            assertArrayEquals(new int[]{0}, negateNode.inputIndices());
            assertArrayEquals(new int[]{1}, negateNode.outputIndices());

            GraphNode absNode = graph.nodes().get(1);
            assertArrayEquals(new int[]{1}, absNode.inputIndices());
            assertArrayEquals(new int[]{2}, absNode.outputIndices());
        }

        @Test
        void compileDiamondPattern() {
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

            assertEquals(4, graph.tensorCount()); // arg0, %0, %1, %2

            // Verify diamond: arg0 branches to negate and abs, then merges in add
            GraphNode addNode = graph.nodes().get(2);
            assertArrayEquals(new int[]{1, 2}, addNode.inputIndices()); // %0 and %1
        }
    }

    @Nested
    @DisplayName("Module Compilation")
    class ModuleCompilationTests {

        @Test
        void compileModule() {
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);

            Function func = new Function(
                "main",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            StableHloAst.Module module = new StableHloAst.Module("test_module", List.of(func));

            ExecutableGraph graph = GraphCompiler.compile(module);
            assertEquals("main", graph.name());
        }

        @Test
        void compileNamedFunction() {
            TensorType t4f32 = tensor(4);
            Value arg0 = new Value("arg0", t4f32);

            Function func1 = new Function(
                "helper",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(new ReturnOp(List.of(arg0))),
                false
            );

            Function func2 = new Function(
                "main",
                List.of(new Argument("arg0", t4f32)),
                List.of(t4f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            StableHloAst.Module module = new StableHloAst.Module("test_module", List.of(func1, func2));

            ExecutableGraph helper = GraphCompiler.compile(module, "helper");
            assertEquals("helper", helper.name());

            ExecutableGraph main = GraphCompiler.compile(module, "main");
            assertEquals("main", main.name());
        }

        @Test
        void missingFunctionThrows() {
            StableHloAst.Module module = new StableHloAst.Module("empty", List.of());

            assertThrows(IllegalArgumentException.class, () ->
                GraphCompiler.compile(module, "missing"));
        }
    }

    @Nested
    @DisplayName("Tensor Specs")
    class TensorSpecTests {

        @Test
        void extractsCorrectTensorSpecs() {
            TensorType t2x3f32 = tensor(2, 3);
            Value arg0 = new Value("arg0", t2x3f32);

            Function func = new Function(
                "test",
                List.of(new Argument("arg0", t2x3f32)),
                List.of(t2x3f32),
                List.of(new ReturnOp(List.of(arg0))),
                true
            );

            ExecutableGraph graph = GraphCompiler.compile(func);

            assertEquals(1, graph.tensorSpecs().size());
            var spec = graph.tensorSpec(0);
            assertArrayEquals(new int[]{2, 3}, spec.shape());
            assertEquals(ScalarType.F32, spec.dtype());
        }
    }
}
