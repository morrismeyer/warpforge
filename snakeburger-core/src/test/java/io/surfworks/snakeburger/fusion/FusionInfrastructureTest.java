package io.surfworks.snakeburger.fusion;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScalarType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Type;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Tests for the fusion infrastructure components.
 */
@DisplayName("Fusion Infrastructure")
class FusionInfrastructureTest {

    private static final TensorType TENSOR_4x8_F32 = new TensorType(List.of(4, 8), ScalarType.F32);
    private static final TensorType TENSOR_4_F32 = new TensorType(List.of(4), ScalarType.F32);

    // ==================== OperationGraph Tests ====================

    @Nested
    @DisplayName("OperationGraph")
    class OperationGraphTests {

        @Test
        @DisplayName("builds graph from simple function")
        void buildsGraphFromSimpleFunction() {
            // %0 = add(%arg0, %arg1)
            Value arg0 = new Value("arg0", TENSOR_4x8_F32);
            Value arg1 = new Value("arg1", TENSOR_4x8_F32);
            Value result = new Value("0", TENSOR_4x8_F32);

            AddOp addOp = new AddOp(result, arg0, arg1, TENSOR_4x8_F32);

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32), new Argument("arg1", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(addOp),
                    true
            );

            OperationGraph graph = OperationGraph.build(func);

            assertNotNull(graph);
            assertEquals(addOp, graph.producer(result));
            assertTrue(graph.consumers(arg0).contains(addOp));
            assertTrue(graph.consumers(arg1).contains(addOp));
        }

        @Test
        @DisplayName("detects single use values")
        void detectsSingleUseValues() {
            Value arg0 = new Value("arg0", TENSOR_4x8_F32);
            Value r0 = new Value("0", TENSOR_4x8_F32);
            Value r1 = new Value("1", TENSOR_4x8_F32);

            ExpOp expOp = new ExpOp(r0, arg0, TENSOR_4x8_F32);
            ExpOp expOp2 = new ExpOp(r1, r0, TENSOR_4x8_F32);

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(expOp, expOp2),
                    true
            );

            OperationGraph graph = OperationGraph.build(func);

            assertTrue(graph.hasSingleUse(arg0));
            assertTrue(graph.hasSingleUse(r0));
            assertEquals(1, graph.useCount(arg0));
        }

        @Test
        @DisplayName("detects multiple use values")
        void detectsMultipleUseValues() {
            Value arg0 = new Value("arg0", TENSOR_4x8_F32);
            Value r0 = new Value("0", TENSOR_4x8_F32);
            Value r1 = new Value("1", TENSOR_4x8_F32);

            // arg0 used twice: in both add operations
            AddOp add1 = new AddOp(r0, arg0, arg0, TENSOR_4x8_F32);
            AddOp add2 = new AddOp(r1, r0, arg0, TENSOR_4x8_F32);

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(add1, add2),
                    true
            );

            OperationGraph graph = OperationGraph.build(func);

            assertFalse(graph.hasSingleUse(arg0));
            assertEquals(3, graph.useCount(arg0)); // twice in add1, once in add2
        }

        @Test
        @DisplayName("identifies function arguments")
        void identifiesFunctionArguments() {
            Value arg0 = new Value("arg0", TENSOR_4x8_F32);
            Value r0 = new Value("0", TENSOR_4x8_F32);

            ExpOp expOp = new ExpOp(r0, arg0, TENSOR_4x8_F32);

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(expOp),
                    true
            );

            OperationGraph graph = OperationGraph.build(func);

            assertTrue(graph.isFunctionArgument(arg0));
            assertFalse(graph.isFunctionArgument(r0));
        }

        @Test
        @DisplayName("returns null for argument producer")
        void returnsNullForArgumentProducer() {
            Value arg0 = new Value("arg0", TENSOR_4x8_F32);
            Value r0 = new Value("0", TENSOR_4x8_F32);

            ExpOp expOp = new ExpOp(r0, arg0, TENSOR_4x8_F32);

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(expOp),
                    true
            );

            OperationGraph graph = OperationGraph.build(func);

            assertNull(graph.producer(arg0));
            assertEquals(expOp, graph.producer(r0));
        }
    }

    // ==================== FusionMatch Tests ====================

    @Nested
    @DisplayName("FusionMatch")
    class FusionMatchTests {

        @Test
        @DisplayName("provides access to captured values")
        void providesAccessToCapturedValues() {
            Value input = new Value("input", TENSOR_4x8_F32);
            Value output = new Value("output", TENSOR_4x8_F32);

            FusionMatch match = new FusionMatch(
                    "test",
                    List.of(),
                    Map.of("input", input, "output", output),
                    Map.of()
            );

            assertEquals(input, match.input());
            assertEquals(output, match.output());
            assertEquals(input, match.capture("input"));
        }

        @Test
        @DisplayName("provides access to attributes")
        void providesAccessToAttributes() {
            FusionMatch match = new FusionMatch(
                    "test",
                    List.of(),
                    Map.of(),
                    Map.of("axis", -1, "epsilon", 1e-5)
            );

            assertEquals(-1, (int) match.attribute("axis"));
            assertEquals(1e-5, (double) match.attribute("epsilon"), 1e-10);
        }

        @Test
        @DisplayName("provides default for missing attributes")
        void providesDefaultForMissingAttributes() {
            FusionMatch match = new FusionMatch(
                    "test",
                    List.of(),
                    Map.of(),
                    Map.of()
            );

            assertEquals(42, (int) match.attribute("missing", 42));
        }
    }

    // ==================== FusedOperation Tests ====================

    @Nested
    @DisplayName("FusedOperation")
    class FusedOperationTests {

        @Test
        @DisplayName("creates fused operation with correct op name")
        void createsFusedOperationWithCorrectOpName() {
            Value input = new Value("input", TENSOR_4x8_F32);
            Value output = new Value("output", TENSOR_4x8_F32);

            FusedOperation fused = new FusedOperation(
                    FusedOperation.SOFTMAX,
                    List.of(input),
                    List.of(output),
                    Map.of("axis", -1),
                    List.of()
            );

            assertEquals("fused.softmax", fused.opName());
            assertEquals(FusedOperation.SOFTMAX, fused.fusionType());
        }

        @Test
        @DisplayName("provides axis accessor")
        void providesAxisAccessor() {
            FusedOperation fused = new FusedOperation(
                    FusedOperation.SOFTMAX,
                    List.of(),
                    List.of(new Value("out", TENSOR_4x8_F32)),
                    Map.of("axis", 2),
                    List.of()
            );

            assertEquals(2, fused.axis());
        }

        @Test
        @DisplayName("provides epsilon accessor with default")
        void providesEpsilonAccessorWithDefault() {
            FusedOperation fused = new FusedOperation(
                    FusedOperation.LAYER_NORM,
                    List.of(),
                    List.of(new Value("out", TENSOR_4x8_F32)),
                    Map.of(),
                    List.of()
            );

            assertEquals(1e-5, fused.epsilon(), 1e-10);
        }

        @Test
        @DisplayName("returns tensor result type")
        void returnsTensorResultType() {
            Value output = new Value("output", TENSOR_4x8_F32);

            FusedOperation fused = new FusedOperation(
                    FusedOperation.SOFTMAX,
                    List.of(),
                    List.of(output),
                    Map.of(),
                    List.of()
            );

            assertEquals(TENSOR_4x8_F32, fused.tensorResultType());
        }

        @Test
        @DisplayName("implements Operation interface")
        void implementsOperationInterface() {
            Value input = new Value("input", TENSOR_4x8_F32);
            Value output = new Value("output", TENSOR_4x8_F32);

            FusedOperation fused = new FusedOperation(
                    FusedOperation.RMS_NORM,
                    List.of(input),
                    List.of(output),
                    Map.of(),
                    List.of()
            );

            // FusedOperation implements Operation
            Operation op = fused;
            assertEquals("fused.rms_norm", op.opName());
            assertEquals(List.of(input), op.operands());
            assertEquals(List.of(output), op.results());
        }
    }

    // ==================== FusionPass Tests ====================

    @Nested
    @DisplayName("FusionPass")
    class FusionPassTests {

        @Test
        @DisplayName("creates pass with no patterns")
        void createsPassWithNoPatterns() {
            FusionPass pass = new FusionPass();

            assertEquals(0, pass.patterns().size());
            assertEquals(0, pass.lastFusionCount());
        }

        @Test
        @DisplayName("adds patterns and sorts by speedup")
        void addsPatternsAndSortsBySpeedup() {
            FusionPass pass = new FusionPass()
                    .addPattern(new RMSNormFusion())    // 2.5x
                    .addPattern(new SoftmaxFusion())    // 3.0x
                    .addPattern(new LayerNormFusion()); // 3.0x

            assertEquals(3, pass.patterns().size());
            // Should be sorted by speedup (highest first)
            assertTrue(pass.patterns().get(0).estimatedSpeedup() >=
                       pass.patterns().get(1).estimatedSpeedup());
        }

        @Test
        @DisplayName("withStandardPatterns includes all standard patterns")
        void withStandardPatternsIncludesAllStandardPatterns() {
            FusionPass pass = FusionPass.withStandardPatterns();

            assertEquals(3, pass.patterns().size());
            assertTrue(pass.patterns().stream()
                    .anyMatch(p -> p.name().equals(FusedOperation.SOFTMAX)));
            assertTrue(pass.patterns().stream()
                    .anyMatch(p -> p.name().equals(FusedOperation.LAYER_NORM)));
            assertTrue(pass.patterns().stream()
                    .anyMatch(p -> p.name().equals(FusedOperation.RMS_NORM)));
        }

        @Test
        @DisplayName("preserves function when no patterns match")
        void preservesFunctionWhenNoPatternsMatch() {
            // Simple function with no fusable patterns
            Value arg0 = new Value("arg0", TENSOR_4x8_F32);
            Value r0 = new Value("0", TENSOR_4x8_F32);

            ExpOp expOp = new ExpOp(r0, arg0, TENSOR_4x8_F32);

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(expOp),
                    true
            );

            FusionPass pass = FusionPass.withStandardPatterns();
            Function result = pass.fuseFunction(func);

            assertEquals(1, result.body().size());
            assertEquals(0, pass.lastFusionCount());
        }
    }

    // ==================== FusionPattern Tests ====================

    @Nested
    @DisplayName("FusionPattern Interface")
    class FusionPatternTests {

        @Test
        @DisplayName("SoftmaxFusion has correct metadata")
        void softmaxFusionHasCorrectMetadata() {
            SoftmaxFusion pattern = new SoftmaxFusion();

            assertEquals(FusedOperation.SOFTMAX, pattern.name());
            assertEquals(3.0, pattern.estimatedSpeedup());
            assertNotNull(pattern.description());
        }

        @Test
        @DisplayName("LayerNormFusion has correct metadata")
        void layerNormFusionHasCorrectMetadata() {
            LayerNormFusion pattern = new LayerNormFusion();

            assertEquals(FusedOperation.LAYER_NORM, pattern.name());
            assertEquals(3.0, pattern.estimatedSpeedup());
            assertNotNull(pattern.description());
        }

        @Test
        @DisplayName("RMSNormFusion has correct metadata")
        void rmsNormFusionHasCorrectMetadata() {
            RMSNormFusion pattern = new RMSNormFusion();

            assertEquals(FusedOperation.RMS_NORM, pattern.name());
            assertEquals(2.5, pattern.estimatedSpeedup());
            assertNotNull(pattern.description());
        }
    }

    // ==================== Integration Tests ====================

    @Nested
    @DisplayName("Integration")
    class IntegrationTests {

        @Test
        @DisplayName("FusedOperation can be added to function body")
        void fusedOperationCanBeAddedToFunctionBody() {
            Value input = new Value("arg0", TENSOR_4x8_F32);
            Value output = new Value("0", TENSOR_4x8_F32);

            FusedOperation fused = new FusedOperation(
                    FusedOperation.SOFTMAX,
                    List.of(input),
                    List.of(output),
                    Map.of("axis", -1),
                    List.of()
            );

            Function func = new Function(
                    "test",
                    List.of(new Argument("arg0", TENSOR_4x8_F32)),
                    List.of(TENSOR_4x8_F32),
                    List.of(fused),
                    true
            );

            assertEquals(1, func.body().size());
            assertTrue(func.body().get(0) instanceof FusedOperation);
        }
    }
}
