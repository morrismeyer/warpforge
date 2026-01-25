package io.surfworks.snakeburger.fusion;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConstantOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DenseAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScalarType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Integration tests for fusion pass on actual StableHLO patterns.
 *
 * <p>These tests construct real StableHLO operation graphs that represent
 * common ML patterns (softmax, layer_norm, rms_norm) and verify that the
 * fusion pass correctly identifies and fuses them.
 */
@DisplayName("Fusion Pass Integration Tests")
class FusionPassIntegrationTest {

    private static final TensorType TENSOR_2x8_F32 = new TensorType(List.of(2, 8), ScalarType.F32);
    private static final TensorType TENSOR_2x1_F32 = new TensorType(List.of(2, 1), ScalarType.F32);
    private static final TensorType TENSOR_8_F32 = new TensorType(List.of(8), ScalarType.F32);
    private static final TensorType SCALAR_F32 = new TensorType(List.of(), ScalarType.F32);

    // ==================== Softmax Fusion Tests ====================

    @Nested
    @DisplayName("Softmax Fusion")
    class SoftmaxFusionTests {

        /**
         * Builds a softmax pattern:
         * %max = reduce_max(%input, dim=-1)
         * %max_broadcast = broadcast_in_dim(%max)
         * %shifted = subtract(%input, %max_broadcast)
         * %exp = exponential(%shifted)
         * %sum = reduce_add(%exp, dim=-1)
         * %sum_broadcast = broadcast_in_dim(%sum)
         * %result = divide(%exp, %sum_broadcast)
         */
        private Function buildSoftmaxFunction() {
            Value input = new Value("arg0", TENSOR_2x8_F32);
            Value initMax = new Value("init_max", SCALAR_F32);
            Value initSum = new Value("init_sum", SCALAR_F32);

            // Constants for reduce init values
            ConstantOp initMaxOp = new ConstantOp(
                    new Value("init_max", SCALAR_F32),
                    new DenseAttr(Float.NEGATIVE_INFINITY, SCALAR_F32),
                    SCALAR_F32
            );
            ConstantOp initSumOp = new ConstantOp(
                    new Value("init_sum", SCALAR_F32),
                    new DenseAttr(0.0f, SCALAR_F32),
                    SCALAR_F32
            );

            // %max = reduce_max(%input, dim=1)
            Value maxResult = new Value("max", TENSOR_2x1_F32);
            ReduceOp maxOp = new ReduceOp(
                    maxResult,
                    input,
                    initMaxOp.result(),
                    List.of(1L),  // reduce over last dimension
                    "max",
                    TENSOR_2x1_F32
            );

            // %max_broadcast = broadcast_in_dim(%max)
            Value maxBroadcast = new Value("max_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp maxBroadcastOp = new BroadcastInDimOp(
                    maxBroadcast,
                    maxResult,
                    List.of(0L, 1L),
                    TENSOR_2x8_F32
            );

            // %shifted = subtract(%input, %max_broadcast)
            Value shifted = new Value("shifted", TENSOR_2x8_F32);
            SubtractOp shiftedOp = new SubtractOp(shifted, input, maxBroadcast, TENSOR_2x8_F32);

            // %exp = exponential(%shifted)
            Value exp = new Value("exp", TENSOR_2x8_F32);
            ExpOp expOp = new ExpOp(exp, shifted, TENSOR_2x8_F32);

            // %sum = reduce_add(%exp, dim=1)
            Value sumResult = new Value("sum", TENSOR_2x1_F32);
            ReduceOp sumOp = new ReduceOp(
                    sumResult,
                    exp,
                    initSumOp.result(),
                    List.of(1L),
                    "add",
                    TENSOR_2x1_F32
            );

            // %sum_broadcast = broadcast_in_dim(%sum)
            Value sumBroadcast = new Value("sum_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp sumBroadcastOp = new BroadcastInDimOp(
                    sumBroadcast,
                    sumResult,
                    List.of(0L, 1L),
                    TENSOR_2x8_F32
            );

            // %result = divide(%exp, %sum_broadcast)
            Value result = new Value("result", TENSOR_2x8_F32);
            DivideOp divOp = new DivideOp(result, exp, sumBroadcast, TENSOR_2x8_F32);

            return new Function(
                    "softmax",
                    List.of(new Argument("arg0", TENSOR_2x8_F32)),
                    List.of(TENSOR_2x8_F32),
                    List.of(initMaxOp, initSumOp, maxOp, maxBroadcastOp, shiftedOp,
                            expOp, sumOp, sumBroadcastOp, divOp),
                    true
            );
        }

        @Test
        @DisplayName("fuses softmax pattern into single operation")
        void fusesSoftmaxPattern() {
            Function func = buildSoftmaxFunction();
            Module module = new Module("test", List.of(func));

            FusionPass pass = FusionPass.withStandardPatterns();
            Module result = pass.apply(module);

            assertEquals(1, pass.lastFusionCount(), "Should fuse one softmax pattern");

            Function fusedFunc = result.functions().get(0);
            // After fusion, we should have: init constants + FusedOperation
            // The fused op replaces max, broadcast, subtract, exp, sum, broadcast, divide

            boolean hasFusedSoftmax = fusedFunc.body().stream()
                    .anyMatch(op -> op instanceof FusedOperation fused &&
                                    fused.fusionType().equals(FusedOperation.SOFTMAX));

            assertTrue(hasFusedSoftmax, "Should contain fused softmax operation");
        }

        @Test
        @DisplayName("extracts correct axis from softmax pattern")
        void extractsCorrectAxis() {
            Function func = buildSoftmaxFunction();
            Module module = new Module("test", List.of(func));

            FusionPass pass = FusionPass.withStandardPatterns();
            pass.apply(module);

            assertEquals(1, pass.lastMatches().size());
            FusionMatch match = pass.lastMatches().get(0);
            assertEquals(1, (int) match.attribute("axis"), "Should extract axis=1");
        }
    }

    // ==================== RMSNorm Fusion Tests ====================

    @Nested
    @DisplayName("RMSNorm Fusion")
    class RMSNormFusionTests {

        /**
         * Builds an RMSNorm pattern:
         * %sq = multiply(%input, %input)
         * %mean_sq = reduce_add(%sq, dim=-1) / N
         * %eps = constant<1e-6>
         * %mean_sq_eps = add(%mean_sq, %eps_broadcast)
         * %rsqrt = rsqrt(%mean_sq_eps)
         * %rsqrt_broadcast = broadcast_in_dim(%rsqrt)
         * %normalized = multiply(%input, %rsqrt_broadcast)
         * %result = multiply(%normalized, %weight)
         */
        private Function buildRMSNormFunction() {
            Value input = new Value("arg0", TENSOR_2x8_F32);
            Value weight = new Value("arg1", TENSOR_8_F32);
            Value initSum = new Value("init_sum", SCALAR_F32);

            // Constants
            ConstantOp initSumOp = new ConstantOp(
                    new Value("init_sum", SCALAR_F32),
                    new DenseAttr(0.0f, SCALAR_F32),
                    SCALAR_F32
            );
            ConstantOp epsOp = new ConstantOp(
                    new Value("eps", SCALAR_F32),
                    new DenseAttr(1e-6f, SCALAR_F32),
                    SCALAR_F32
            );
            ConstantOp divConstOp = new ConstantOp(
                    new Value("div_const", SCALAR_F32),
                    new DenseAttr(8.0f, SCALAR_F32),  // N = 8
                    SCALAR_F32
            );

            // %sq = multiply(%input, %input)
            Value sq = new Value("sq", TENSOR_2x8_F32);
            MultiplyOp sqOp = new MultiplyOp(sq, input, input, TENSOR_2x8_F32);

            // %sum_sq = reduce_add(%sq, dim=1)
            Value sumSq = new Value("sum_sq", TENSOR_2x1_F32);
            ReduceOp sumSqOp = new ReduceOp(
                    sumSq,
                    sq,
                    initSumOp.result(),
                    List.of(1L),
                    "add",
                    TENSOR_2x1_F32
            );

            // %div_broadcast = broadcast_in_dim(%div_const)
            Value divBroadcast = new Value("div_broadcast", TENSOR_2x1_F32);
            BroadcastInDimOp divBroadcastOp = new BroadcastInDimOp(
                    divBroadcast,
                    divConstOp.result(),
                    List.of(),
                    TENSOR_2x1_F32
            );

            // %mean_sq = divide(%sum_sq, %div_broadcast)
            Value meanSq = new Value("mean_sq", TENSOR_2x1_F32);
            DivideOp meanSqOp = new DivideOp(meanSq, sumSq, divBroadcast, TENSOR_2x1_F32);

            // %eps_broadcast = broadcast_in_dim(%eps)
            Value epsBroadcast = new Value("eps_broadcast", TENSOR_2x1_F32);
            BroadcastInDimOp epsBroadcastOp = new BroadcastInDimOp(
                    epsBroadcast,
                    epsOp.result(),
                    List.of(),
                    TENSOR_2x1_F32
            );

            // %mean_sq_eps = add(%mean_sq, %eps_broadcast)
            Value meanSqEps = new Value("mean_sq_eps", TENSOR_2x1_F32);
            AddOp meanSqEpsOp = new AddOp(meanSqEps, meanSq, epsBroadcast, TENSOR_2x1_F32);

            // %rsqrt = rsqrt(%mean_sq_eps)
            Value rsqrt = new Value("rsqrt", TENSOR_2x1_F32);
            RsqrtOp rsqrtOp = new RsqrtOp(rsqrt, meanSqEps, TENSOR_2x1_F32);

            // %rsqrt_broadcast = broadcast_in_dim(%rsqrt)
            Value rsqrtBroadcast = new Value("rsqrt_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp rsqrtBroadcastOp = new BroadcastInDimOp(
                    rsqrtBroadcast,
                    rsqrt,
                    List.of(0L, 1L),
                    TENSOR_2x8_F32
            );

            // %normalized = multiply(%input, %rsqrt_broadcast)
            Value normalized = new Value("normalized", TENSOR_2x8_F32);
            MultiplyOp normalizedOp = new MultiplyOp(normalized, input, rsqrtBroadcast, TENSOR_2x8_F32);

            // %weight_broadcast = broadcast_in_dim(%weight)
            Value weightBroadcast = new Value("weight_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp weightBroadcastOp = new BroadcastInDimOp(
                    weightBroadcast,
                    weight,
                    List.of(1L),
                    TENSOR_2x8_F32
            );

            // %result = multiply(%normalized, %weight_broadcast)
            Value result = new Value("result", TENSOR_2x8_F32);
            MultiplyOp resultOp = new MultiplyOp(result, normalized, weightBroadcast, TENSOR_2x8_F32);

            return new Function(
                    "rms_norm",
                    List.of(new Argument("arg0", TENSOR_2x8_F32),
                            new Argument("arg1", TENSOR_8_F32)),
                    List.of(TENSOR_2x8_F32),
                    List.of(initSumOp, epsOp, divConstOp, sqOp, sumSqOp, divBroadcastOp,
                            meanSqOp, epsBroadcastOp, meanSqEpsOp, rsqrtOp, rsqrtBroadcastOp,
                            normalizedOp, weightBroadcastOp, resultOp),
                    true
            );
        }

        @Test
        @DisplayName("fuses RMSNorm pattern into single operation")
        void fusesRMSNormPattern() {
            Function func = buildRMSNormFunction();
            Module module = new Module("test", List.of(func));

            FusionPass pass = FusionPass.withStandardPatterns();
            Module result = pass.apply(module);

            assertEquals(1, pass.lastFusionCount(), "Should fuse one RMSNorm pattern");

            Function fusedFunc = result.functions().get(0);
            boolean hasFusedRMSNorm = fusedFunc.body().stream()
                    .anyMatch(op -> op instanceof FusedOperation fused &&
                                    fused.fusionType().equals(FusedOperation.RMS_NORM));

            assertTrue(hasFusedRMSNorm, "Should contain fused RMSNorm operation");
        }
    }

    // ==================== LayerNorm Fusion Tests ====================

    @Nested
    @DisplayName("LayerNorm Fusion")
    class LayerNormFusionTests {

        /**
         * Builds a LayerNorm pattern:
         * %mean = reduce_add(%input, dim=-1) / N
         * %centered = subtract(%input, %mean_broadcast)
         * %sq = multiply(%centered, %centered)
         * %var = reduce_add(%sq, dim=-1) / N
         * %std = sqrt(%var + epsilon)
         * %normalized = divide(%centered, %std_broadcast)
         * %scaled = multiply(%normalized, %weight)
         * %result = add(%scaled, %bias)
         */
        private Function buildLayerNormFunction() {
            Value input = new Value("arg0", TENSOR_2x8_F32);
            Value weight = new Value("arg1", TENSOR_8_F32);
            Value bias = new Value("arg2", TENSOR_8_F32);
            Value initSum = new Value("init_sum", SCALAR_F32);

            // Constants
            ConstantOp initSumOp = new ConstantOp(
                    new Value("init_sum", SCALAR_F32),
                    new DenseAttr(0.0f, SCALAR_F32),
                    SCALAR_F32
            );
            ConstantOp divConstOp = new ConstantOp(
                    new Value("div_const", SCALAR_F32),
                    new DenseAttr(8.0f, SCALAR_F32),
                    SCALAR_F32
            );
            ConstantOp epsOp = new ConstantOp(
                    new Value("eps", SCALAR_F32),
                    new DenseAttr(1e-5f, SCALAR_F32),
                    SCALAR_F32
            );

            // %sum = reduce_add(%input, dim=1)
            Value sum = new Value("sum", TENSOR_2x1_F32);
            ReduceOp sumOp = new ReduceOp(
                    sum,
                    input,
                    initSumOp.result(),
                    List.of(1L),
                    "add",
                    TENSOR_2x1_F32
            );

            // %div_broadcast = broadcast_in_dim(%div_const)
            Value divBroadcast = new Value("div_broadcast", TENSOR_2x1_F32);
            BroadcastInDimOp divBroadcastOp = new BroadcastInDimOp(
                    divBroadcast,
                    divConstOp.result(),
                    List.of(),
                    TENSOR_2x1_F32
            );

            // %mean = divide(%sum, %div_broadcast)
            Value mean = new Value("mean", TENSOR_2x1_F32);
            DivideOp meanOp = new DivideOp(mean, sum, divBroadcast, TENSOR_2x1_F32);

            // %mean_broadcast = broadcast_in_dim(%mean)
            Value meanBroadcast = new Value("mean_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp meanBroadcastOp = new BroadcastInDimOp(
                    meanBroadcast,
                    mean,
                    List.of(0L, 1L),
                    TENSOR_2x8_F32
            );

            // %centered = subtract(%input, %mean_broadcast)
            Value centered = new Value("centered", TENSOR_2x8_F32);
            SubtractOp centeredOp = new SubtractOp(centered, input, meanBroadcast, TENSOR_2x8_F32);

            // %sq = multiply(%centered, %centered)
            Value sq = new Value("sq", TENSOR_2x8_F32);
            MultiplyOp sqOp = new MultiplyOp(sq, centered, centered, TENSOR_2x8_F32);

            // %sum_sq = reduce_add(%sq, dim=1)
            Value sumSq = new Value("sum_sq", TENSOR_2x1_F32);
            ReduceOp sumSqOp = new ReduceOp(
                    sumSq,
                    sq,
                    initSumOp.result(),
                    List.of(1L),
                    "add",
                    TENSOR_2x1_F32
            );

            // %var = divide(%sum_sq, %div_broadcast)
            Value var = new Value("var", TENSOR_2x1_F32);
            DivideOp varOp = new DivideOp(var, sumSq, divBroadcast, TENSOR_2x1_F32);

            // %eps_broadcast = broadcast_in_dim(%eps)
            Value epsBroadcast = new Value("eps_broadcast", TENSOR_2x1_F32);
            BroadcastInDimOp epsBroadcastOp = new BroadcastInDimOp(
                    epsBroadcast,
                    epsOp.result(),
                    List.of(),
                    TENSOR_2x1_F32
            );

            // %var_eps = add(%var, %eps_broadcast)
            Value varEps = new Value("var_eps", TENSOR_2x1_F32);
            AddOp varEpsOp = new AddOp(varEps, var, epsBroadcast, TENSOR_2x1_F32);

            // %std = sqrt(%var_eps)
            Value std = new Value("std", TENSOR_2x1_F32);
            SqrtOp stdOp = new SqrtOp(std, varEps, TENSOR_2x1_F32);

            // %std_broadcast = broadcast_in_dim(%std)
            Value stdBroadcast = new Value("std_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp stdBroadcastOp = new BroadcastInDimOp(
                    stdBroadcast,
                    std,
                    List.of(0L, 1L),
                    TENSOR_2x8_F32
            );

            // %normalized = divide(%centered, %std_broadcast)
            Value normalized = new Value("normalized", TENSOR_2x8_F32);
            DivideOp normalizedOp = new DivideOp(normalized, centered, stdBroadcast, TENSOR_2x8_F32);

            // %weight_broadcast = broadcast_in_dim(%weight)
            Value weightBroadcast = new Value("weight_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp weightBroadcastOp = new BroadcastInDimOp(
                    weightBroadcast,
                    weight,
                    List.of(1L),
                    TENSOR_2x8_F32
            );

            // %scaled = multiply(%normalized, %weight_broadcast)
            Value scaled = new Value("scaled", TENSOR_2x8_F32);
            MultiplyOp scaledOp = new MultiplyOp(scaled, normalized, weightBroadcast, TENSOR_2x8_F32);

            // %bias_broadcast = broadcast_in_dim(%bias)
            Value biasBroadcast = new Value("bias_broadcast", TENSOR_2x8_F32);
            BroadcastInDimOp biasBroadcastOp = new BroadcastInDimOp(
                    biasBroadcast,
                    bias,
                    List.of(1L),
                    TENSOR_2x8_F32
            );

            // %result = add(%scaled, %bias_broadcast)
            Value result = new Value("result", TENSOR_2x8_F32);
            AddOp resultOp = new AddOp(result, scaled, biasBroadcast, TENSOR_2x8_F32);

            return new Function(
                    "layer_norm",
                    List.of(new Argument("arg0", TENSOR_2x8_F32),
                            new Argument("arg1", TENSOR_8_F32),
                            new Argument("arg2", TENSOR_8_F32)),
                    List.of(TENSOR_2x8_F32),
                    List.of(initSumOp, divConstOp, epsOp, sumOp, divBroadcastOp, meanOp,
                            meanBroadcastOp, centeredOp, sqOp, sumSqOp, varOp, epsBroadcastOp,
                            varEpsOp, stdOp, stdBroadcastOp, normalizedOp, weightBroadcastOp,
                            scaledOp, biasBroadcastOp, resultOp),
                    true
            );
        }

        @Test
        @DisplayName("fuses LayerNorm pattern into single operation")
        void fusesLayerNormPattern() {
            Function func = buildLayerNormFunction();
            Module module = new Module("test", List.of(func));

            FusionPass pass = FusionPass.withStandardPatterns();
            Module result = pass.apply(module);

            assertEquals(1, pass.lastFusionCount(), "Should fuse one LayerNorm pattern");

            Function fusedFunc = result.functions().get(0);
            boolean hasFusedLayerNorm = fusedFunc.body().stream()
                    .anyMatch(op -> op instanceof FusedOperation fused &&
                                    fused.fusionType().equals(FusedOperation.LAYER_NORM));

            assertTrue(hasFusedLayerNorm, "Should contain fused LayerNorm operation");
        }
    }

    // ==================== Multiple Pattern Tests ====================

    @Nested
    @DisplayName("Multiple Patterns")
    class MultiplePatternTests {

        @Test
        @DisplayName("fuses multiple patterns in same module")
        void fusesMultiplePatternsInModule() {
            // Build a module with both softmax and RMSNorm patterns
            Function softmaxFunc = new SoftmaxFusionTests().buildSoftmaxFunction();
            Function rmsNormFunc = new RMSNormFusionTests().buildRMSNormFunction();

            Module module = new Module("test", List.of(softmaxFunc, rmsNormFunc));

            FusionPass pass = FusionPass.withStandardPatterns();
            Module result = pass.apply(module);

            assertEquals(2, pass.lastFusionCount(), "Should fuse two patterns");

            // Verify both functions have fused operations
            boolean hasSoftmax = result.functions().get(0).body().stream()
                    .anyMatch(op -> op instanceof FusedOperation fused &&
                                    fused.fusionType().equals(FusedOperation.SOFTMAX));
            boolean hasRMSNorm = result.functions().get(1).body().stream()
                    .anyMatch(op -> op instanceof FusedOperation fused &&
                                    fused.fusionType().equals(FusedOperation.RMS_NORM));

            assertTrue(hasSoftmax, "First function should have fused softmax");
            assertTrue(hasRMSNorm, "Second function should have fused RMSNorm");
        }
    }

    // ==================== Non-Matching Pattern Tests ====================

    @Nested
    @DisplayName("Non-Matching Patterns")
    class NonMatchingPatternTests {

        @Test
        @DisplayName("does not fuse incomplete softmax pattern")
        void doesNotFuseIncompleteSoftmax() {
            Value input = new Value("arg0", TENSOR_2x8_F32);

            // Just exp and divide - not a full softmax
            Value exp = new Value("exp", TENSOR_2x8_F32);
            ExpOp expOp = new ExpOp(exp, input, TENSOR_2x8_F32);

            Value result = new Value("result", TENSOR_2x8_F32);
            DivideOp divOp = new DivideOp(result, exp, exp, TENSOR_2x8_F32);

            Function func = new Function(
                    "incomplete_softmax",
                    List.of(new Argument("arg0", TENSOR_2x8_F32)),
                    List.of(TENSOR_2x8_F32),
                    List.of(expOp, divOp),
                    true
            );

            Module module = new Module("test", List.of(func));
            FusionPass pass = FusionPass.withStandardPatterns();
            Module result2 = pass.apply(module);

            assertEquals(0, pass.lastFusionCount(), "Should not fuse incomplete pattern");
            assertEquals(2, result2.functions().get(0).body().size(),
                    "Should preserve original operations");
        }

        @Test
        @DisplayName("preserves operations that cannot be fused")
        void preservesNonFusableOperations() {
            Value input = new Value("arg0", TENSOR_2x8_F32);

            // Simple add - not a fusion pattern
            Value result = new Value("result", TENSOR_2x8_F32);
            AddOp addOp = new AddOp(result, input, input, TENSOR_2x8_F32);

            Function func = new Function(
                    "simple_add",
                    List.of(new Argument("arg0", TENSOR_2x8_F32)),
                    List.of(TENSOR_2x8_F32),
                    List.of(addOp),
                    true
            );

            Module module = new Module("test", List.of(func));
            FusionPass pass = FusionPass.withStandardPatterns();
            Module result2 = pass.apply(module);

            assertEquals(0, pass.lastFusionCount());
            assertEquals(1, result2.functions().get(0).body().size());
            assertInstanceOf(AddOp.class, result2.functions().get(0).body().get(0));
        }
    }

    // ==================== FusedOperation Verification Tests ====================

    @Nested
    @DisplayName("FusedOperation Verification")
    class FusedOperationVerificationTests {

        @Test
        @DisplayName("fused operation has correct operands and results")
        void fusedOperationHasCorrectOperandsAndResults() {
            Function func = new SoftmaxFusionTests().buildSoftmaxFunction();
            Module module = new Module("test", List.of(func));

            FusionPass pass = FusionPass.withStandardPatterns();
            Module result = pass.apply(module);

            Function fusedFunc = result.functions().get(0);
            FusedOperation fusedOp = fusedFunc.body().stream()
                    .filter(op -> op instanceof FusedOperation)
                    .map(op -> (FusedOperation) op)
                    .findFirst()
                    .orElseThrow();

            assertEquals("fused.softmax", fusedOp.opName());
            assertEquals(1, fusedOp.operands().size(), "Softmax has one input");
            assertEquals(1, fusedOp.results().size(), "Softmax has one output");
            assertEquals("arg0", fusedOp.operands().get(0).name(), "Input should be arg0");
            assertEquals(TENSOR_2x8_F32, fusedOp.tensorResultType());
        }

        @Test
        @DisplayName("fused operation preserves original ops for debugging")
        void fusedOperationPreservesOriginalOps() {
            Function func = new SoftmaxFusionTests().buildSoftmaxFunction();
            Module module = new Module("test", List.of(func));

            FusionPass pass = FusionPass.withStandardPatterns();
            pass.apply(module);

            FusionMatch match = pass.lastMatches().get(0);
            assertTrue(match.matchedOps().size() >= 5,
                    "Should capture at least 5 ops: max, subtract, exp, sum, divide");
        }
    }
}
