package io.surfworks.snakeburger.fusion;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConstantOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DenseAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScalarType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Debug tests to understand fusion pattern matching.
 */
class FusionPassDebugTest {

    private static final TensorType TENSOR_2x8_F32 = new TensorType(List.of(2, 8), ScalarType.F32);
    private static final TensorType TENSOR_2x1_F32 = new TensorType(List.of(2, 1), ScalarType.F32);
    private static final TensorType SCALAR_F32 = new TensorType(List.of(), ScalarType.F32);

    @Test
    void debugSoftmaxGraphBuilding() {
        Value input = new Value("arg0", TENSOR_2x8_F32);

        // Constants
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
                List.of(1L),
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
        Value expValue = new Value("exp", TENSOR_2x8_F32);
        ExpOp expOp = new ExpOp(expValue, shifted, TENSOR_2x8_F32);

        // %sum = reduce_add(%exp, dim=1)
        Value sumResult = new Value("sum", TENSOR_2x1_F32);
        ReduceOp sumOp = new ReduceOp(
                sumResult,
                expValue,  // Using expValue, which is the result of expOp
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
        DivideOp divOp = new DivideOp(result, expValue, sumBroadcast, TENSOR_2x8_F32);

        Function func = new Function(
                "softmax",
                List.of(new Argument("arg0", TENSOR_2x8_F32)),
                List.of(TENSOR_2x8_F32),
                List.of(initMaxOp, initSumOp, maxOp, maxBroadcastOp, shiftedOp,
                        expOp, sumOp, sumBroadcastOp, divOp),
                true
        );

        // Build graph
        OperationGraph graph = OperationGraph.build(func);

        // Debug output
        System.out.println("=== Graph Debug ===");
        System.out.println("divOp.lhs() = " + divOp.lhs() + " (name: " + divOp.lhs().name() + ")");
        System.out.println("expValue = " + expValue + " (name: " + expValue.name() + ")");
        System.out.println("expOp.result() = " + expOp.result() + " (name: " + expOp.result().name() + ")");
        System.out.println("divOp.lhs().equals(expValue) = " + divOp.lhs().equals(expValue));
        System.out.println("divOp.lhs().equals(expOp.result()) = " + divOp.lhs().equals(expOp.result()));

        Operation producerOfDivLhs = graph.producer(divOp.lhs());
        System.out.println("producer(divOp.lhs()) = " + producerOfDivLhs);
        System.out.println("producerOfDivLhs instanceof ExpOp = " + (producerOfDivLhs instanceof ExpOp));

        // Check expOp input
        if (producerOfDivLhs instanceof ExpOp exp) {
            System.out.println("exp.operand() = " + exp.operand() + " (name: " + exp.operand().name() + ")");
            Operation expInputProducer = graph.producer(exp.operand());
            System.out.println("producer(exp.operand()) = " + expInputProducer);
            System.out.println("expInputProducer instanceof SubtractOp = " + (expInputProducer instanceof SubtractOp));

            if (expInputProducer instanceof SubtractOp sub) {
                System.out.println("sub.rhs() = " + sub.rhs() + " (name: " + sub.rhs().name() + ")");
                Operation subRhsProducer = graph.producer(sub.rhs());
                System.out.println("producer(sub.rhs()) = " + subRhsProducer);
            }
        }

        // Check sum operand
        System.out.println("\nsumOp.operand() = " + sumOp.operand() + " (name: " + sumOp.operand().name() + ")");
        System.out.println("sumOp.operand().equals(expOp.result()) = " + sumOp.operand().equals(expOp.result()));

        // Test the pattern matcher directly
        System.out.println("\n=== Pattern Matching ===");
        SoftmaxFusion pattern = new SoftmaxFusion();
        var match = pattern.match(divOp, graph);
        System.out.println("Pattern match result: " + match);

        // Assertions
        assertNotNull(producerOfDivLhs, "div.lhs() should have a producer");
        assertTrue(producerOfDivLhs instanceof ExpOp, "producer should be ExpOp");
        assertTrue(match.isPresent(), "Softmax pattern should match");
    }

    @Test
    void debugRMSNormGraphBuilding() {
        Value input = new Value("arg0", TENSOR_2x8_F32);
        Value weight = new Value("arg1", new TensorType(List.of(8), ScalarType.F32));

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
                new DenseAttr(8.0f, SCALAR_F32),
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

        Function func = new Function(
                "rms_norm",
                List.of(new Argument("arg0", TENSOR_2x8_F32),
                        new Argument("arg1", new TensorType(List.of(8), ScalarType.F32))),
                List.of(TENSOR_2x8_F32),
                List.of(initSumOp, epsOp, divConstOp, sqOp, sumSqOp, divBroadcastOp,
                        meanSqOp, epsBroadcastOp, meanSqEpsOp, rsqrtOp, rsqrtBroadcastOp,
                        normalizedOp, weightBroadcastOp, resultOp),
                true
        );

        // Build graph
        OperationGraph graph = OperationGraph.build(func);

        // Debug: trace through the pattern
        System.out.println("=== RMSNorm Graph Debug ===");
        System.out.println("resultOp is MultiplyOp: " + (resultOp instanceof MultiplyOp));

        // RMSNormFusion starts from the final multiply (scale)
        Operation lhsProducer = graph.producer(resultOp.lhs());
        Operation rhsProducer = graph.producer(resultOp.rhs());
        System.out.println("resultOp.lhs() producer: " + (lhsProducer != null ? lhsProducer.opName() : "null"));
        System.out.println("resultOp.rhs() producer: " + (rhsProducer != null ? rhsProducer.opName() : "null"));

        // The pattern expects one operand to be normalize (multiply with rsqrt) and the other to be weight
        // normalizedOp = multiply(input, rsqrt_broadcast)
        if (lhsProducer instanceof MultiplyOp normMul) {
            System.out.println("LHS is MultiplyOp");
            Operation normLhsProducer = graph.producer(normMul.lhs());
            Operation normRhsProducer = graph.producer(normMul.rhs());
            System.out.println("  normMul.lhs() producer: " + (normLhsProducer != null ? normLhsProducer.opName() : "null (arg)"));
            System.out.println("  normMul.rhs() producer: " + (normRhsProducer != null ? normRhsProducer.opName() : "null (arg)"));

            // Check if either is RsqrtOp or traces to one
            System.out.println("  normRhsProducer instanceof RsqrtOp: " + (normRhsProducer instanceof RsqrtOp));
            if (normRhsProducer instanceof BroadcastInDimOp) {
                Operation bcastInput = graph.producer(((BroadcastInDimOp) normRhsProducer).operand());
                System.out.println("  broadcast input producer: " + (bcastInput != null ? bcastInput.opName() : "null"));
                System.out.println("  broadcast input instanceof RsqrtOp: " + (bcastInput instanceof RsqrtOp));
            }
        }

        // Test the pattern matcher directly
        System.out.println("\n=== Pattern Matching ===");
        RMSNormFusion pattern = new RMSNormFusion();
        var match = pattern.match(resultOp, graph);
        System.out.println("Pattern match result: " + match);

        assertTrue(match.isPresent(), "RMSNorm pattern should match");

        // Test the full FusionPass - manually iterate to debug
        System.out.println("\n=== FusionPass Debug ===");
        System.out.println("func.body() size: " + func.body().size());
        for (int i = 0; i < func.body().size(); i++) {
            Operation op = func.body().get(i);
            System.out.println(i + ": " + op.opName() + " result=" +
                    (op.results().isEmpty() ? "none" : op.results().get(0).name()));
        }

        System.out.println("\nTrying RMSNorm pattern on each MultiplyOp:");
        RMSNormFusion rmsPattern = new RMSNormFusion();
        for (Operation op : func.body()) {
            if (op instanceof MultiplyOp mul) {
                System.out.println("Trying on " + mul.result().name() + "...");
                var m = rmsPattern.match(op, graph);
                System.out.println("  Result: " + m);

                if (m.isPresent()) {
                    // Check canFuseWithoutDuplication
                    boolean canFuse = graph.canFuseWithoutDuplication(m.get().matchedOps(), m.get().output());
                    System.out.println("  canFuseWithoutDuplication: " + canFuse);

                    if (!canFuse) {
                        // Debug: which intermediate has external use?
                        java.util.Set<Operation> opsSet = new java.util.HashSet<>(m.get().matchedOps());
                        for (Operation matchedOp : m.get().matchedOps()) {
                            for (Value res : matchedOp.results()) {
                                if (res.equals(m.get().output())) continue;
                                for (Operation consumer : graph.consumers(res)) {
                                    if (!opsSet.contains(consumer)) {
                                        System.out.println("  External use: " + res.name() + " used by " + consumer.opName());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        io.surfworks.snakeburger.stablehlo.StableHloAst.Module module =
                new io.surfworks.snakeburger.stablehlo.StableHloAst.Module("test", List.of(func));
        FusionPass pass = FusionPass.withStandardPatterns();
        var resultModule = pass.apply(module);
        System.out.println("\nFusions applied: " + pass.lastFusionCount());
        System.out.println("Last matches: " + pass.lastMatches());

        assertEquals(1, pass.lastFusionCount(), "Should fuse one RMSNorm pattern");
    }
}
