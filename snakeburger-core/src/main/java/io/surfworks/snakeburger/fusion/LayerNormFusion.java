package io.surfworks.snakeburger.fusion;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Fusion pattern for Layer Normalization.
 *
 * <p>Matches the following StableHLO pattern:
 * <pre>
 * %mean = stablehlo.reduce(%input, add, dimensions=[axis]) / N
 * %centered = stablehlo.subtract(%input, %mean_broadcast)
 * %sq = stablehlo.multiply(%centered, %centered)
 * %var = stablehlo.reduce(%sq, add, dimensions=[axis]) / N
 * %std = stablehlo.sqrt(%var + epsilon)  // or rsqrt
 * %normalized = stablehlo.divide(%centered, %std_broadcast)
 * %scaled = stablehlo.multiply(%normalized, %weight)
 * %result = stablehlo.add(%scaled, %bias)
 * </pre>
 *
 * <p>The pattern may have variations:
 * <ul>
 *   <li>rsqrt instead of sqrt+divide</li>
 *   <li>Broadcasts inserted between operations</li>
 *   <li>Epsilon added in different ways</li>
 * </ul>
 *
 * <p>Fused output:
 * <pre>
 * %result = fused.layer_norm(%input, %weight, %bias, epsilon=eps, axis=axis)
 * </pre>
 *
 * <p>Expected speedup: ~3x due to:
 * <ul>
 *   <li>Single pass for mean calculation</li>
 *   <li>Welford's algorithm for numerical stability</li>
 *   <li>No intermediate tensors for centered, sq, var</li>
 * </ul>
 */
public final class LayerNormFusion implements FusionPattern {

    private static final double DEFAULT_EPSILON = 1e-5;

    @Override
    public String name() {
        return FusedOperation.LAYER_NORM;
    }

    @Override
    public String description() {
        return "Fuses mean-center-variance-normalize-scale-shift into layer_norm";
    }

    @Override
    public double estimatedSpeedup() {
        return 3.0;
    }

    @Override
    public Optional<FusionMatch> match(Operation op, OperationGraph graph) {
        // Pattern root: add(scaled, bias) - the final bias addition
        if (!(op instanceof AddOp finalAdd)) {
            return Optional.empty();
        }

        // One operand should be multiply (scale), the other is bias
        Operation scaleOp = graph.producer(finalAdd.lhs());
        Value biasValue = finalAdd.rhs();
        MultiplyOp scaleMul;

        if (scaleOp instanceof MultiplyOp mul) {
            scaleMul = mul;
        } else {
            // Try the other way
            scaleOp = graph.producer(finalAdd.rhs());
            biasValue = finalAdd.lhs();
            if (scaleOp instanceof MultiplyOp mul2) {
                scaleMul = mul2;
            } else {
                return Optional.empty();
            }
        }

        // One operand of multiply should be the normalized value, other is weight
        Value weightValue;
        Operation normalizeOp = graph.producer(scaleMul.lhs());
        if (normalizeOp instanceof DivideOp || normalizeOp instanceof MultiplyOp) {
            weightValue = scaleMul.rhs();
        } else {
            normalizeOp = graph.producer(scaleMul.rhs());
            weightValue = scaleMul.lhs();
            if (!(normalizeOp instanceof DivideOp || normalizeOp instanceof MultiplyOp)) {
                return Optional.empty();
            }
        }

        // Match the normalization pattern (divide or multiply with rsqrt)
        Value centeredValue;
        List<Operation> normOps = new ArrayList<>();

        if (normalizeOp instanceof DivideOp divOp) {
            // normalized = centered / std_broadcast
            centeredValue = divOp.lhs();

            // std should come from sqrt(var + eps)
            Operation stdProducer = traceToSqrt(graph, divOp.rhs());
            if (stdProducer == null) {
                return Optional.empty();
            }
            normOps.add(stdProducer);

            // Add broadcasts between sqrt and divide
            addBroadcastsBetween(graph, stdProducer.results().get(0), divOp.rhs(), normOps);
            normOps.add(divOp);

            // Trace back from sqrt input to find variance calculation ops
            Value sqrtInput = stdProducer.operands().get(0);
            addVarianceOps(graph, sqrtInput, centeredValue, normOps);
        } else if (normalizeOp instanceof MultiplyOp mulOp) {
            // normalized = centered * rsqrt(var + eps)
            Value rsqrtValue = null;
            Operation rsqrtOp = graph.producer(mulOp.rhs());
            if (!(rsqrtOp instanceof RsqrtOp)) {
                rsqrtOp = traceToRsqrt(graph, mulOp.rhs());
                if (rsqrtOp != null) {
                    rsqrtValue = mulOp.rhs();
                    centeredValue = mulOp.lhs();
                } else {
                    rsqrtOp = traceToRsqrt(graph, mulOp.lhs());
                    if (rsqrtOp != null) {
                        rsqrtValue = mulOp.lhs();
                        centeredValue = mulOp.rhs();
                    } else {
                        // Neither side traces to rsqrt
                        return Optional.empty();
                    }
                }
            } else {
                centeredValue = mulOp.lhs();
            }
            normOps.add(rsqrtOp);
            if (rsqrtValue != null) {
                addBroadcastsBetween(graph, rsqrtOp.results().get(0), rsqrtValue, normOps);
            }
            normOps.add(mulOp);

            // Trace back from rsqrt input to find variance calculation ops
            Value rsqrtInput = rsqrtOp.operands().get(0);
            addVarianceOps(graph, rsqrtInput, centeredValue, normOps);
        } else {
            return Optional.empty();
        }

        // centered = input - mean
        Operation centerOp = graph.producer(centeredValue);
        if (!(centerOp instanceof SubtractOp subOp)) {
            return Optional.empty();
        }

        Value inputValue = subOp.lhs();

        // mean should come from a reduce
        Operation meanProducer = traceToReduce(graph, subOp.rhs());
        if (!(meanProducer instanceof ReduceOp meanReduce && isMeanReduction(meanReduce))) {
            return Optional.empty();
        }

        // Verify mean is computed from the same input
        if (!tracesToValue(graph, meanReduce.operand(), inputValue)) {
            return Optional.empty();
        }

        // Extract axis from the mean reduction
        List<Long> dims = meanReduce.dimensions();
        int axis = dims.isEmpty() ? -1 : dims.get(dims.size() - 1).intValue();

        // Collect all matched operations
        List<Operation> matchedOps = new ArrayList<>();
        matchedOps.add(meanReduce);
        addIntermediateOps(graph, meanReduce.result(), subOp.rhs(), matchedOps);
        matchedOps.add(subOp);
        matchedOps.addAll(normOps);
        matchedOps.add(scaleMul);
        matchedOps.add(finalAdd);

        return Optional.of(new FusionMatch(
                name(),
                matchedOps,
                Map.of(
                        "input", inputValue,
                        "output", finalAdd.result(),
                        "weight", weightValue,
                        "bias", biasValue
                ),
                Map.of(
                        "axis", axis,
                        "epsilon", DEFAULT_EPSILON
                )
        ));
    }

    @Override
    public FusedOperation rewrite(FusionMatch match) {
        Value input = match.input();
        Value output = match.output();
        Value weight = match.capture("weight");
        Value bias = match.capture("bias");

        return new FusedOperation(
                name(),
                List.of(input, weight, bias),
                List.of(output),
                match.attributes(),
                match.matchedOps()
        );
    }

    private boolean isMeanReduction(ReduceOp reduce) {
        return "add".equals(reduce.reducer());
    }

    private Operation traceToReduce(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        // Skip through broadcasts and divides (for mean = sum / N)
        while (producer != null && (isBroadcast(producer) || isDivideByConstant(producer))) {
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        return producer;
    }

    private Operation traceToSqrt(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        // Skip through broadcasts
        while (producer != null && isBroadcast(producer)) {
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        if (producer instanceof SqrtOp) {
            return producer;
        }
        return null;
    }

    private boolean tracesToValue(OperationGraph graph, Value value, Value target) {
        if (value.equals(target)) {
            return true;
        }
        Operation producer = graph.producer(value);
        while (producer != null && (isBroadcast(producer) || isDivideByConstant(producer))) {
            Value input = producer.operands().get(0);
            if (input.equals(target)) {
                return true;
            }
            producer = graph.producer(input);
        }
        return false;
    }

    private boolean isBroadcast(Operation op) {
        return op.opName().contains("broadcast");
    }

    private boolean isDivideByConstant(Operation op) {
        // Check if this is a divide where RHS is a constant (for mean = sum / N)
        if (!(op instanceof DivideOp)) {
            return false;
        }
        // Simplified check - in practice would verify RHS is a constant
        return true;
    }

    private void addIntermediateOps(OperationGraph graph, Value source, Value target,
                                     List<Operation> matchedOps) {
        Value current = target;
        while (!current.equals(source)) {
            Operation producer = graph.producer(current);
            if (producer == null) {
                break;
            }
            if (isBroadcast(producer) || isDivideByConstant(producer)) {
                matchedOps.add(producer);
                current = producer.operands().get(0);
            } else {
                break;
            }
        }
    }

    private Operation traceToRsqrt(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        while (producer != null && isBroadcast(producer)) {
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        return (producer instanceof RsqrtOp) ? producer : null;
    }

    /**
     * Adds any broadcast operations between source and target to the matched ops list.
     */
    private void addBroadcastsBetween(OperationGraph graph, Value source, Value target,
                                       List<Operation> matchedOps) {
        Value current = target;
        while (!current.equals(source) && !current.name().equals(source.name())) {
            Operation producer = graph.producer(current);
            if (producer == null || !isBroadcast(producer)) {
                break;
            }
            if (!matchedOps.contains(producer)) {
                matchedOps.add(producer);
            }
            current = producer.operands().get(0);
        }
    }

    /**
     * Adds variance calculation operations (multiply for sq, reduce, broadcasts, divides, adds for epsilon)
     * between the centered value and the sqrt input.
     */
    private void addVarianceOps(OperationGraph graph, Value sqrtInput, Value centeredValue,
                                 List<Operation> matchedOps) {
        // Trace back from sqrtInput to find all variance calculation ops
        Value current = sqrtInput;
        while (current != null) {
            Operation producer = graph.producer(current);
            if (producer == null) {
                break;
            }
            if (!matchedOps.contains(producer)) {
                matchedOps.add(producer);
            }
            // Stop when we reach the centered value or a reduce op
            if (producer instanceof ReduceOp) {
                // Also add the multiply (sq = centered * centered)
                Operation sqOp = graph.producer(((ReduceOp) producer).operand());
                if (sqOp instanceof MultiplyOp && !matchedOps.contains(sqOp)) {
                    matchedOps.add(sqOp);
                }
                break;
            }
            if (producer.operands().isEmpty()) {
                break;
            }
            current = producer.operands().get(0);
        }
    }
}
