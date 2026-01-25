package io.surfworks.snakeburger.fusion;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Fusion pattern for Root Mean Square Normalization (RMSNorm).
 *
 * <p>RMSNorm is simpler than LayerNorm - it doesn't subtract the mean.
 * Used in LLaMA, Gemma, and other modern LLMs.
 *
 * <p>Matches the following StableHLO pattern:
 * <pre>
 * %sq = stablehlo.multiply(%input, %input)
 * %mean_sq = stablehlo.reduce(%sq, add, dimensions=[axis]) / N
 * %rms = stablehlo.rsqrt(%mean_sq + epsilon)  // or sqrt then divide
 * %normalized = stablehlo.multiply(%input, %rms_broadcast)
 * %result = stablehlo.multiply(%normalized, %weight)
 * </pre>
 *
 * <p>Fused output:
 * <pre>
 * %result = fused.rms_norm(%input, %weight, epsilon=eps, axis=axis)
 * </pre>
 *
 * <p>Expected speedup: ~2.5x due to:
 * <ul>
 *   <li>Single pass for RMS calculation</li>
 *   <li>No mean computation (simpler than LayerNorm)</li>
 *   <li>No intermediate tensors for sq, mean_sq</li>
 * </ul>
 */
public final class RMSNormFusion implements FusionPattern {

    private static final double DEFAULT_EPSILON = 1e-6;

    @Override
    public String name() {
        return FusedOperation.RMS_NORM;
    }

    @Override
    public String description() {
        return "Fuses square-mean-rsqrt-normalize-scale into rms_norm";
    }

    @Override
    public double estimatedSpeedup() {
        return 2.5;
    }

    @Override
    public Optional<FusionMatch> match(Operation op, OperationGraph graph) {
        // Pattern root: multiply(normalized, weight) - the final scaling
        if (!(op instanceof MultiplyOp scaleMul)) {
            return Optional.empty();
        }

        // One operand should be the normalized value, other is weight
        Value weightValue;
        Operation normalizeOp = graph.producer(scaleMul.lhs());
        if (isNormalizeOp(normalizeOp)) {
            weightValue = scaleMul.rhs();
        } else {
            normalizeOp = graph.producer(scaleMul.rhs());
            weightValue = scaleMul.lhs();
            if (!isNormalizeOp(normalizeOp)) {
                return Optional.empty();
            }
        }

        // The normalize op should be multiply(input, rsqrt) or divide(input, sqrt)
        Value inputValue;
        Operation rsqrtOrSqrtOp;
        List<Operation> normOps = new ArrayList<>();

        if (normalizeOp instanceof MultiplyOp normMul) {
            // normalized = input * rsqrt(rms)
            Operation lhsProducer = graph.producer(normMul.lhs());
            Operation rhsProducer = graph.producer(normMul.rhs());
            Value rsqrtValue = null;  // Track which value leads to rsqrt

            if (rhsProducer instanceof RsqrtOp) {
                inputValue = normMul.lhs();
                rsqrtOrSqrtOp = rhsProducer;
            } else if (lhsProducer instanceof RsqrtOp) {
                inputValue = normMul.rhs();
                rsqrtOrSqrtOp = lhsProducer;
            } else {
                // Try tracing through broadcasts
                rsqrtOrSqrtOp = traceToRsqrt(graph, normMul.rhs());
                if (rsqrtOrSqrtOp != null) {
                    inputValue = normMul.lhs();
                    rsqrtValue = normMul.rhs();
                } else {
                    rsqrtOrSqrtOp = traceToRsqrt(graph, normMul.lhs());
                    if (rsqrtOrSqrtOp != null) {
                        inputValue = normMul.rhs();
                        rsqrtValue = normMul.lhs();
                    } else {
                        return Optional.empty();
                    }
                }
            }
            normOps.add(rsqrtOrSqrtOp);
            // Add broadcasts between rsqrt and normMul
            if (rsqrtValue != null) {
                addBroadcastsBetween(graph, rsqrtOrSqrtOp.results().get(0), rsqrtValue, normOps);
            }
            normOps.add(normMul);
        } else if (normalizeOp instanceof DivideOp normDiv) {
            // normalized = input / sqrt(rms)
            inputValue = normDiv.lhs();
            rsqrtOrSqrtOp = traceToSqrt(graph, normDiv.rhs());
            if (rsqrtOrSqrtOp == null) {
                return Optional.empty();
            }
            normOps.add(normDiv);
            normOps.add(rsqrtOrSqrtOp);
        } else {
            return Optional.empty();
        }

        // The rsqrt/sqrt input should come from: reduce(sq) + epsilon
        // Where sq = input * input
        Value rsqrtInput = rsqrtOrSqrtOp.operands().get(0);
        Operation addEpsOp = graph.producer(rsqrtInput);

        // Trace through to find the reduce
        Operation reduceOp = traceToReduce(graph, rsqrtInput);
        if (!(reduceOp instanceof ReduceOp meanSqReduce && isSumReduction(meanSqReduce))) {
            return Optional.empty();
        }

        // The reduce input should be sq = input * input
        Operation sqOp = graph.producer(meanSqReduce.operand());
        if (!(sqOp instanceof MultiplyOp sqMul)) {
            return Optional.empty();
        }

        // Verify sq is input * input
        if (!sqMul.lhs().equals(sqMul.rhs())) {
            // Could be input * input with different value references
            // Check if both trace to the same source
            if (!tracesToSameSource(graph, sqMul.lhs(), sqMul.rhs(), inputValue)) {
                return Optional.empty();
            }
        }

        // Verify the input to sq matches the normalize input
        if (!tracesToValue(graph, sqMul.lhs(), inputValue) &&
            !tracesToValue(graph, sqMul.rhs(), inputValue)) {
            return Optional.empty();
        }

        // Extract axis from the reduce
        List<Long> dims = meanSqReduce.dimensions();
        int axis = dims.isEmpty() ? -1 : dims.get(dims.size() - 1).intValue();

        // Collect all matched operations - including all intermediate broadcasts and ops
        List<Operation> matchedOps = new ArrayList<>();
        matchedOps.add(sqMul);
        matchedOps.add(meanSqReduce);

        // Add all operations between reduce result and rsqrt input (broadcasts, divides, adds)
        addIntermediateOps(graph, meanSqReduce.result(), rsqrtInput, matchedOps);

        // Add rsqrt and any broadcasts after it
        matchedOps.addAll(normOps);

        // If normalizeOp is different from scaleMul, add it and intermediate broadcasts
        if (normalizeOp != scaleMul) {
            if (!matchedOps.contains(normalizeOp)) {
                matchedOps.add(normalizeOp);
            }
            // Add broadcasts between normalizeOp result and scaleMul operand
            addIntermediateOps(graph, normalizeOp.results().get(0), scaleMul.lhs(), matchedOps);
            addIntermediateOps(graph, normalizeOp.results().get(0), scaleMul.rhs(), matchedOps);
        }

        matchedOps.add(scaleMul);

        return Optional.of(new FusionMatch(
                name(),
                matchedOps,
                Map.of(
                        "input", inputValue,
                        "output", scaleMul.result(),
                        "weight", weightValue
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

        return new FusedOperation(
                name(),
                List.of(input, weight),
                List.of(output),
                match.attributes(),
                match.matchedOps()
        );
    }

    private boolean isNormalizeOp(Operation op) {
        return op instanceof MultiplyOp || op instanceof DivideOp;
    }

    private boolean isSumReduction(ReduceOp reduce) {
        return "add".equals(reduce.reducer());
    }

    private Operation traceToRsqrt(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        while (producer != null && isBroadcast(producer)) {
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        return (producer instanceof RsqrtOp) ? producer : null;
    }

    private Operation traceToSqrt(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        while (producer != null && isBroadcast(producer)) {
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        return (producer instanceof SqrtOp) ? producer : null;
    }

    private Operation traceToReduce(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        // Skip through broadcasts, adds (for epsilon), divides (for mean)
        while (producer != null && !isReduceOp(producer)) {
            if (producer.operands().isEmpty()) {
                return null;
            }
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        return producer;
    }

    private boolean isReduceOp(Operation op) {
        return op instanceof ReduceOp;
    }

    private boolean isBroadcast(Operation op) {
        return op.opName().contains("broadcast");
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
     * Adds all operations between source and target values to the matched ops list.
     * This includes broadcasts, divides (for mean), and adds (for epsilon).
     */
    private void addIntermediateOps(OperationGraph graph, Value source, Value target,
                                     List<Operation> matchedOps) {
        if (source.equals(target) || source.name().equals(target.name())) {
            return;
        }
        Value current = target;
        while (!current.equals(source) && !current.name().equals(source.name())) {
            Operation producer = graph.producer(current);
            if (producer == null) {
                break;
            }
            if (!matchedOps.contains(producer)) {
                matchedOps.add(producer);
            }
            if (producer.operands().isEmpty()) {
                break;
            }
            current = producer.operands().get(0);
        }
    }

    private boolean tracesToValue(OperationGraph graph, Value value, Value target) {
        if (value.equals(target)) {
            return true;
        }
        if (value.name().equals(target.name())) {
            return true;
        }
        Operation producer = graph.producer(value);
        while (producer != null && isBroadcast(producer)) {
            Value input = producer.operands().get(0);
            if (input.equals(target) || input.name().equals(target.name())) {
                return true;
            }
            producer = graph.producer(input);
        }
        return false;
    }

    private boolean tracesToSameSource(OperationGraph graph, Value a, Value b, Value expected) {
        return tracesToValue(graph, a, expected) && tracesToValue(graph, b, expected);
    }
}
