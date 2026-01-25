package io.surfworks.snakeburger.fusion;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Fusion pattern for numerically stable softmax.
 *
 * <p>Matches the following StableHLO pattern:
 * <pre>
 * %max = stablehlo.reduce(%input, max, dimensions=[axis])
 * %shifted = stablehlo.subtract(%input, %max_broadcast)
 * %exp = stablehlo.exponential(%shifted)
 * %sum = stablehlo.reduce(%exp, add, dimensions=[axis])
 * %result = stablehlo.divide(%exp, %sum_broadcast)
 * </pre>
 *
 * <p>The pattern accounts for broadcast operations that may be inserted
 * between reduce and element-wise operations.
 *
 * <p>Fused output:
 * <pre>
 * %result = fused.softmax(%input, axis=axis)
 * </pre>
 *
 * <p>Expected speedup: ~3x due to:
 * <ul>
 *   <li>Single pass over data instead of 5 operations</li>
 *   <li>Intermediate values stay in registers</li>
 *   <li>No memory traffic for max, shifted, exp, sum intermediates</li>
 * </ul>
 */
public final class SoftmaxFusion implements FusionPattern {

    @Override
    public String name() {
        return FusedOperation.SOFTMAX;
    }

    @Override
    public String description() {
        return "Fuses max-subtract-exp-sum-divide into softmax";
    }

    @Override
    public double estimatedSpeedup() {
        return 3.0;
    }

    @Override
    public Optional<FusionMatch> match(Operation op, OperationGraph graph) {
        // Pattern root: divide(exp_result, sum_result)
        if (!(op instanceof DivideOp div)) {
            return Optional.empty();
        }

        // The numerator should be exp
        Operation numProducer = graph.producer(div.lhs());
        if (!(numProducer instanceof ExpOp exp)) {
            return Optional.empty();
        }

        // The exp input should be subtract
        Operation expInputProducer = graph.producer(exp.operand());
        if (!(expInputProducer instanceof SubtractOp sub)) {
            return Optional.empty();
        }

        // The subtract RHS should trace back to a max reduce
        // Note: There may be a broadcast in between
        Operation maxProducer = traceToReduce(graph, sub.rhs());
        if (!(maxProducer instanceof ReduceOp maxReduce && isMaxReduction(maxReduce))) {
            return Optional.empty();
        }

        // The denominator should trace back to a sum reduce
        // Note: There may be a broadcast in between
        Operation sumProducer = traceToReduce(graph, div.rhs());
        if (!(sumProducer instanceof ReduceOp sumReduce && isSumReduction(sumReduce))) {
            return Optional.empty();
        }

        // Verify the sum reduction is over the exp result
        if (!tracesToValue(graph, sumReduce.operand(), exp.result())) {
            return Optional.empty();
        }

        // Verify max reduction is over the same input as subtract LHS
        Value softmaxInput = sub.lhs();
        if (!tracesToValue(graph, maxReduce.operand(), softmaxInput)) {
            return Optional.empty();
        }

        // Extract the axis from the reduce operations
        List<Long> dims = maxReduce.dimensions();
        int axis = dims.isEmpty() ? -1 : dims.get(0).intValue();

        // Collect all matched operations
        List<Operation> matchedOps = new ArrayList<>();
        matchedOps.add(maxReduce);
        addBroadcastsBetween(graph, maxReduce.result(), sub.rhs(), matchedOps);
        matchedOps.add(sub);
        matchedOps.add(exp);
        matchedOps.add(sumReduce);
        addBroadcastsBetween(graph, sumReduce.result(), div.rhs(), matchedOps);
        matchedOps.add(div);

        return Optional.of(new FusionMatch(
                name(),
                matchedOps,
                Map.of("input", softmaxInput, "output", div.result()),
                Map.of("axis", axis)
        ));
    }

    @Override
    public FusedOperation rewrite(FusionMatch match) {
        return createFusedOp(match, match.attributes());
    }

    private boolean isMaxReduction(ReduceOp reduce) {
        return "max".equals(reduce.reducer());
    }

    private boolean isSumReduction(ReduceOp reduce) {
        return "add".equals(reduce.reducer());
    }

    /**
     * Traces through broadcast operations to find the underlying reduce.
     */
    private Operation traceToReduce(OperationGraph graph, Value value) {
        Operation producer = graph.producer(value);
        // Skip through broadcasts
        while (producer != null && isBroadcast(producer)) {
            Value input = producer.operands().get(0);
            producer = graph.producer(input);
        }
        return producer;
    }

    /**
     * Checks if a value traces back to a specific target value (possibly through broadcasts).
     */
    private boolean tracesToValue(OperationGraph graph, Value value, Value target) {
        if (value.equals(target)) {
            return true;
        }
        Operation producer = graph.producer(value);
        while (producer != null && isBroadcast(producer)) {
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

    /**
     * Adds any broadcast operations between source and target to the matched ops list.
     */
    private void addBroadcastsBetween(OperationGraph graph, Value source, Value target,
                                       List<Operation> matchedOps) {
        Value current = target;
        while (!current.equals(source)) {
            Operation producer = graph.producer(current);
            if (producer == null || !isBroadcast(producer)) {
                break;
            }
            matchedOps.add(producer);
            current = producer.operands().get(0);
        }
    }
}
