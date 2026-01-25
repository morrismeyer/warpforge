package io.surfworks.snakeburger.fusion;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;

/**
 * Provides def-use chain analysis for a StableHLO function.
 *
 * <p>OperationGraph builds maps from:
 * <ul>
 *   <li>Value → Operation that produces it (producer)</li>
 *   <li>Value → List of Operations that consume it (consumers)</li>
 * </ul>
 *
 * <p>This enables efficient pattern matching by allowing traversal of the
 * operation graph in both directions.
 *
 * <p>Example:
 * <pre>{@code
 * OperationGraph graph = OperationGraph.build(function);
 *
 * // Find what produces a value
 * Operation producer = graph.producer(value);
 *
 * // Find what consumes a value
 * List<Operation> users = graph.consumers(value);
 *
 * // Check if a value has only one use (safe to fuse without duplication)
 * boolean canFuse = graph.hasSingleUse(value);
 * }</pre>
 */
public final class OperationGraph {

    private final Map<Value, Operation> producers;
    private final Map<Value, List<Operation>> consumers;
    private final Map<String, Value> valuesByName;

    private OperationGraph(
            Map<Value, Operation> producers,
            Map<Value, List<Operation>> consumers,
            Map<String, Value> valuesByName) {
        this.producers = producers;
        this.consumers = consumers;
        this.valuesByName = valuesByName;
    }

    /**
     * Builds an OperationGraph from a function's body.
     *
     * @param func the function to analyze
     * @return the operation graph
     */
    public static OperationGraph build(Function func) {
        Map<Value, Operation> producers = new HashMap<>();
        Map<Value, List<Operation>> consumers = new HashMap<>();
        Map<String, Value> valuesByName = new HashMap<>();

        // Register function arguments as available values (no producer)
        for (var arg : func.arguments()) {
            Value argValue = new Value(arg.name(), arg.type());
            valuesByName.put(arg.name(), argValue);
        }

        // Process each operation
        for (Operation op : func.body()) {
            // Record this op as producer of its results
            for (Value result : op.results()) {
                producers.put(result, op);
                valuesByName.put(result.name(), result);
            }

            // Record this op as consumer of its operands
            for (Value operand : op.operands()) {
                consumers.computeIfAbsent(operand, k -> new ArrayList<>()).add(op);
            }
        }

        return new OperationGraph(producers, consumers, valuesByName);
    }

    /**
     * Returns the operation that produces the given value.
     *
     * @param value the value to look up
     * @return the producing operation, or null if the value is a function argument
     */
    public Operation producer(Value value) {
        return producers.get(value);
    }

    /**
     * Returns all operations that consume the given value.
     *
     * @param value the value to look up
     * @return list of consuming operations (empty if none)
     */
    public List<Operation> consumers(Value value) {
        return consumers.getOrDefault(value, List.of());
    }

    /**
     * Returns true if the value has exactly one consumer.
     *
     * <p>This is important for fusion: if a value has multiple consumers,
     * fusing it might require duplicating computation.
     *
     * @param value the value to check
     * @return true if the value has exactly one consumer
     */
    public boolean hasSingleUse(Value value) {
        return consumers(value).size() == 1;
    }

    /**
     * Returns true if the value has no consumers (dead code).
     *
     * @param value the value to check
     * @return true if the value is unused
     */
    public boolean isUnused(Value value) {
        return consumers(value).isEmpty();
    }

    /**
     * Returns the number of times a value is used.
     *
     * @param value the value to check
     * @return the use count
     */
    public int useCount(Value value) {
        return consumers(value).size();
    }

    /**
     * Looks up a value by its SSA name.
     *
     * @param name the SSA name (without % prefix)
     * @return the value, or null if not found
     */
    public Value valueByName(String name) {
        return valuesByName.get(name);
    }

    /**
     * Returns true if the given value is a function argument (not produced by any op).
     *
     * @param value the value to check
     * @return true if the value is a function argument
     */
    public boolean isFunctionArgument(Value value) {
        return valuesByName.containsKey(value.name()) && !producers.containsKey(value);
    }

    /**
     * Checks if all intermediate values in a chain have single use.
     *
     * <p>This is used to verify that a subgraph can be safely fused without
     * duplicating computation. If any intermediate result is used elsewhere,
     * fusion might cause the computation to be done multiple times.
     *
     * @param ops the operations in the potential fusion
     * @param finalResult the final result that's allowed to have multiple uses
     * @return true if all intermediates have single use
     */
    public boolean canFuseWithoutDuplication(List<Operation> ops, Value finalResult) {
        for (Operation op : ops) {
            for (Value result : op.results()) {
                // Skip the final result - it's OK if that has multiple uses
                if (result.equals(finalResult)) {
                    continue;
                }
                // Intermediate results must have single use
                if (!hasSingleUse(result)) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return String.format("OperationGraph[producers=%d, values=%d]",
                producers.size(), valuesByName.size());
    }
}
