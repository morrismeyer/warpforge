package io.surfworks.warpforge.core.graph;

import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.Collections;
import java.util.List;

/**
 * A compiled StableHLO function ready for execution.
 * Contains topologically sorted operations and tensor specifications.
 */
public final class ExecutableGraph {

    private final String name;
    private final List<GraphNode> nodes;
    private final List<TensorSpec> tensorSpecs;
    private final int[] inputIndices;
    private final int[] outputIndices;
    private final int tensorCount;

    ExecutableGraph(
        String name,
        List<GraphNode> nodes,
        List<TensorSpec> tensorSpecs,
        int[] inputIndices,
        int[] outputIndices
    ) {
        this.name = name;
        this.nodes = List.copyOf(nodes);
        this.tensorSpecs = List.copyOf(tensorSpecs);
        this.inputIndices = inputIndices.clone();
        this.outputIndices = outputIndices.clone();
        this.tensorCount = tensorSpecs.size();
    }

    /**
     * Get the function name.
     */
    public String name() {
        return name;
    }

    /**
     * Get the topologically sorted nodes.
     */
    public List<GraphNode> nodes() {
        return nodes;
    }

    /**
     * Get the tensor specifications for all intermediate values.
     */
    public List<TensorSpec> tensorSpecs() {
        return tensorSpecs;
    }

    /**
     * Get tensor spec for a specific index.
     */
    public TensorSpec tensorSpec(int index) {
        return tensorSpecs.get(index);
    }

    /**
     * Get the indices of input tensors (function arguments).
     */
    public int[] inputIndices() {
        return inputIndices.clone();
    }

    /**
     * Get the indices of output tensors (return values).
     */
    public int[] outputIndices() {
        return outputIndices.clone();
    }

    /**
     * Get the number of function inputs.
     */
    public int inputCount() {
        return inputIndices.length;
    }

    /**
     * Get the number of function outputs.
     */
    public int outputCount() {
        return outputIndices.length;
    }

    /**
     * Get the total number of tensors needed for execution.
     */
    public int tensorCount() {
        return tensorCount;
    }

    /**
     * Get the number of operations (excluding return).
     */
    public int operationCount() {
        return (int) nodes.stream().filter(n -> !n.isReturn()).count();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ExecutableGraph '").append(name).append("'\n");
        sb.append("  Inputs: ").append(inputIndices.length).append("\n");
        sb.append("  Outputs: ").append(outputIndices.length).append("\n");
        sb.append("  Tensors: ").append(tensorCount).append("\n");
        sb.append("  Operations: ").append(operationCount()).append("\n");
        sb.append("  Nodes:\n");
        for (int i = 0; i < nodes.size(); i++) {
            sb.append("    [").append(i).append("] ").append(nodes.get(i)).append("\n");
        }
        return sb.toString();
    }
}
