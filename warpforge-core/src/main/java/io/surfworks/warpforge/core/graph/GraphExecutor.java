package io.surfworks.warpforge.core.graph;

import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Executes an {@link ExecutableGraph} on a {@link Backend}.
 *
 * <p>The executor manages a tensor buffer array, routes inputs/outputs between
 * operations, and iterates through nodes in topological order.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (CpuBackend backend = new CpuBackend();
 *      Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
 *     GraphExecutor executor = new GraphExecutor(backend);
 *     List<Tensor> outputs = executor.execute(graph, input);
 *     // Use outputs...
 * }
 * }</pre>
 */
public final class GraphExecutor {

    private final Backend backend;

    /**
     * Create an executor using the given backend.
     *
     * @param backend The backend to execute operations on
     */
    public GraphExecutor(Backend backend) {
        this.backend = backend;
    }

    /**
     * Execute the graph with the given inputs.
     *
     * @param graph  The compiled graph to execute
     * @param inputs Input tensors (count must match graph.inputCount())
     * @return Output tensors (count matches graph.outputCount())
     * @throws IllegalArgumentException if input count doesn't match
     * @throws UnsupportedOperationException if an operation is not supported
     */
    public List<Tensor> execute(ExecutableGraph graph, List<Tensor> inputs) {
        // Validate input count
        if (inputs.size() != graph.inputCount()) {
            throw new IllegalArgumentException(
                "Expected " + graph.inputCount() + " inputs, got " + inputs.size());
        }

        // Validate input shapes match graph expectations
        int[] inputIndices = graph.inputIndices();
        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            var expectedSpec = graph.tensorSpec(inputIndices[i]);
            if (!java.util.Arrays.equals(input.shape(), expectedSpec.shape())) {
                throw new IllegalArgumentException(
                    "Input " + i + " shape " + java.util.Arrays.toString(input.shape()) +
                    " doesn't match expected " + java.util.Arrays.toString(expectedSpec.shape()));
            }
        }

        // Create tensor buffer array
        Tensor[] tensors = new Tensor[graph.tensorCount()];

        // Place inputs at their designated indices
        for (int i = 0; i < inputs.size(); i++) {
            tensors[inputIndices[i]] = inputs.get(i);
        }

        // Execute nodes in topological order
        for (GraphNode node : graph.nodes()) {
            if (node.isReturn()) {
                // Return node just marks outputs, no execution needed
                continue;
            }

            // Gather inputs for this operation
            List<Tensor> opInputs = new ArrayList<>(node.inputCount());
            for (int idx : node.inputIndices()) {
                Tensor t = tensors[idx];
                if (t == null) {
                    throw new IllegalStateException(
                        "Tensor at index " + idx + " is null when executing " + node.opName());
                }
                opInputs.add(t);
            }

            // Execute operation
            List<Tensor> opOutputs = backend.execute(node.operation(), opInputs);

            // Store outputs at their designated indices
            int[] outputIndices = node.outputIndices();
            if (opOutputs.size() != outputIndices.length) {
                throw new IllegalStateException(
                    "Operation " + node.opName() + " returned " + opOutputs.size() +
                    " outputs but expected " + outputIndices.length);
            }
            for (int i = 0; i < outputIndices.length; i++) {
                tensors[outputIndices[i]] = opOutputs.get(i);
            }
        }

        // Collect outputs from designated indices
        int[] outputIndices = graph.outputIndices();
        List<Tensor> outputs = new ArrayList<>(outputIndices.length);
        for (int idx : outputIndices) {
            Tensor t = tensors[idx];
            if (t == null) {
                throw new IllegalStateException("Output tensor at index " + idx + " is null");
            }
            outputs.add(t);
        }

        return outputs;
    }

    /**
     * Execute with varargs inputs.
     */
    public List<Tensor> execute(ExecutableGraph graph, Tensor... inputs) {
        return execute(graph, List.of(inputs));
    }

    /**
     * Get the backend used by this executor.
     */
    public Backend backend() {
        return backend;
    }
}
