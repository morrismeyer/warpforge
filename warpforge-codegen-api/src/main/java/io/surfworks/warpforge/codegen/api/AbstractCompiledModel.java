package io.surfworks.warpforge.codegen.api;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base class for compiled models.
 *
 * <p>This class provides the execution logic for compiled models.
 * Generated subclasses provide:
 * <ul>
 *   <li>{@link #operations()} - The list of StableHLO operations</li>
 *   <li>{@link #inputIndices()} - Indices of input tensors</li>
 *   <li>{@link #outputIndices()} - Indices of output tensors</li>
 *   <li>{@link #tensorCount()} - Total number of tensors in the execution buffer</li>
 *   <li>{@link #operationInputIndices()} - Input indices for each operation</li>
 *   <li>{@link #operationOutputIndices()} - Output indices for each operation</li>
 *   <li>{@link #metadata()} - Model metadata</li>
 * </ul>
 */
public abstract class AbstractCompiledModel implements CompiledModel {

    @Override
    public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
        if (inputs.size() != inputCount()) {
            throw new IllegalArgumentException(
                "Expected " + inputCount() + " inputs, got " + inputs.size());
        }

        // Create tensor buffer for all SSA values
        Tensor[] tensors = new Tensor[tensorCount()];

        // Place inputs at their designated indices
        int[] inIndices = inputIndices();
        for (int i = 0; i < inIndices.length; i++) {
            tensors[inIndices[i]] = inputs.get(i);
        }

        // Execute operations in order
        List<StableHloAst.Operation> ops = operations();
        int[][] opInputs = operationInputIndices();
        int[][] opOutputs = operationOutputIndices();

        for (int opIdx = 0; opIdx < ops.size(); opIdx++) {
            StableHloAst.Operation op = ops.get(opIdx);

            // Skip ReturnOp - it just marks the outputs
            if (op instanceof StableHloAst.ReturnOp) {
                continue;
            }

            // Gather inputs for this operation
            int[] inputIdxs = opInputs[opIdx];
            List<Tensor> opInputTensors = new ArrayList<>(inputIdxs.length);
            for (int idx : inputIdxs) {
                opInputTensors.add(tensors[idx]);
            }

            // Execute operation
            List<Tensor> results = backend.execute(op, opInputTensors);

            // Store results
            int[] outputIdxs = opOutputs[opIdx];
            for (int i = 0; i < outputIdxs.length && i < results.size(); i++) {
                tensors[outputIdxs[i]] = results.get(i);
            }
        }

        // Gather outputs
        int[] outIndices = outputIndices();
        List<Tensor> outputs = new ArrayList<>(outIndices.length);
        for (int idx : outIndices) {
            outputs.add(tensors[idx]);
        }

        return outputs;
    }

    @Override
    public int inputCount() {
        return inputIndices().length;
    }

    @Override
    public int outputCount() {
        return outputIndices().length;
    }

    /**
     * Returns the list of StableHLO operations to execute.
     */
    protected abstract List<StableHloAst.Operation> operations();

    /**
     * Returns the tensor indices for function inputs.
     */
    protected abstract int[] inputIndices();

    /**
     * Returns the tensor indices for function outputs.
     */
    protected abstract int[] outputIndices();

    /**
     * Returns the total number of tensors in the execution buffer.
     */
    protected abstract int tensorCount();

    /**
     * Returns the input tensor indices for each operation.
     * operationInputIndices()[opIdx] contains the indices of tensors
     * that are inputs to operation opIdx.
     */
    protected abstract int[][] operationInputIndices();

    /**
     * Returns the output tensor indices for each operation.
     * operationOutputIndices()[opIdx] contains the indices of tensors
     * that are outputs from operation opIdx.
     */
    protected abstract int[][] operationOutputIndices();
}
