package io.surfworks.warpforge.core.graph;

import io.surfworks.snakeburger.stablehlo.StableHloAst;

import java.util.List;

/**
 * A node in the executable graph representing a single StableHLO operation.
 * Contains resolved tensor indices for inputs and outputs.
 *
 * @param operation    The underlying StableHLO operation
 * @param inputIndices Indices into the tensor array for each input operand
 * @param outputIndices Indices into the tensor array for each output result
 */
public record GraphNode(
    StableHloAst.Operation operation,
    int[] inputIndices,
    int[] outputIndices
) {
    /**
     * Get the operation name.
     */
    public String opName() {
        return operation.opName();
    }

    /**
     * Get the number of inputs.
     */
    public int inputCount() {
        return inputIndices.length;
    }

    /**
     * Get the number of outputs.
     */
    public int outputCount() {
        return outputIndices.length;
    }

    /**
     * Check if this is a constant operation.
     */
    public boolean isConstant() {
        return operation instanceof StableHloAst.ConstantOp;
    }

    /**
     * Check if this is a return operation.
     */
    public boolean isReturn() {
        return operation instanceof StableHloAst.ReturnOp;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(opName());
        sb.append(" inputs=[");
        for (int i = 0; i < inputIndices.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(inputIndices[i]);
        }
        sb.append("] outputs=[");
        for (int i = 0; i < outputIndices.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(outputIndices[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
