package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.Arrays;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicPadOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.dynamic_pad.
 *
 * <p>Like pad but with runtime-determined padding values.
 * Inputs: operand, padding_value, edge_padding_low, edge_padding_high, interior_padding.
 */
public final class DynamicPadKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        if (inputs.size() != 5) {
            throw new IllegalArgumentException(
                "dynamic_pad requires 5 inputs (operand, padding_value, edge_padding_low, " +
                "edge_padding_high, interior_padding)");
        }

        Tensor operand = inputs.get(0);
        Tensor paddingValueTensor = inputs.get(1);
        Tensor edgePaddingLowTensor = inputs.get(2);
        Tensor edgePaddingHighTensor = inputs.get(3);
        Tensor interiorPaddingTensor = inputs.get(4);

        float paddingValue = paddingValueTensor.toFloatArray()[0];

        int[] inputShape = operand.shape();
        int rank = inputShape.length;

        // Read padding configurations
        float[] lowData = edgePaddingLowTensor.toFloatArray();
        float[] highData = edgePaddingHighTensor.toFloatArray();
        float[] interiorData = interiorPaddingTensor.toFloatArray();

        int[] edgePaddingLow = new int[rank];
        int[] edgePaddingHigh = new int[rank];
        int[] interiorPadding = new int[rank];

        for (int i = 0; i < rank; i++) {
            edgePaddingLow[i] = (int) lowData[i];
            edgePaddingHigh[i] = (int) highData[i];
            interiorPadding[i] = (int) interiorData[i];
        }

        // Calculate output shape
        int[] outputShape = new int[rank];
        for (int i = 0; i < rank; i++) {
            int interiorSize = inputShape[i] > 0 ? (inputShape[i] - 1) * interiorPadding[i] : 0;
            outputShape[i] = edgePaddingLow[i] + inputShape[i] + interiorSize + edgePaddingHigh[i];
        }

        // Calculate output size
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }

        // Initialize output with padding value
        float[] outputData = new float[outputSize];
        Arrays.fill(outputData, paddingValue);

        // Copy input data to correct positions
        float[] inputData = operand.toFloatArray();

        // Calculate strides
        int[] inputStrides = new int[rank];
        int[] outputStrides = new int[rank];
        if (rank > 0) {
            inputStrides[rank - 1] = 1;
            outputStrides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--) {
                inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
                outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
            }
        }

        // Map input elements to output positions
        int inputSize = (int) operand.elementCount();
        for (int inIdx = 0; inIdx < inputSize; inIdx++) {
            // Convert to input coordinates
            int[] inCoords = new int[rank];
            int remaining = inIdx;
            for (int d = 0; d < rank; d++) {
                inCoords[d] = remaining / inputStrides[d];
                remaining = remaining % inputStrides[d];
            }

            // Map to output coordinates
            int outIdx = 0;
            for (int d = 0; d < rank; d++) {
                int outCoord = edgePaddingLow[d] + inCoords[d] * (1 + interiorPadding[d]);
                outIdx += outCoord * outputStrides[d];
            }

            outputData[outIdx] = inputData[inIdx];
        }

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.DynamicPadOp;
    }
}
