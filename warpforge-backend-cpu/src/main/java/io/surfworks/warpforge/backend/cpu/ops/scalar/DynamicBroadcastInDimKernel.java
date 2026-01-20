package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicBroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.dynamic_broadcast_in_dim.
 *
 * <p>Like broadcast_in_dim but with runtime-determined output shape.
 * The second input tensor contains the output dimensions.
 */
public final class DynamicBroadcastInDimKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        DynamicBroadcastInDimOp broadcastOp = (DynamicBroadcastInDimOp) op;

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "dynamic_broadcast_in_dim requires 2 inputs (operand, output_dimensions)");
        }

        Tensor operand = inputs.get(0);
        Tensor outputDims = inputs.get(1);
        List<Long> broadcastDims = broadcastOp.broadcastDimensions();

        // Read output shape from the second tensor
        float[] dimData = outputDims.toFloatArray();
        int[] outputShape = new int[dimData.length];
        for (int i = 0; i < dimData.length; i++) {
            outputShape[i] = (int) dimData[i];
        }

        // Calculate output size
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }

        float[] inputData = operand.toFloatArray();
        int[] inputShape = operand.shape();
        float[] outputData = new float[outputSize];

        // Compute strides for output
        int[] outputStrides = new int[outputShape.length];
        outputStrides[outputShape.length - 1] = 1;
        for (int i = outputShape.length - 2; i >= 0; i--) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
        }

        // Compute strides for input
        int[] inputStrides = new int[inputShape.length];
        if (inputShape.length > 0) {
            inputStrides[inputShape.length - 1] = 1;
            for (int i = inputShape.length - 2; i >= 0; i--) {
                inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1];
            }
        }

        // Fill output with broadcasted values
        for (int outIdx = 0; outIdx < outputSize; outIdx++) {
            // Convert flat index to multi-dimensional
            int[] outCoords = new int[outputShape.length];
            int remaining = outIdx;
            for (int d = 0; d < outputShape.length; d++) {
                outCoords[d] = remaining / outputStrides[d];
                remaining = remaining % outputStrides[d];
            }

            // Map to input coordinates using broadcast dimensions
            int inIdx = 0;
            for (int i = 0; i < broadcastDims.size(); i++) {
                int outDim = broadcastDims.get(i).intValue();
                int coord = outCoords[outDim];
                // Handle broadcasting (dimension size 1 maps to 0)
                if (inputShape[i] == 1) {
                    coord = 0;
                }
                inIdx += coord * inputStrides[i];
            }

            outputData[outIdx] = inputData[inIdx];
        }

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.DynamicBroadcastInDimOp;
    }
}
