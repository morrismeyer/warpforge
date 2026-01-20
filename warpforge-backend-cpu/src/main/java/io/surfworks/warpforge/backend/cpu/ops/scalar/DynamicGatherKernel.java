package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicGatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.dynamic_gather.
 *
 * <p>Like gather but with runtime-determined slice sizes.
 * Inputs: operand, start_indices, slice_sizes.
 */
public final class DynamicGatherKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        DynamicGatherOp gatherOp = (DynamicGatherOp) op;

        if (inputs.size() != 3) {
            throw new IllegalArgumentException(
                "dynamic_gather requires 3 inputs (operand, start_indices, slice_sizes)");
        }

        Tensor operand = inputs.get(0);
        Tensor startIndices = inputs.get(1);
        Tensor sliceSizesTensor = inputs.get(2);

        // Read slice sizes from the third tensor
        float[] sliceSizesData = sliceSizesTensor.toFloatArray();
        int[] sliceSizes = new int[sliceSizesData.length];
        for (int i = 0; i < sliceSizesData.length; i++) {
            sliceSizes[i] = (int) sliceSizesData[i];
        }

        // Delegate to static gather implementation with dynamic slice sizes
        return executeGather(operand, startIndices, sliceSizes, gatherOp);
    }

    private List<Tensor> executeGather(Tensor operand, Tensor startIndices,
                                        int[] sliceSizes, DynamicGatherOp gatherOp) {
        int[] operandShape = operand.shape();
        int[] indicesShape = startIndices.shape();
        float[] operandData = operand.toFloatArray();
        float[] indicesData = startIndices.toFloatArray();

        // Simplified gather: assume indices are 1D pointing into first dimension
        int numGathers = indicesShape.length > 0 ? indicesShape[0] : 1;
        if (indicesShape.length == 0) {
            numGathers = 1;
        }

        // Calculate output shape
        int[] outputShape = new int[sliceSizes.length];
        outputShape[0] = numGathers;
        System.arraycopy(sliceSizes, 1, outputShape, 1, sliceSizes.length - 1);

        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }

        // Calculate strides
        int[] operandStrides = new int[operandShape.length];
        operandStrides[operandShape.length - 1] = 1;
        for (int i = operandShape.length - 2; i >= 0; i--) {
            operandStrides[i] = operandStrides[i + 1] * operandShape[i + 1];
        }

        float[] outputData = new float[outputSize];

        // Simple case: 1D indices gathering slices
        int sliceSize = outputSize / numGathers;
        for (int i = 0; i < numGathers; i++) {
            int startIdx = (int) indicesData[i];
            for (int j = 0; j < sliceSize; j++) {
                int srcIdx = startIdx * operandStrides[0] + j;
                int dstIdx = i * sliceSize + j;
                if (srcIdx < operandData.length && dstIdx < outputData.length) {
                    outputData[dstIdx] = operandData[srcIdx];
                }
            }
        }

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.DynamicGatherOp;
    }
}
