package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.Arrays;
import java.util.List;

/**
 * stablehlo.dynamic_update_slice - Update slice at dynamic indices.
 */
public class DynamicUpdateSliceKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() < 2) {
            throw new IllegalArgumentException("DynamicUpdateSlice requires at least 2 inputs");
        }

        Tensor operand = inputs.get(0);
        Tensor update = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] operandShape = operand.shape();
        int[] updateShape = update.shape();
        int[] outputShape = outputSpec.shape();
        int rank = operandShape.length;

        // Get dynamic start indices from inputs (inputs 2 to N are scalar indices)
        int[] startIndices = new int[rank];
        for (int d = 0; d < rank; d++) {
            if (d + 2 < inputs.size()) {
                startIndices[d] = inputs.get(d + 2).toIntArray()[0];
            }
        }

        // Clamp start indices to valid range
        for (int d = 0; d < rank; d++) {
            int maxStart = operandShape[d] - updateShape[d];
            startIndices[d] = Math.max(0, Math.min(startIndices[d], maxStart));
        }

        float[] operandData = operand.toFloatArray();
        float[] updateData = update.toFloatArray();

        // Start with a copy of the operand
        float[] outputData = Arrays.copyOf(operandData, operandData.length);

        long[] operandStrides = computeStrides(operandShape);
        long[] updateStrides = computeStrides(updateShape);

        int[] updateIdx = new int[rank];
        for (int updateFlatIdx = 0; updateFlatIdx < updateData.length; updateFlatIdx++) {
            unflattenIndex(updateFlatIdx, updateStrides, updateIdx);

            // Map update index to operand index
            int[] operandIdx = new int[rank];
            for (int d = 0; d < rank; d++) {
                operandIdx[d] = startIndices[d] + updateIdx[d];
            }

            int operandFlatIdx = flattenIndex(operandIdx, operandStrides);
            outputData[operandFlatIdx] = updateData[updateFlatIdx];
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        return List.of(output);
    }

    private long[] computeStrides(int[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private void unflattenIndex(long flatIdx, long[] strides, int[] result) {
        for (int i = 0; i < strides.length; i++) {
            result[i] = (int) (flatIdx / strides[i]);
            flatIdx %= strides[i];
        }
    }

    private int flattenIndex(int[] indices, long[] strides) {
        long result = 0;
        for (int i = 0; i < indices.length; i++) {
            result += indices[i] * strides[i];
        }
        return (int) result;
    }
}
