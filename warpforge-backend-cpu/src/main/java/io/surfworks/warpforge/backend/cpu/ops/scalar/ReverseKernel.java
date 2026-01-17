package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * stablehlo.reverse - Reverse tensor along specified dimensions.
 */
public class ReverseKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Reverse requires exactly 1 input, got " + inputs.size());
        }

        StableHloAst.ReverseOp reverseOp = (StableHloAst.ReverseOp) op;
        Tensor input = inputs.getFirst();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] shape = input.shape();
        int rank = shape.length;

        Set<Integer> reverseDims = new HashSet<>();
        for (Long dim : reverseOp.dimensions()) {
            reverseDims.add(dim.intValue());
        }

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];

        long[] strides = computeStrides(shape);
        int[] idx = new int[rank];

        for (int flatIdx = 0; flatIdx < inputData.length; flatIdx++) {
            unflattenIndex(flatIdx, strides, idx);

            // Compute reversed indices
            int[] reversedIdx = new int[rank];
            for (int d = 0; d < rank; d++) {
                if (reverseDims.contains(d)) {
                    reversedIdx[d] = shape[d] - 1 - idx[d];
                } else {
                    reversedIdx[d] = idx[d];
                }
            }

            int reversedFlatIdx = flattenIndex(reversedIdx, strides);
            outputData[flatIdx] = inputData[reversedFlatIdx];
        }

        Tensor output = Tensor.fromFloatArray(outputData, shape);
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
