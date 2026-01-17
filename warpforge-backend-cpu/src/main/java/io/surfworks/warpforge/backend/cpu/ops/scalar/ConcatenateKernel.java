package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.concatenate - Concatenate tensors along a dimension.
 */
public class ConcatenateKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Concatenate requires at least 1 input");
        }

        StableHloAst.ConcatenateOp concatOp = (StableHloAst.ConcatenateOp) op;
        int dimension = (int) concatOp.dimension();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] outputShape = outputSpec.shape();
        int rank = outputShape.length;

        // Compute strides for output
        long[] outputStrides = computeStrides(outputShape);

        float[] outputData = new float[(int) outputSpec.elementCount()];

        int offsetAlongDim = 0;
        for (Tensor input : inputs) {
            int[] inputShape = input.shape();
            float[] inputData = input.toFloatArray();
            long[] inputStrides = computeStrides(inputShape);

            // Copy elements
            int[] inputIdx = new int[rank];
            for (int flatIdx = 0; flatIdx < inputData.length; flatIdx++) {
                // Convert flat index to multi-dimensional indices
                unflattenIndex(flatIdx, inputStrides, inputIdx);

                // Map to output indices (add offset along concatenation dimension)
                int[] outputIdx = inputIdx.clone();
                outputIdx[dimension] += offsetAlongDim;

                // Convert to output flat index
                int outputFlatIdx = flattenIndex(outputIdx, outputStrides);
                outputData[outputFlatIdx] = inputData[flatIdx];
            }

            offsetAlongDim += inputShape[dimension];
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
