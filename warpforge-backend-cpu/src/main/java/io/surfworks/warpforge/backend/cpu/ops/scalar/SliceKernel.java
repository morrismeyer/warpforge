package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.slice - Extract a slice from a tensor.
 */
public class SliceKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Slice requires exactly 1 input, got " + inputs.size());
        }

        StableHloAst.SliceOp sliceOp = (StableHloAst.SliceOp) op;
        Tensor input = inputs.getFirst();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputShape = input.shape();
        int[] outputShape = outputSpec.shape();
        int rank = inputShape.length;

        List<Long> startIndices = sliceOp.startIndices();
        List<Long> limitIndices = sliceOp.limitIndices();
        List<Long> strides = sliceOp.strides();

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[(int) outputSpec.elementCount()];

        long[] inputStrides = computeStrides(inputShape);
        long[] outputStrides = computeStrides(outputShape);

        int[] outputIdx = new int[rank];
        for (int outFlatIdx = 0; outFlatIdx < outputData.length; outFlatIdx++) {
            unflattenIndex(outFlatIdx, outputStrides, outputIdx);

            // Map output index to input index
            int[] inputIdx = new int[rank];
            for (int d = 0; d < rank; d++) {
                inputIdx[d] = (int) (startIndices.get(d) + outputIdx[d] * strides.get(d));
            }

            int inFlatIdx = flattenIndex(inputIdx, inputStrides);
            outputData[outFlatIdx] = inputData[inFlatIdx];
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
