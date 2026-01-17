package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.dynamic_slice - Extract slice with dynamic start indices.
 */
public class DynamicSliceKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() < 1) {
            throw new IllegalArgumentException("DynamicSlice requires at least 1 input");
        }

        StableHloAst.DynamicSliceOp sliceOp = (StableHloAst.DynamicSliceOp) op;
        Tensor operand = inputs.get(0);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputShape = operand.shape();
        int[] outputShape = outputSpec.shape();
        int rank = inputShape.length;

        // Get dynamic start indices from inputs (inputs 1 to N are scalar indices)
        int[] startIndices = new int[rank];
        for (int d = 0; d < rank; d++) {
            if (d + 1 < inputs.size()) {
                startIndices[d] = inputs.get(d + 1).toIntArray()[0];
            }
        }

        List<Long> sliceSizes = sliceOp.sliceSizes();

        // Clamp start indices to valid range
        for (int d = 0; d < rank; d++) {
            int maxStart = inputShape[d] - sliceSizes.get(d).intValue();
            startIndices[d] = Math.max(0, Math.min(startIndices[d], maxStart));
        }

        float[] inputData = operand.toFloatArray();
        float[] outputData = new float[(int) outputSpec.elementCount()];

        long[] inputStrides = computeStrides(inputShape);
        long[] outputStrides = computeStrides(outputShape);

        int[] outputIdx = new int[rank];
        for (int outFlatIdx = 0; outFlatIdx < outputData.length; outFlatIdx++) {
            unflattenIndex(outFlatIdx, outputStrides, outputIdx);

            // Map output index to input index
            int[] inputIdx = new int[rank];
            for (int d = 0; d < rank; d++) {
                inputIdx[d] = startIndices[d] + outputIdx[d];
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
