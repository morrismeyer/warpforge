package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.Arrays;
import java.util.List;

/**
 * stablehlo.pad - Pad tensor with a constant value.
 */
public class PadKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Pad requires exactly 2 inputs, got " + inputs.size());
        }

        StableHloAst.PadOp padOp = (StableHloAst.PadOp) op;
        Tensor input = inputs.get(0);
        Tensor paddingValue = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputShape = input.shape();
        int[] outputShape = outputSpec.shape();
        int rank = inputShape.length;

        List<Long> edgePaddingLow = padOp.edgePaddingLow();
        List<Long> edgePaddingHigh = padOp.edgePaddingHigh();
        List<Long> interiorPadding = padOp.interiorPadding();

        float[] inputData = input.toFloatArray();
        float padValue = paddingValue.toFloatArray()[0];

        float[] outputData = new float[(int) outputSpec.elementCount()];
        Arrays.fill(outputData, padValue);

        long[] inputStrides = computeStrides(inputShape);
        long[] outputStrides = computeStrides(outputShape);

        int[] inputIdx = new int[rank];
        for (int inFlatIdx = 0; inFlatIdx < inputData.length; inFlatIdx++) {
            unflattenIndex(inFlatIdx, inputStrides, inputIdx);

            // Map input index to output index
            int[] outputIdx = new int[rank];
            for (int d = 0; d < rank; d++) {
                outputIdx[d] = (int) (edgePaddingLow.get(d) + inputIdx[d] * (1 + interiorPadding.get(d)));
            }

            int outFlatIdx = flattenIndex(outputIdx, outputStrides);
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
