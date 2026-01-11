package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.transpose - Permute tensor dimensions.
 */
public class TransposeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.TransposeOp transposeOp)) {
            throw new IllegalArgumentException("Expected TransposeOp");
        }
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Transpose requires exactly 1 input");
        }

        Tensor input = inputs.getFirst();
        int[] inputShape = input.shape();
        List<Long> perm = transposeOp.permutation();

        // Compute output shape
        int[] outputShape = new int[perm.size()];
        for (int i = 0; i < perm.size(); i++) {
            outputShape[i] = inputShape[perm.get(i).intValue()];
        }

        // Compute strides for input and output
        long[] inputStrides = computeStrides(inputShape);

        // Transpose by iterating over output and looking up input
        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];

        int[] outputIndices = new int[outputShape.length];
        for (int flatOut = 0; flatOut < outputData.length; flatOut++) {
            // Convert flat index to multi-dimensional output indices
            int remaining = flatOut;
            for (int d = outputShape.length - 1; d >= 0; d--) {
                outputIndices[d] = remaining % outputShape[d];
                remaining /= outputShape[d];
            }

            // Map output indices to input indices using permutation
            long flatIn = 0;
            for (int d = 0; d < perm.size(); d++) {
                flatIn += outputIndices[d] * inputStrides[perm.get(d).intValue()];
            }

            outputData[flatOut] = inputData[(int) flatIn];
        }

        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());
        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
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
}
