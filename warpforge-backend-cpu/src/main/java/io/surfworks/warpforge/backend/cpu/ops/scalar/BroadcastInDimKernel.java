package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.broadcast_in_dim - Broadcast tensor to a larger shape.
 * The broadcastDimensions specify which output dimensions correspond to input dimensions.
 */
public class BroadcastInDimKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.BroadcastInDimOp broadcastOp)) {
            throw new IllegalArgumentException("Expected BroadcastInDimOp");
        }
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("BroadcastInDim requires exactly 1 input");
        }

        Tensor input = inputs.getFirst();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());
        int[] inputShape = input.shape();
        int[] outputShape = outputSpec.shape();
        List<Long> broadcastDims = broadcastOp.broadcastDimensions();

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[(int) outputSpec.elementCount()];

        // Compute input strides
        long[] inputStrides = computeStrides(inputShape);

        // Iterate over output and map back to input
        int[] outputIndices = new int[outputShape.length];
        for (int flatOut = 0; flatOut < outputData.length; flatOut++) {
            // Convert flat index to multi-dimensional output indices
            int remaining = flatOut;
            for (int d = outputShape.length - 1; d >= 0; d--) {
                outputIndices[d] = remaining % outputShape[d];
                remaining /= outputShape[d];
            }

            // Map output indices to input indices using broadcastDimensions
            long flatIn = 0;
            for (int i = 0; i < broadcastDims.size(); i++) {
                int outDim = broadcastDims.get(i).intValue();
                int inputIdx = outputIndices[outDim];
                // Handle broadcasting: if input dim is 1, always use index 0
                if (inputShape[i] == 1) {
                    inputIdx = 0;
                }
                flatIn += inputIdx * inputStrides[i];
            }

            outputData[flatOut] = inputData[(int) flatIn];
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
}
