package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.iota - Generate indices along a dimension.
 */
public class IotaKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        StableHloAst.IotaOp iotaOp = (StableHloAst.IotaOp) op;
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] outputShape = outputSpec.shape();
        int rank = outputShape.length;
        int iotaDim = (int) iotaOp.iotaDimension();

        float[] outputData = new float[(int) outputSpec.elementCount()];
        long[] strides = computeStrides(outputShape);

        int[] idx = new int[rank];
        for (int flatIdx = 0; flatIdx < outputData.length; flatIdx++) {
            unflattenIndex(flatIdx, strides, idx);
            outputData[flatIdx] = idx[iotaDim];
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
}
