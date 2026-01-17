package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.batch_norm_inference - Batch normalization for inference.
 * output = (operand - mean) / sqrt(variance + epsilon) * scale + offset
 */
public class BatchNormInferenceKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 5) {
            throw new IllegalArgumentException("BatchNormInference requires 5 inputs, got " + inputs.size());
        }

        StableHloAst.BatchNormInferenceOp bnOp = (StableHloAst.BatchNormInferenceOp) op;
        Tensor operand = inputs.get(0);
        Tensor scale = inputs.get(1);
        Tensor offset = inputs.get(2);
        Tensor mean = inputs.get(3);
        Tensor variance = inputs.get(4);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] operandShape = operand.shape();
        int[] outputShape = outputSpec.shape();
        int rank = operandShape.length;

        float epsilon = bnOp.epsilon();
        int featureIndex = (int) bnOp.featureIndex();

        float[] operandData = operand.toFloatArray();
        float[] scaleData = scale.toFloatArray();
        float[] offsetData = offset.toFloatArray();
        float[] meanData = mean.toFloatArray();
        float[] varianceData = variance.toFloatArray();

        float[] outputData = new float[operandData.length];

        long[] strides = computeStrides(operandShape);
        int[] idx = new int[rank];

        for (int flatIdx = 0; flatIdx < operandData.length; flatIdx++) {
            unflattenIndex(flatIdx, strides, idx);
            int featureIdx = idx[featureIndex];

            float x = operandData[flatIdx];
            float m = meanData[featureIdx];
            float v = varianceData[featureIdx];
            float s = scaleData[featureIdx];
            float o = offsetData[featureIdx];

            // Normalize and scale
            outputData[flatIdx] = (x - m) / (float) Math.sqrt(v + epsilon) * s + o;
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
