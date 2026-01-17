package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.batch_norm_training - Batch normalization for training.
 * Computes normalized output, batch mean, and batch variance.
 */
public class BatchNormTrainingKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 3) {
            throw new IllegalArgumentException("BatchNormTraining requires 3 inputs, got " + inputs.size());
        }

        StableHloAst.BatchNormTrainingOp bnOp = (StableHloAst.BatchNormTrainingOp) op;
        Tensor operand = inputs.get(0);
        Tensor scale = inputs.get(1);
        Tensor offset = inputs.get(2);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] operandShape = operand.shape();
        int[] outputShape = outputSpec.shape();
        int rank = operandShape.length;

        float epsilon = bnOp.epsilon();
        int featureIndex = (int) bnOp.featureIndex();
        int numFeatures = operandShape[featureIndex];

        float[] operandData = operand.toFloatArray();
        float[] scaleData = scale.toFloatArray();
        float[] offsetData = offset.toFloatArray();

        // Compute mean and variance per feature
        float[] batchMean = new float[numFeatures];
        float[] batchVar = new float[numFeatures];
        int[] featureCounts = new int[numFeatures];

        long[] strides = computeStrides(operandShape);
        int[] idx = new int[rank];

        // First pass: compute mean
        for (int flatIdx = 0; flatIdx < operandData.length; flatIdx++) {
            unflattenIndex(flatIdx, strides, idx);
            int featureIdx = idx[featureIndex];
            batchMean[featureIdx] += operandData[flatIdx];
            featureCounts[featureIdx]++;
        }
        for (int f = 0; f < numFeatures; f++) {
            batchMean[f] /= featureCounts[f];
        }

        // Second pass: compute variance
        for (int flatIdx = 0; flatIdx < operandData.length; flatIdx++) {
            unflattenIndex(flatIdx, strides, idx);
            int featureIdx = idx[featureIndex];
            float diff = operandData[flatIdx] - batchMean[featureIdx];
            batchVar[featureIdx] += diff * diff;
        }
        for (int f = 0; f < numFeatures; f++) {
            batchVar[f] /= featureCounts[f];
        }

        // Third pass: normalize
        float[] outputData = new float[operandData.length];
        for (int flatIdx = 0; flatIdx < operandData.length; flatIdx++) {
            unflattenIndex(flatIdx, strides, idx);
            int featureIdx = idx[featureIndex];

            float x = operandData[flatIdx];
            float m = batchMean[featureIdx];
            float v = batchVar[featureIdx];
            float s = scaleData[featureIdx];
            float o = offsetData[featureIdx];

            outputData[flatIdx] = (x - m) / (float) Math.sqrt(v + epsilon) * s + o;
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        Tensor meanTensor = Tensor.fromFloatArray(batchMean, numFeatures);
        Tensor varTensor = Tensor.fromFloatArray(batchVar, numFeatures);

        return List.of(output, meanTensor, varTensor);
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
