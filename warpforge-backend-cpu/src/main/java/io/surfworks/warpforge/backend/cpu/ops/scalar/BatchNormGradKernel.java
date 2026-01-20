package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormGradOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.batch_norm_grad.
 *
 * <p>Computes gradients of batch normalization for backpropagation.
 * Inputs: operand, scale, mean, variance, grad_output.
 * Outputs: grad_operand, grad_scale, grad_offset.
 */
public final class BatchNormGradKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        BatchNormGradOp gradOp = (BatchNormGradOp) op;

        if (inputs.size() != 5) {
            throw new IllegalArgumentException(
                "batch_norm_grad requires 5 inputs (operand, scale, mean, variance, grad_output)");
        }

        Tensor operand = inputs.get(0);
        Tensor scale = inputs.get(1);
        Tensor mean = inputs.get(2);
        Tensor variance = inputs.get(3);
        Tensor gradOutput = inputs.get(4);

        float epsilon = gradOp.epsilon();
        int featureIndex = (int) gradOp.featureIndex();

        int[] shape = operand.shape();
        int featureSize = shape[featureIndex];

        float[] operandData = operand.toFloatArray();
        float[] scaleData = scale.toFloatArray();
        float[] meanData = mean.toFloatArray();
        float[] varianceData = variance.toFloatArray();
        float[] gradOutputData = gradOutput.toFloatArray();

        // Calculate batch size (product of all dims except feature)
        int batchSize = 1;
        for (int i = 0; i < shape.length; i++) {
            if (i != featureIndex) {
                batchSize *= shape[i];
            }
        }

        float[] gradOperandData = new float[operandData.length];
        float[] gradScaleData = new float[featureSize];
        float[] gradOffsetData = new float[featureSize];

        // Calculate strides
        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Compute gradients for each feature
        for (int f = 0; f < featureSize; f++) {
            float m = meanData[f];
            float v = varianceData[f];
            float s = scaleData[f];
            float invStd = (float) (1.0 / Math.sqrt(v + epsilon));

            // Accumulate gradients for scale and offset
            float dScale = 0;
            float dOffset = 0;
            float dMeanPart = 0;
            float dVarPart = 0;

            // First pass: compute dScale, dOffset, and intermediate values
            for (int i = 0; i < operandData.length; i++) {
                int featureCoord = (i / strides[featureIndex]) % featureSize;
                if (featureCoord == f) {
                    float x = operandData[i];
                    float xNorm = (x - m) * invStd;
                    float dy = gradOutputData[i];

                    dScale += dy * xNorm;
                    dOffset += dy;
                    dMeanPart += dy;
                    dVarPart += dy * (x - m);
                }
            }

            gradScaleData[f] = dScale;
            gradOffsetData[f] = dOffset;

            // Second pass: compute gradient w.r.t. operand
            float invN = 1.0f / batchSize;
            float dxNormSum = dMeanPart * s * invStd;
            float dxVarSum = dVarPart * s * (-0.5f) * invStd * invStd * invStd;

            for (int i = 0; i < operandData.length; i++) {
                int featureCoord = (i / strides[featureIndex]) % featureSize;
                if (featureCoord == f) {
                    float x = operandData[i];
                    float dy = gradOutputData[i];
                    float dxNorm = dy * s * invStd;
                    float dxCenter = dxNorm - dxNormSum * invN + 2.0f * (x - m) * dxVarSum * invN;
                    gradOperandData[i] = dxCenter;
                }
            }
        }

        Tensor gradOperand = Tensor.zeros(shape);
        gradOperand.copyFrom(gradOperandData);

        Tensor gradScale = Tensor.zeros(new int[]{featureSize});
        gradScale.copyFrom(gradScaleData);

        Tensor gradOffset = Tensor.zeros(new int[]{featureSize});
        gradOffset.copyFrom(gradOffsetData);

        return List.of(gradOperand, gradScale, gradOffset);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.BatchNormGradOp;
    }
}
