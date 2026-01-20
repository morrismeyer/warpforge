package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicConvOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.dynamic_conv.
 *
 * <p>Like convolution but with runtime-determined padding values.
 * Inputs: lhs, rhs, padding (tensor containing padding values).
 */
public final class DynamicConvKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        DynamicConvOp convOp = (DynamicConvOp) op;

        if (inputs.size() < 3) {
            throw new IllegalArgumentException(
                "dynamic_conv requires at least 3 inputs (lhs, rhs, padding)");
        }

        Tensor lhs = inputs.get(0);  // Input
        Tensor rhs = inputs.get(1);  // Kernel/filter
        Tensor paddingTensor = inputs.get(2);  // Dynamic padding

        // Read padding from tensor
        float[] paddingData = paddingTensor.toFloatArray();

        // Extract dimensions
        int[] lhsShape = lhs.shape();  // [N, H, W, C_in] or [N, C_in, H, W]
        int[] rhsShape = rhs.shape();  // [K_h, K_w, C_in, C_out] or [C_out, C_in, K_h, K_w]

        // For simplicity, assume NHWC format
        // Full implementation would handle arbitrary layouts via dimension_numbers

        int batch = lhsShape[0];
        int inputHeight = lhsShape[1];
        int inputWidth = lhsShape[2];
        int inputChannels = lhsShape[3];

        int kernelHeight = rhsShape[0];
        int kernelWidth = rhsShape[1];
        int outputChannels = rhsShape[3];

        // Parse dynamic padding: assume format is [pad_top, pad_bottom, pad_left, pad_right]
        int padTop = paddingData.length > 0 ? (int) paddingData[0] : 0;
        int padBottom = paddingData.length > 1 ? (int) paddingData[1] : 0;
        int padLeft = paddingData.length > 2 ? (int) paddingData[2] : 0;
        int padRight = paddingData.length > 3 ? (int) paddingData[3] : 0;

        // Get strides (default to 1)
        List<Long> windowStrides = convOp.windowStrides();
        int strideH = windowStrides.size() > 0 ? windowStrides.get(0).intValue() : 1;
        int strideW = windowStrides.size() > 1 ? windowStrides.get(1).intValue() : 1;

        // Calculate output dimensions
        int paddedHeight = inputHeight + padTop + padBottom;
        int paddedWidth = inputWidth + padLeft + padRight;
        int outputHeight = (paddedHeight - kernelHeight) / strideH + 1;
        int outputWidth = (paddedWidth - kernelWidth) / strideW + 1;

        // Allocate output
        int[] outputShape = new int[]{batch, outputHeight, outputWidth, outputChannels};
        float[] outputData = new float[batch * outputHeight * outputWidth * outputChannels];

        float[] lhsData = lhs.toFloatArray();
        float[] rhsData = rhs.toFloatArray();

        // Perform convolution
        for (int n = 0; n < batch; n++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    for (int oc = 0; oc < outputChannels; oc++) {
                        float sum = 0;

                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                for (int ic = 0; ic < inputChannels; ic++) {
                                    int ih = oh * strideH + kh - padTop;
                                    int iw = ow * strideW + kw - padLeft;

                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        int lhsIdx = n * inputHeight * inputWidth * inputChannels +
                                                     ih * inputWidth * inputChannels +
                                                     iw * inputChannels +
                                                     ic;
                                        int rhsIdx = kh * kernelWidth * inputChannels * outputChannels +
                                                     kw * inputChannels * outputChannels +
                                                     ic * outputChannels +
                                                     oc;
                                        sum += lhsData[lhsIdx] * rhsData[rhsIdx];
                                    }
                                }
                            }
                        }

                        int outIdx = n * outputHeight * outputWidth * outputChannels +
                                     oh * outputWidth * outputChannels +
                                     ow * outputChannels +
                                     oc;
                        outputData[outIdx] = sum;
                    }
                }
            }
        }

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.DynamicConvOp;
    }
}
