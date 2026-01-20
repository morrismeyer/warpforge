package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicIotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.dynamic_iota.
 *
 * <p>Like iota but with runtime-determined output shape.
 * Creates a tensor with values [0, 1, 2, ...] along the iota dimension.
 */
public final class DynamicIotaKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        DynamicIotaOp iotaOp = (DynamicIotaOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "dynamic_iota requires 1 input (output_shape)");
        }

        Tensor shapeTensor = inputs.get(0);
        int iotaDimension = (int) iotaOp.iotaDimension();

        // Read output shape from the input tensor
        float[] shapeData = shapeTensor.toFloatArray();
        int[] outputShape = new int[shapeData.length];
        for (int i = 0; i < shapeData.length; i++) {
            outputShape[i] = (int) shapeData[i];
        }

        // Calculate output size and strides
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }

        int[] strides = new int[outputShape.length];
        strides[outputShape.length - 1] = 1;
        for (int i = outputShape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * outputShape[i + 1];
        }

        // Fill with iota values
        float[] outputData = new float[outputSize];
        for (int outIdx = 0; outIdx < outputSize; outIdx++) {
            // Extract coordinate along iota dimension
            int coord = (outIdx / strides[iotaDimension]) % outputShape[iotaDimension];
            outputData[outIdx] = coord;
        }

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.DynamicIotaOp;
    }
}
