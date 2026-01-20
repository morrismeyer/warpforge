package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.dynamic_reshape.
 *
 * <p>Like reshape but with runtime-determined output shape.
 * The second input tensor contains the output dimensions.
 */
public final class DynamicReshapeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "dynamic_reshape requires 2 inputs (operand, output_shape)");
        }

        Tensor operand = inputs.get(0);
        Tensor shapeTensor = inputs.get(1);

        // Read output shape from the second tensor
        float[] shapeData = shapeTensor.toFloatArray();
        int[] outputShape = new int[shapeData.length];
        int outputSize = 1;
        for (int i = 0; i < shapeData.length; i++) {
            outputShape[i] = (int) shapeData[i];
            outputSize *= outputShape[i];
        }

        // Verify element counts match
        if (outputSize != operand.elementCount()) {
            throw new IllegalArgumentException(
                "Cannot reshape tensor of " + operand.elementCount() +
                " elements to shape with " + outputSize + " elements");
        }

        // Copy data to new shape
        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(operand.toFloatArray());
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.DynamicReshapeOp;
    }
}
