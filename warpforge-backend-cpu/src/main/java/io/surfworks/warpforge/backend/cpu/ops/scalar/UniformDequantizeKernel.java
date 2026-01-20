package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformDequantizeOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.uniform_dequantize.
 *
 * <p>Converts quantized integer values back to floating-point representation
 * using uniform dequantization: x = (q - zero_point) * scale.
 */
public final class UniformDequantizeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        UniformDequantizeOp dequantOp = (UniformDequantizeOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("uniform_dequantize requires exactly 1 input");
        }

        Tensor input = inputs.get(0);
        float[] inputData = input.toFloatArray();

        // Default int8 quantization parameters
        // In a full implementation, these would be extracted from the quantized type
        float scale = 1.0f / 127.0f;  // Scale for int8 to [-1, 1] range
        int zeroPoint = 0;

        float[] outputData = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = (inputData[i] - zeroPoint) * scale;
        }

        Tensor output = Tensor.zeros(input.shape());
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.UniformDequantizeOp;
    }
}
