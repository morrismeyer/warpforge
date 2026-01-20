package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformQuantizeOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.uniform_quantize.
 *
 * <p>Converts floating-point values to quantized integer representation
 * using uniform quantization: q = clamp(round(x / scale) + zero_point, qmin, qmax).
 */
public final class UniformQuantizeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        UniformQuantizeOp quantOp = (UniformQuantizeOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("uniform_quantize requires exactly 1 input");
        }

        Tensor input = inputs.get(0);
        float[] inputData = input.toFloatArray();

        // Default int8 quantization parameters
        // In a full implementation, these would be extracted from the quantized type
        float scale = 1.0f / 127.0f;  // Scale for [-1, 1] range to int8
        int zeroPoint = 0;
        int qmin = -128;
        int qmax = 127;

        float[] outputData = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            float quantized = Math.round(inputData[i] / scale) + zeroPoint;
            quantized = Math.max(qmin, Math.min(qmax, quantized));
            outputData[i] = quantized;
        }

        Tensor output = Tensor.zeros(input.shape());
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.UniformQuantizeOp;
    }
}
