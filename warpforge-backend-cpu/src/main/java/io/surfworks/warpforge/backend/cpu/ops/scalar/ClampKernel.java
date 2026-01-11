package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.clamp - Element-wise clamping.
 * clamp(min, operand, max) returns max(min, min(operand, max)).
 */
public class ClampKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 3) {
            throw new IllegalArgumentException("Clamp requires exactly 3 inputs (min, operand, max)");
        }

        Tensor min = inputs.get(0);
        Tensor operand = inputs.get(1);
        Tensor max = inputs.get(2);

        float[] minData = min.toFloatArray();
        float[] operandData = operand.toFloatArray();
        float[] maxData = max.toFloatArray();
        float[] outputData = new float[operandData.length];

        for (int i = 0; i < operandData.length; i++) {
            float minVal = minData.length == 1 ? minData[0] : minData[i];
            float maxVal = maxData.length == 1 ? maxData[0] : maxData[i];
            outputData[i] = Math.max(minVal, Math.min(operandData[i], maxVal));
        }

        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());
        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
        return List.of(output);
    }
}
