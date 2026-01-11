package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.select - Element-wise selection based on predicate.
 * select(pred, on_true, on_false) returns on_true where pred is true, on_false otherwise.
 */
public class SelectKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 3) {
            throw new IllegalArgumentException("Select requires exactly 3 inputs (pred, on_true, on_false)");
        }

        Tensor pred = inputs.get(0);
        Tensor onTrue = inputs.get(1);
        Tensor onFalse = inputs.get(2);

        float[] predData = pred.toFloatArray();
        float[] trueData = onTrue.toFloatArray();
        float[] falseData = onFalse.toFloatArray();
        float[] outputData = new float[predData.length];

        for (int i = 0; i < predData.length; i++) {
            // pred is boolean: 1.0 = true, 0.0 = false
            outputData[i] = predData[i] != 0.0f ? trueData[i] : falseData[i];
        }

        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());
        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
        return List.of(output);
    }
}
