package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.convert - Convert tensor element type.
 */
public class ConvertKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Convert requires exactly 1 input");
        }

        Tensor input = inputs.getFirst();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        // For now, we work through float arrays (supports F32/F64/I32/I64)
        // More sophisticated implementations would handle each type specifically
        float[] inputData = input.toFloatArray();
        float[] outputData = inputData.clone();

        Tensor output = Tensor.fromFloatArray(outputData, outputSpec.shape());
        return List.of(output);
    }
}
