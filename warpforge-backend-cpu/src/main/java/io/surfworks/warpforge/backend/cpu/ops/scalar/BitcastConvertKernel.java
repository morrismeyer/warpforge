package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.bitcast_convert - Bitcast to different element type.
 * Reinterprets the raw bytes as a different type without conversion.
 */
public class BitcastConvertKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("BitcastConvert requires exactly 1 input, got " + inputs.size());
        }

        Tensor input = inputs.getFirst();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        // For now, just copy the raw data
        // A proper implementation would reinterpret bits based on source/target types
        float[] inputData = input.toFloatArray();
        float[] outputData = inputData.clone();

        Tensor output = Tensor.fromFloatArray(outputData, outputSpec.shape());
        return List.of(output);
    }
}
