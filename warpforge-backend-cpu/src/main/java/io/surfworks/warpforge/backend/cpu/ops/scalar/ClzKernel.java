package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.count_leading_zeros - Element-wise count of leading zero bits.
 * Operates on integer types.
 */
public class ClzKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Clz requires exactly 1 input, got " + inputs.size());
        }

        Tensor input = inputs.getFirst();
        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputData = input.toIntArray();
        int[] outputData = new int[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = Integer.numberOfLeadingZeros(inputData[i]);
        }

        Tensor output = Tensor.fromIntArray(outputData, spec.shape());
        return List.of(output);
    }
}
