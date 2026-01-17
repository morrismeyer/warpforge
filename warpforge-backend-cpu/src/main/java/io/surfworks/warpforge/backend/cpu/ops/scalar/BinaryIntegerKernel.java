package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * Base class for binary elementwise operations on integers.
 */
public abstract class BinaryIntegerKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Binary operation requires exactly 2 inputs, got " + inputs.size());
        }

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);
        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());

        int[] lhsData = lhs.toIntArray();
        int[] rhsData = rhs.toIntArray();

        if (lhsData.length != rhsData.length) {
            throw new IllegalArgumentException(
                "Input tensors must have same number of elements, got " +
                lhsData.length + " and " + rhsData.length);
        }

        int[] outputData = new int[lhsData.length];
        for (int i = 0; i < lhsData.length; i++) {
            outputData[i] = apply(lhsData[i], rhsData[i]);
        }

        Tensor output = Tensor.fromIntArray(outputData, spec.shape());
        return List.of(output);
    }

    /**
     * Apply the binary function to a pair of integer values.
     */
    protected abstract int apply(int a, int b);
}
