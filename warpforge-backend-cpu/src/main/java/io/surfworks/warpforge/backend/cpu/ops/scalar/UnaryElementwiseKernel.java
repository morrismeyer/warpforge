package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * Base class for unary elementwise operations.
 * Applies a function to each element of the input tensor.
 */
public abstract class UnaryElementwiseKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Unary operation requires exactly 1 input, got " + inputs.size());
        }

        Tensor input = inputs.getFirst();
        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = apply(inputData[i]);
        }

        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
        return List.of(output);
    }

    /**
     * Apply the unary function to a single value.
     */
    protected abstract float apply(float x);
}
