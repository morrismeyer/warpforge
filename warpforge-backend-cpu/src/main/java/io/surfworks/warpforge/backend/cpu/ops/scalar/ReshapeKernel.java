package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.reshape - Reshape tensor to new shape.
 * Data is preserved in row-major order.
 */
public class ReshapeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Reshape requires exactly 1 input");
        }

        Tensor input = inputs.getFirst();
        TensorSpec newSpec = TensorSpec.fromAst(op.tensorResultType());

        // Verify element count matches
        if (input.spec().elementCount() != newSpec.elementCount()) {
            throw new IllegalArgumentException(
                "Cannot reshape tensor of " + input.spec().elementCount() +
                " elements to shape with " + newSpec.elementCount() + " elements");
        }

        // Simply copy data to new shape (row-major to row-major is just a view change)
        float[] data = input.toFloatArray();
        Tensor output = Tensor.fromFloatArray(data, newSpec.shape());
        return List.of(output);
    }
}
