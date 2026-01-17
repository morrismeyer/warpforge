package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.get_dimension_size - Get the size of a dimension as a scalar tensor.
 */
public class GetDimensionSizeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("GetDimensionSize requires exactly 1 input, got " + inputs.size());
        }

        StableHloAst.GetDimensionSizeOp getDimOp = (StableHloAst.GetDimensionSizeOp) op;
        Tensor input = inputs.getFirst();

        int dimension = (int) getDimOp.dimension();
        int dimSize = input.shape()[dimension];

        // Return as a scalar int32 tensor
        Tensor output = Tensor.fromIntArray(new int[]{dimSize});
        return List.of(output);
    }
}
