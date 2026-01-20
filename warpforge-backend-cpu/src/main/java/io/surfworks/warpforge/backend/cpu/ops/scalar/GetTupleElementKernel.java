package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetTupleElementOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.get_tuple_element.
 *
 * <p>Extracts an element from a tuple at the specified index.
 * Since tuples are represented as lists of tensors in WarpForge,
 * this selects the appropriate tensor from the input list.
 */
public final class GetTupleElementKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        GetTupleElementOp getOp = (GetTupleElementOp) op;
        int index = getOp.index();

        if (index < 0 || index >= inputs.size()) {
            throw new IllegalArgumentException(
                "Tuple index " + index + " out of bounds for tuple of size " + inputs.size());
        }

        return List.of(inputs.get(index).copy());
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.GetTupleElementOp;
    }
}
