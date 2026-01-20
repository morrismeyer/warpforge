package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.ArrayList;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TupleOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.tuple.
 *
 * <p>Creates a tuple from multiple input tensors. Since WarpForge uses
 * a flat tensor list model, this kernel simply passes through the inputs
 * as a list of outputs representing the tuple elements.
 */
public final class TupleKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        TupleOp tupleOp = (TupleOp) op;
        // Return all inputs as the tuple elements
        // The tuple is represented as a list of tensors
        List<Tensor> outputs = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            outputs.add(input.copy());
        }
        return outputs;
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.TupleOp;
    }
}
