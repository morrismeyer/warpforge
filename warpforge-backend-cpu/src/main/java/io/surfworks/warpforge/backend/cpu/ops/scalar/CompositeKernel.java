package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.ArrayList;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CompositeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.composite.
 *
 * <p>A composite operation encapsulates a subcomputation that can be
 * matched and replaced during optimization. The decomposition field
 * contains the actual computation to execute.
 *
 * <p>Note: Full support requires interpreting the decomposition region.
 * This implementation handles simple cases or returns input unchanged
 * when decomposition is empty.
 */
public final class CompositeKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        CompositeOp compositeOp = (CompositeOp) op;

        String decomposition = compositeOp.decomposition();

        if (decomposition == null || decomposition.isEmpty()) {
            // No decomposition provided - pass through inputs
            List<Tensor> outputs = new ArrayList<>(inputs.size());
            for (Tensor input : inputs) {
                outputs.add(input.copy());
            }
            return outputs;
        }

        // For now, if decomposition is present but not interpreted,
        // we throw an error rather than silently producing wrong results
        throw new UnsupportedOperationException(
            "Composite operation '" + compositeOp.name() +
            "' has decomposition that requires interpretation. " +
            "Use an interpreter that can execute nested regions.");
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.CompositeOp;
    }
}
