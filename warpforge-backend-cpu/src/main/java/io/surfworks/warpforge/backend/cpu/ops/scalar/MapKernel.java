package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MapOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.map.
 *
 * <p>Applies a computation element-wise to the inputs.
 * The computation is specified as a region containing the mapping function.
 *
 * <p>Note: Full support requires interpreting the nested computation region.
 * This implementation handles common cases where the computation can be
 * directly executed.
 */
public final class MapKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        MapOp mapOp = (MapOp) op;

        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("map requires at least 1 input");
        }

        List<Long> dimensions = mapOp.dimensions();
        List<Operation> computation = mapOp.computation();

        // For now, support identity mapping (passthrough)
        // Full implementation would interpret the computation region
        if (computation.isEmpty()) {
            // Empty computation = identity
            return List.of(inputs.get(0).copy());
        }

        // Check if computation is a simple return of input
        if (computation.size() == 1 && computation.get(0) instanceof StableHloAst.ReturnOp) {
            return List.of(inputs.get(0).copy());
        }

        // For more complex computations, we would need a full interpreter
        // This is a stub that handles the common identity case
        throw new UnsupportedOperationException(
            "Complex map computations require a full interpreter. " +
            "Current implementation only supports identity mapping.");
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.MapOp;
    }
}
