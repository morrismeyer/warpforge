package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.ArrayList;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OptimizationBarrierOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.optimization_barrier.
 *
 * <p>This operation is a compiler directive that prevents optimizations
 * from moving computations across the barrier. At runtime, it simply
 * passes through the inputs unchanged.
 */
public final class OptimizationBarrierKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        // Optimization barrier is a passthrough at runtime
        List<Tensor> outputs = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            outputs.add(input.copy());
        }
        return outputs;
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.OptimizationBarrierOp;
    }
}
