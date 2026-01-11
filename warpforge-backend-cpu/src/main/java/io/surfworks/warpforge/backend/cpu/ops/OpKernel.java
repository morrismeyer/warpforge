package io.surfworks.warpforge.backend.cpu.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;

/**
 * Interface for StableHLO operation kernels.
 * Each kernel implements execution logic for one or more operations.
 */
@FunctionalInterface
public interface OpKernel {

    /**
     * Execute the operation.
     *
     * @param op     The operation to execute
     * @param inputs Input tensors
     * @return Output tensors
     */
    List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs);

    /**
     * Check if this kernel supports the given operation.
     * Default implementation returns true (kernel handles validation).
     *
     * @param op The operation to check
     * @return true if supported
     */
    default boolean supports(StableHloAst.Operation op) {
        return true;
    }
}
