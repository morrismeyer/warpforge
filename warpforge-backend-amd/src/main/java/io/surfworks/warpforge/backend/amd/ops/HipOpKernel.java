package io.surfworks.warpforge.backend.amd.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;

/**
 * Interface for HIP kernel implementations.
 */
@FunctionalInterface
public interface HipOpKernel {

    /**
     * Execute the operation on HIP device.
     *
     * @param op     The StableHLO operation to execute
     * @param inputs Input tensors (device memory)
     * @return Output tensors (device memory)
     */
    List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs);

    /**
     * Check if this kernel supports the given operation.
     */
    default boolean supports(StableHloAst.Operation op) {
        return true;
    }

    /**
     * Get the HIP stream to use for this kernel.
     * Default is the null stream (synchronous).
     */
    default long getStream() {
        return 0;
    }
}
