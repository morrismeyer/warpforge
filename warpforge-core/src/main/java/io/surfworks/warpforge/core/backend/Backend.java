package io.surfworks.warpforge.core.backend;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * Backend interface for executing StableHLO operations.
 * Implementations provide execution on different hardware targets (CPU, GPU, etc.).
 */
public interface Backend extends AutoCloseable {

    /**
     * Returns the name of this backend (e.g., "cpu", "nvidia", "amd").
     */
    String name();

    /**
     * Returns the capabilities of this backend.
     */
    BackendCapabilities capabilities();

    /**
     * Execute a single StableHLO operation.
     *
     * @param op     The operation to execute
     * @param inputs Input tensors (in order matching operation operands)
     * @return Output tensors (in order matching operation results)
     */
    List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs);

    /**
     * Execute a single operation with varargs inputs.
     */
    default List<Tensor> execute(StableHloAst.Operation op, Tensor... inputs) {
        return execute(op, List.of(inputs));
    }

    /**
     * Allocate a tensor on this backend.
     *
     * @param spec The specification of the tensor to allocate
     * @return A newly allocated, zero-initialized tensor
     */
    Tensor allocate(TensorSpec spec);

    /**
     * Check if this backend supports a specific operation.
     *
     * @param op The operation to check
     * @return true if the operation is supported
     */
    default boolean supports(StableHloAst.Operation op) {
        return true; // Default: assume all operations supported
    }

    /**
     * Close this backend and release any resources.
     */
    @Override
    void close();
}
