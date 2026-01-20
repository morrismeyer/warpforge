package io.surfworks.warpforge.codegen.api;

import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;

/**
 * Interface for compiled StableHLO models.
 *
 * <p>This interface is implemented by bytecode-generated model classes.
 * The generated code has NO Babylon runtime dependencies, enabling
 * native-image compilation and deployment without the Babylon JDK.
 *
 * <h2>Execution Model</h2>
 * <pre>
 * Build Time (Babylon JDK 26):
 *   StableHLO MLIR → BabylonFuncOpBuilder → BytecodeGenerator → Model.class
 *
 * Runtime (No Babylon):
 *   Model.class + Backend → model.forward(inputs, backend) → outputs
 * </pre>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Load the compiled model (e.g., from JAR)
 * CompiledModel model = ModelLoader.load("model.jar");
 *
 * // Create inputs
 * List<Tensor> inputs = List.of(
 *     Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2),
 *     Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 2, 2)
 * );
 *
 * // Execute with a backend
 * try (Backend backend = new CpuBackend()) {
 *     List<Tensor> outputs = model.forward(inputs, backend);
 *     // Process outputs...
 * }
 * }</pre>
 */
public interface CompiledModel {

    /**
     * Execute the forward pass of this model.
     *
     * @param inputs  Input tensors, in order matching the model's function signature
     * @param backend The backend to use for executing operations
     * @return Output tensors, in order matching the model's return values
     * @throws IllegalArgumentException if the number of inputs doesn't match {@link #inputCount()}
     */
    List<Tensor> forward(List<Tensor> inputs, Backend backend);

    /**
     * Returns the number of input tensors this model expects.
     *
     * @return The number of inputs
     */
    int inputCount();

    /**
     * Returns the number of output tensors this model produces.
     *
     * @return The number of outputs
     */
    int outputCount();

    /**
     * Returns metadata about this compiled model.
     *
     * @return The model metadata
     */
    ModelMetadata metadata();

    /**
     * Execute the forward pass with varargs inputs (convenience method).
     *
     * @param backend The backend to use
     * @param inputs  Input tensors
     * @return Output tensors
     */
    default List<Tensor> forward(Backend backend, Tensor... inputs) {
        return forward(List.of(inputs), backend);
    }
}
