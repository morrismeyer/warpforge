package io.surfworks.warpforge.runner;

import io.surfworks.warpforge.codegen.api.CompiledModel;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Registry for AOT-compiled models in native-image (Mode 3).
 *
 * <p>In Mode 3 (Native), models are compiled directly into the native binary
 * at build time. This eliminates the need for dynamic JAR loading, providing
 * the fastest possible execution with the smallest footprint.
 *
 * <h2>Architecture</h2>
 * <pre>
 * Build Time (Babylon JDK):
 *   1. StableHLO MLIR → snakeburger-codegen → Model.class
 *   2. Model.class registered in NativeModelRegistry
 *   3. native-image compiles everything together
 *
 * Runtime (Native):
 *   1. NativeModelRegistry.get("modelName") → CompiledModel
 *   2. model.forward(inputs, backend) → outputs
 *   3. No JVM, no class loading, pure native execution
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>
 * // Register models at build time (in static initializer or GraalVM Feature)
 * NativeModelRegistry.register("addmodel", new AddModel());
 *
 * // Use at runtime
 * CompiledModel model = NativeModelRegistry.get("addmodel").orElseThrow();
 * List&lt;Tensor&gt; outputs = model.forward(inputs, backend);
 * </pre>
 *
 * <h2>Future Work</h2>
 * <ul>
 *   <li>Auto-registration via GraalVM native-image Feature</li>
 *   <li>Model discovery from build directory</li>
 *   <li>Model versioning and compatibility checking</li>
 * </ul>
 */
public final class NativeModelRegistry {

    private static final Map<String, CompiledModel> REGISTRY = new HashMap<>();

    private NativeModelRegistry() {}

    /**
     * Register a compiled model with the given name.
     *
     * <p>This should be called at build time (during native-image generation)
     * to make models available at runtime.
     *
     * @param name  Model name (e.g., "addmodel", "mlp-classifier")
     * @param model The compiled model instance
     */
    public static void register(String name, CompiledModel model) {
        REGISTRY.put(name.toLowerCase(), model);
    }

    /**
     * Get a compiled model by name.
     *
     * @param name Model name (case-insensitive)
     * @return The model, or empty if not registered
     */
    public static Optional<CompiledModel> get(String name) {
        return Optional.ofNullable(REGISTRY.get(name.toLowerCase()));
    }

    /**
     * Check if a model is registered.
     *
     * @param name Model name (case-insensitive)
     * @return true if the model is registered
     */
    public static boolean contains(String name) {
        return REGISTRY.containsKey(name.toLowerCase());
    }

    /**
     * Get all registered model names.
     *
     * @return Set of model names
     */
    public static Set<String> listModels() {
        return Set.copyOf(REGISTRY.keySet());
    }

    /**
     * Get the number of registered models.
     *
     * @return Number of models
     */
    public static int size() {
        return REGISTRY.size();
    }

    /**
     * Check if any models are registered.
     *
     * @return true if at least one model is registered
     */
    public static boolean isEmpty() {
        return REGISTRY.isEmpty();
    }
}
