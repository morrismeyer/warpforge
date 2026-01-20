package io.surfworks.snakeburger.codegen;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Tracks SSA value-to-index mappings during code generation.
 * Similar to GraphCompiler's CompilationContext but tailored for bytecode emission.
 */
final class CodegenContext {
    private final Map<String, Integer> valueIndices = new HashMap<>();
    private final List<TensorSpec> tensorSpecs = new ArrayList<>();
    private final List<StableHloAst.Operation> operations = new ArrayList<>();

    /**
     * Register a new SSA value and return its index.
     *
     * @param name The SSA value name (e.g., "arg0", "1")
     * @param spec The tensor specification
     * @return The assigned tensor index
     */
    int registerValue(String name, TensorSpec spec) {
        int index = tensorSpecs.size();
        valueIndices.put(name, index);
        tensorSpecs.add(spec);
        return index;
    }

    /**
     * Get the index for an SSA value.
     *
     * @param name The SSA value name
     * @return The tensor index
     * @throws IllegalArgumentException if the value is not registered
     */
    int getValueIndex(String name) {
        Integer index = valueIndices.get(name);
        if (index == null) {
            throw new IllegalArgumentException("Unknown SSA value: %" + name);
        }
        return index;
    }

    /**
     * Check if a value is registered.
     */
    boolean hasValue(String name) {
        return valueIndices.containsKey(name);
    }

    /**
     * Record an operation for later serialization.
     */
    void recordOperation(StableHloAst.Operation op) {
        operations.add(op);
    }

    /**
     * Get the total number of tensors (inputs + intermediates + outputs).
     */
    int tensorCount() {
        return tensorSpecs.size();
    }

    /**
     * Get all tensor specifications.
     */
    List<TensorSpec> tensorSpecs() {
        return tensorSpecs;
    }

    /**
     * Get all recorded operations.
     */
    List<StableHloAst.Operation> operations() {
        return operations;
    }

    /**
     * Convert a StableHLO type to a TensorSpec.
     */
    static TensorSpec tensorSpecFromType(StableHloAst.Type type) {
        if (type instanceof StableHloAst.TensorType tt) {
            return TensorSpec.fromAst(tt);
        }
        throw new IllegalArgumentException("Expected TensorType but got: " + type.getClass().getSimpleName());
    }
}
