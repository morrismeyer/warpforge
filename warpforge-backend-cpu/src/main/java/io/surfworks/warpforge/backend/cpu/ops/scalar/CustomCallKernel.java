package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.custom_call.
 *
 * <p>Custom calls invoke external functions identified by a call_target_name.
 * This kernel maintains a registry of known custom call implementations.
 * Unknown custom calls throw an error.
 */
public final class CustomCallKernel implements OpKernel {

    private final Map<String, BiFunction<CustomCallOp, List<Tensor>, List<Tensor>>> handlers =
        new ConcurrentHashMap<>();

    public CustomCallKernel() {
        registerDefaultHandlers();
    }

    /**
     * Register a handler for a custom call target.
     */
    public void registerHandler(String targetName,
                                 BiFunction<CustomCallOp, List<Tensor>, List<Tensor>> handler) {
        handlers.put(targetName, handler);
    }

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        CustomCallOp customOp = (CustomCallOp) op;
        String targetName = customOp.callTarget();

        BiFunction<CustomCallOp, List<Tensor>, List<Tensor>> handler = handlers.get(targetName);
        if (handler != null) {
            return handler.apply(customOp, inputs);
        }

        // Check for common patterns that can be handled generically
        if (targetName.startsWith("__") && targetName.endsWith("__")) {
            // Likely an internal marker that can be ignored
            return inputs.isEmpty() ? List.of(Tensor.zeros(new int[]{1})) :
                   List.of(inputs.get(0).copy());
        }

        throw new UnsupportedOperationException(
            "Unknown custom_call target: " + targetName + ". " +
            "Register a handler using CustomCallKernel.registerHandler()");
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.CustomCallOp;
    }

    private void registerDefaultHandlers() {
        // Identity passthrough for debug markers
        handlers.put("debug_marker", (op, inputs) ->
            inputs.isEmpty() ? List.of(Tensor.zeros(new int[]{1})) :
                              List.of(inputs.get(0).copy()));

        // CUDA synchronization (no-op on CPU)
        handlers.put("cuda_stream_synchronize", (op, inputs) ->
            List.of(Tensor.zeros(new int[]{1})));

        // Memory allocation hints (no-op on CPU)
        handlers.put("memory_allocation_hint", (op, inputs) ->
            inputs.isEmpty() ? List.of(Tensor.zeros(new int[]{1})) :
                              List.of(inputs.get(0).copy()));
    }
}
