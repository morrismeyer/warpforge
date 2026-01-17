package io.surfworks.warpforge.backend.cpu.integration;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReturnOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.warpforge.backend.cpu.CpuBackend;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple interpreter that executes StableHLO modules using the CPU backend.
 * Maps SSA values to tensors and executes operations in order.
 */
public class StableHloInterpreter {

    private final CpuBackend backend;
    private final Map<String, Tensor> valueMap = new HashMap<>();

    public StableHloInterpreter() {
        this.backend = new CpuBackend();
    }

    public StableHloInterpreter(CpuBackend backend) {
        this.backend = backend;
    }

    /**
     * Execute a module's forward function with given inputs.
     *
     * @param module The parsed StableHLO module
     * @param inputs Input tensors in order
     * @return Output tensors from the forward function
     */
    public List<Tensor> execute(StableHloAst.Module module, List<Tensor> inputs) {
        // Find the forward function
        Function forwardFn = module.getFunction("forward")
            .orElseThrow(() -> new IllegalArgumentException("Module has no 'forward' function"));

        return executeFunction(forwardFn, inputs);
    }

    /**
     * Execute a function with given inputs.
     */
    public List<Tensor> executeFunction(Function function, List<Tensor> inputs) {
        valueMap.clear();

        // Bind inputs to function arguments
        List<Argument> args = function.arguments();
        if (args.size() != inputs.size()) {
            throw new IllegalArgumentException(
                "Input count mismatch: expected " + args.size() + ", got " + inputs.size());
        }

        for (int i = 0; i < args.size(); i++) {
            valueMap.put(args.get(i).name(), inputs.get(i));
        }

        // Execute each operation in the function body
        List<Tensor> result = null;
        for (Operation op : function.body()) {
            result = executeOperation(op);
        }

        return result;
    }

    /**
     * Execute a single operation.
     */
    private List<Tensor> executeOperation(Operation op) {
        // Handle return operation specially
        if (op instanceof ReturnOp returnOp) {
            return returnOp.operands().stream()
                .map(v -> valueMap.get(v.name()))
                .toList();
        }

        // Gather input tensors from value map
        List<Tensor> opInputs = op.operands().stream()
            .map(v -> {
                Tensor t = valueMap.get(v.name());
                if (t == null) {
                    throw new IllegalStateException("Undefined value: " + v.name());
                }
                return t;
            })
            .toList();

        // Execute through backend
        List<Tensor> outputs;
        try {
            outputs = backend.execute(op, opInputs);
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedOperationException(
                "Operation not supported by CPU backend: " + op.opName(), e);
        }

        // Store outputs in value map
        List<Value> results = op.results();
        for (int i = 0; i < results.size(); i++) {
            valueMap.put(results.get(i).name(), outputs.get(i));
        }

        return outputs;
    }

    /**
     * Get a value from the current execution context.
     */
    public Tensor getValue(String name) {
        return valueMap.get(name);
    }

    public void close() {
        backend.close();
    }
}
