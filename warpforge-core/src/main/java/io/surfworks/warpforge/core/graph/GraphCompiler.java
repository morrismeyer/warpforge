package io.surfworks.warpforge.core.graph;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Compiles StableHLO AST into an ExecutableGraph.
 * Resolves SSA value names to tensor indices and extracts tensor specifications.
 */
public final class GraphCompiler {

    private GraphCompiler() {} // Utility class

    /**
     * Compile a StableHLO module, using the first public function.
     *
     * @param module The module to compile
     * @return Compiled executable graph
     * @throws IllegalArgumentException if the module has no public functions
     */
    public static ExecutableGraph compile(StableHloAst.Module module) {
        var publicFunctions = module.functions().stream()
            .filter(StableHloAst.Function::isPublic)
            .toList();

        if (publicFunctions.isEmpty()) {
            // Fall back to first function if no public functions
            if (module.functions().isEmpty()) {
                throw new IllegalArgumentException("Module has no functions");
            }
            return compile(module.functions().getFirst());
        }
        return compile(publicFunctions.getFirst());
    }

    /**
     * Compile a StableHLO function.
     *
     * @param function The function to compile
     * @return Compiled executable graph
     */
    public static ExecutableGraph compile(StableHloAst.Function function) {
        CompilationContext ctx = new CompilationContext();

        // Register function arguments
        int[] inputIndices = new int[function.arguments().size()];
        for (int i = 0; i < function.arguments().size(); i++) {
            var arg = function.arguments().get(i);
            String valueName = arg.name();
            TensorSpec spec = tensorSpecFromType(arg.type());
            inputIndices[i] = ctx.registerValue(valueName, spec);
        }

        // Process operations
        List<GraphNode> nodes = new ArrayList<>();
        int[] outputIndices = null;

        for (var op : function.body()) {
            if (op instanceof StableHloAst.ReturnOp returnOp) {
                // Return op - capture output indices
                outputIndices = resolveOperandIndices(returnOp.operands(), ctx);
                nodes.add(new GraphNode(returnOp, outputIndices, new int[0]));
            } else {
                // Regular operation
                int[] opInputs = resolveOperandIndices(op.operands(), ctx);
                int[] opOutputs = registerResults(op.results(), op.tensorResultType(), ctx);
                nodes.add(new GraphNode(op, opInputs, opOutputs));
            }
        }

        if (outputIndices == null) {
            throw new IllegalArgumentException("Function missing return statement");
        }

        return new ExecutableGraph(
            function.name(),
            nodes,
            ctx.tensorSpecs(),
            inputIndices,
            outputIndices
        );
    }

    /**
     * Compile a named function from a module.
     *
     * @param module       The module containing the function
     * @param functionName The name of the function to compile
     * @return Compiled executable graph
     * @throws IllegalArgumentException if the function is not found
     */
    public static ExecutableGraph compile(StableHloAst.Module module, String functionName) {
        return module.getFunction(functionName)
            .map(GraphCompiler::compile)
            .orElseThrow(() -> new IllegalArgumentException(
                "Function '" + functionName + "' not found in module"));
    }

    // ==================== Internal Helpers ====================

    private static int[] resolveOperandIndices(List<StableHloAst.Value> operands, CompilationContext ctx) {
        int[] indices = new int[operands.size()];
        for (int i = 0; i < operands.size(); i++) {
            indices[i] = ctx.getValueIndex(operands.get(i).name());
        }
        return indices;
    }

    private static int[] registerResults(List<StableHloAst.Value> results,
                                         StableHloAst.TensorType resultType,
                                         CompilationContext ctx) {
        int[] indices = new int[results.size()];
        for (int i = 0; i < results.size(); i++) {
            var result = results.get(i);
            TensorSpec spec;
            if (result.type() instanceof StableHloAst.TensorType tt) {
                spec = tensorSpecFromType(tt);
            } else if (resultType != null) {
                spec = tensorSpecFromType(resultType);
            } else {
                throw new IllegalArgumentException("Cannot determine type for result: " + result.name());
            }
            indices[i] = ctx.registerValue(result.name(), spec);
        }
        return indices;
    }

    private static TensorSpec tensorSpecFromType(StableHloAst.Type type) {
        if (type instanceof StableHloAst.TensorType tt) {
            return TensorSpec.fromAst(tt);
        }
        throw new IllegalArgumentException("Expected TensorType but got: " + type.getClass().getSimpleName());
    }

    /**
     * Compilation context for tracking SSA values and tensor specs.
     */
    private static class CompilationContext {
        private final Map<String, Integer> valueIndices = new HashMap<>();
        private final List<TensorSpec> specs = new ArrayList<>();

        int registerValue(String name, TensorSpec spec) {
            int index = specs.size();
            valueIndices.put(name, index);
            specs.add(spec);
            return index;
        }

        int getValueIndex(String name) {
            Integer index = valueIndices.get(name);
            if (index == null) {
                throw new IllegalArgumentException("Unknown value: %" + name);
            }
            return index;
        }

        List<TensorSpec> tensorSpecs() {
            return specs;
        }
    }
}
