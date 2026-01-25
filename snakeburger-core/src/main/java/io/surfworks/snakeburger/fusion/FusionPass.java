package io.surfworks.snakeburger.fusion;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;

/**
 * Applies fusion patterns to a StableHLO module.
 *
 * <p>FusionPass walks the operation graph and applies registered patterns to
 * detect fusable subgraphs. When a pattern matches, the matched operations are
 * replaced with a single {@link FusedOperation}.
 *
 * <p>Patterns are applied in priority order (highest estimated speedup first).
 * Once an operation is part of a fused subgraph, it is not considered for
 * other patterns.
 *
 * <p>Example usage:
 * <pre>{@code
 * FusionPass pass = new FusionPass()
 *     .addPattern(new SoftmaxFusion())
 *     .addPattern(new LayerNormFusion())
 *     .addPattern(new RMSNormFusion());
 *
 * Module optimized = pass.apply(module);
 *
 * // Check fusion statistics
 * System.out.println("Fusions applied: " + pass.lastFusionCount());
 * }</pre>
 */
public final class FusionPass {

    private final List<FusionPattern> patterns;
    private int lastFusionCount;
    private final List<FusionMatch> lastMatches;

    /**
     * Creates a FusionPass with no patterns.
     *
     * <p>Use {@link #addPattern(FusionPattern)} to register patterns.
     */
    public FusionPass() {
        this.patterns = new ArrayList<>();
        this.lastMatches = new ArrayList<>();
    }

    /**
     * Creates a FusionPass with the given patterns.
     *
     * @param patterns the patterns to apply
     */
    public FusionPass(List<FusionPattern> patterns) {
        this.patterns = new ArrayList<>(patterns);
        this.lastMatches = new ArrayList<>();
        sortPatternsByPriority();
    }

    /**
     * Adds a fusion pattern.
     *
     * @param pattern the pattern to add
     * @return this pass for chaining
     */
    public FusionPass addPattern(FusionPattern pattern) {
        patterns.add(pattern);
        sortPatternsByPriority();
        return this;
    }

    /**
     * Creates a FusionPass with the standard set of patterns.
     *
     * <p>Includes: SoftmaxFusion, LayerNormFusion, RMSNormFusion
     *
     * @return a pass with standard patterns
     */
    public static FusionPass withStandardPatterns() {
        return new FusionPass()
                .addPattern(new SoftmaxFusion())
                .addPattern(new LayerNormFusion())
                .addPattern(new RMSNormFusion());
    }

    /**
     * Applies fusion patterns to a module.
     *
     * @param module the module to optimize
     * @return a new module with fused operations
     */
    public Module apply(Module module) {
        lastFusionCount = 0;
        lastMatches.clear();

        List<Function> optimizedFunctions = module.functions().stream()
                .map(this::fuseFunction)
                .toList();

        return new Module(module.name(), optimizedFunctions);
    }

    /**
     * Applies fusion patterns to a single function.
     *
     * @param func the function to optimize
     * @return a new function with fused operations
     */
    public Function fuseFunction(Function func) {
        if (patterns.isEmpty()) {
            return func;
        }

        OperationGraph graph = OperationGraph.build(func);
        List<Operation> newBody = new ArrayList<>();
        Set<Operation> fusedAway = new HashSet<>();

        for (Operation op : func.body()) {
            // Skip if already consumed by a fusion
            if (fusedAway.contains(op)) {
                continue;
            }

            // Try each pattern (in priority order)
            Optional<FusionResult> fusionResult = tryPatterns(op, graph);

            if (fusionResult.isPresent()) {
                FusionResult result = fusionResult.get();
                fusedAway.addAll(result.match.matchedOps());
                newBody.add(result.fusedOp);
                lastMatches.add(result.match);
                lastFusionCount++;
            } else {
                newBody.add(op);
            }
        }

        return new Function(
                func.name(),
                func.arguments(),
                func.resultTypes(),
                newBody,
                func.isPublic()
        );
    }

    private Optional<FusionResult> tryPatterns(Operation op, OperationGraph graph) {
        for (FusionPattern pattern : patterns) {
            Optional<FusionMatch> match = pattern.match(op, graph);
            if (match.isPresent()) {
                FusionMatch m = match.get();
                // Verify the subgraph can be safely fused
                if (graph.canFuseWithoutDuplication(m.matchedOps(), m.output())) {
                    FusedOperation fusedOp = pattern.rewrite(m);
                    return Optional.of(new FusionResult(m, fusedOp));
                }
            }
        }
        return Optional.empty();
    }

    private void sortPatternsByPriority() {
        patterns.sort(Comparator.comparingDouble(FusionPattern::estimatedSpeedup).reversed());
    }

    /**
     * Returns the number of fusions applied in the last {@link #apply} call.
     */
    public int lastFusionCount() {
        return lastFusionCount;
    }

    /**
     * Returns the matches from the last {@link #apply} call.
     */
    public List<FusionMatch> lastMatches() {
        return List.copyOf(lastMatches);
    }

    /**
     * Returns the registered patterns.
     */
    public List<FusionPattern> patterns() {
        return List.copyOf(patterns);
    }

    @Override
    public String toString() {
        return String.format("FusionPass[patterns=%d, lastFusions=%d]",
                patterns.size(), lastFusionCount);
    }

    private record FusionResult(FusionMatch match, FusedOperation fusedOp) {}
}
