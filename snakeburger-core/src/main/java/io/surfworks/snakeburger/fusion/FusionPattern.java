package io.surfworks.snakeburger.fusion;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import io.surfworks.snakeburger.stablehlo.FusedOperation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;

/**
 * Interface for fusion patterns that match and rewrite operation subgraphs.
 *
 * <p>A FusionPattern defines:
 * <ul>
 *   <li>A name identifying the pattern</li>
 *   <li>A matching function that detects the pattern in an operation graph</li>
 *   <li>A rewrite function that creates the fused replacement operation</li>
 *   <li>An estimated speedup for prioritization</li>
 * </ul>
 *
 * <p>Patterns are applied by {@link FusionPass}, which walks the operation graph
 * and applies patterns in priority order.
 *
 * <p>Example implementation:
 * <pre>{@code
 * public class SoftmaxFusion implements FusionPattern {
 *     @Override
 *     public String name() { return "softmax"; }
 *
 *     @Override
 *     public Optional<FusionMatch> match(Operation op, OperationGraph graph) {
 *         // Match: div(exp(sub(x, max(x))), sum(exp(sub(x, max(x)))))
 *         if (!(op instanceof DivideOp div)) return Optional.empty();
 *         // ... pattern matching logic ...
 *         return Optional.of(new FusionMatch(...));
 *     }
 *
 *     @Override
 *     public FusedOperation rewrite(FusionMatch match) {
 *         return new FusedOperation("softmax", ...);
 *     }
 * }
 * }</pre>
 */
public interface FusionPattern {

    /**
     * Returns the unique name of this fusion pattern.
     *
     * <p>This name is used in logging, debugging, and as the fusion type
     * in the resulting {@link FusedOperation}.
     *
     * @return the pattern name (e.g., "softmax", "layer_norm", "rms_norm")
     */
    String name();

    /**
     * Attempts to match this pattern starting from the given operation.
     *
     * <p>Patterns typically match backwards from a "root" operation. For example,
     * softmax matching starts from the final divide operation and traces back
     * through exp, subtract, and reduce operations.
     *
     * <p>The graph parameter provides def-use information for traversing the
     * operation graph and checking for valid fusion conditions (e.g., single use).
     *
     * @param op the operation to try matching from (potential pattern root)
     * @param graph the operation graph for def-use analysis
     * @return a FusionMatch if the pattern matches, empty otherwise
     */
    Optional<FusionMatch> match(Operation op, OperationGraph graph);

    /**
     * Creates a fused operation from a successful match.
     *
     * <p>The returned FusedOperation will replace all operations in the match.
     *
     * @param match the successful pattern match
     * @return the fused replacement operation
     */
    FusedOperation rewrite(FusionMatch match);

    /**
     * Returns the estimated speedup factor for this fusion.
     *
     * <p>This is used to prioritize patterns when multiple could match.
     * Higher speedup patterns are tried first.
     *
     * <p>Typical values:
     * <ul>
     *   <li>Attention fusion: 5.0x (major memory savings)</li>
     *   <li>Softmax fusion: 3.0x (multiple passes reduced to one)</li>
     *   <li>LayerNorm fusion: 3.0x (multiple passes reduced to one)</li>
     *   <li>RMSNorm fusion: 2.5x (simpler than LayerNorm)</li>
     *   <li>Bias+Activation: 1.5x (simple element-wise fusion)</li>
     * </ul>
     *
     * @return the estimated speedup factor (default 1.5)
     */
    default double estimatedSpeedup() {
        return 1.5;
    }

    /**
     * Returns a human-readable description of what this pattern matches.
     *
     * @return the pattern description
     */
    default String description() {
        return name() + " fusion pattern";
    }

    /**
     * Creates a FusedOperation with standard configuration.
     *
     * <p>Helper method for pattern implementations.
     *
     * @param match the fusion match
     * @param config additional configuration attributes
     * @return the fused operation
     */
    default FusedOperation createFusedOp(FusionMatch match, Map<String, Object> config) {
        return new FusedOperation(
                name(),
                List.of(match.input()),
                List.of(match.output()),
                config,
                match.matchedOps()
        );
    }
}
