package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Interface for model-level benchmarks.
 *
 * <p>Implementations define how to:
 * <ul>
 *   <li>Prepare inputs for the model</li>
 *   <li>Run inference</li>
 *   <li>Identify which outputs to validate</li>
 * </ul>
 *
 * <p>Example implementation:
 * <pre>{@code
 * public class BertBenchmark implements ModelBenchmark {
 *     private final ModelSource model;
 *
 *     @Override
 *     public String name() { return "bert-base-qa"; }
 *
 *     @Override
 *     public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
 *         return Map.of(
 *             "input_ids", createInputIds(config.batchSize(), config.sequenceLength()),
 *             "attention_mask", createAttentionMask(config.batchSize(), config.sequenceLength())
 *         );
 *     }
 *
 *     @Override
 *     public Map<String, TensorView> runInference(Map<String, TensorView> inputs) {
 *         // Run through backend
 *         return model.forward(inputs);
 *     }
 *
 *     @Override
 *     public List<String> outputsToValidate() {
 *         return List.of("pooler_output", "last_hidden_state");
 *     }
 * }
 * }</pre>
 */
public interface ModelBenchmark {

    /**
     * Unique name for this benchmark.
     */
    String name();

    /**
     * Model identifier (e.g., HuggingFace repo ID or local path).
     */
    String modelId();

    /**
     * Initialize the benchmark (load model, etc.).
     * Called once before warmup.
     */
    void setup(BenchmarkConfig config) throws IOException;

    /**
     * Prepare inputs for a single inference iteration.
     *
     * @param config Benchmark configuration
     * @return Map of input name to tensor
     */
    Map<String, TensorView> prepareInputs(BenchmarkConfig config);

    /**
     * Run a single inference iteration.
     *
     * @param inputs Input tensors from prepareInputs()
     * @return Map of output name to tensor
     * @throws IOException if inference fails
     */
    Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException;

    /**
     * Names of outputs that should be validated against golden outputs.
     */
    List<String> outputsToValidate();

    /**
     * Clean up resources (close model, etc.).
     * Called after all iterations complete.
     */
    void teardown();

    /**
     * Optional: provide golden output IDs for validation.
     * Default implementation derives IDs from benchmark name and output names.
     *
     * @param outputName Name of the output tensor
     * @return Golden output ID to look up in GoldenStore
     */
    default String goldenIdFor(String outputName) {
        return name() + "/" + outputName;
    }

    /**
     * Optional: custom tolerance per output.
     * Returns null to use the config default.
     */
    default Double toleranceFor(String outputName) {
        return null;
    }
}
