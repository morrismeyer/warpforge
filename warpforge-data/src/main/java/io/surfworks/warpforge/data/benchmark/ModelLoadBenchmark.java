package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.WarpForge;
import io.surfworks.warpforge.data.hub.ProgressListener;
import io.surfworks.warpforge.data.model.ModelSource;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Benchmark for model loading performance.
 *
 * <p>Measures:
 * <ul>
 *   <li>Time to download model from HuggingFace (first run)</li>
 *   <li>Time to load model from cache (subsequent runs)</li>
 *   <li>Memory mapping performance</li>
 *   <li>Tensor access patterns</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * ModelLoadBenchmark benchmark = ModelLoadBenchmark.builder("gpt2-loading")
 *     .modelId("gpt2")
 *     .accessAllTensors(true)
 *     .build();
 *
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class ModelLoadBenchmark implements ModelBenchmark {

    private final String name;
    private final String modelId;
    private final boolean accessAllTensors;
    private final boolean warmupTensorAccess;
    private final Path cacheDir;

    private ModelSource model;

    private ModelLoadBenchmark(Builder builder) {
        this.name = builder.name;
        this.modelId = builder.modelId;
        this.accessAllTensors = builder.accessAllTensors;
        this.warmupTensorAccess = builder.warmupTensorAccess;
        this.cacheDir = builder.cacheDir;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public void setup(BenchmarkConfig config) throws IOException {
        // Configure WarpForge for this benchmark
        WarpForge.ConfigBuilder configBuilder = WarpForge.configure()
                .progressListener(ProgressListener.NONE);

        if (cacheDir != null) {
            configBuilder.cacheDir(cacheDir);
        }

        configBuilder.apply();
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        // No inputs needed - we're benchmarking the load itself
        return Map.of();
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        // Close any previously loaded model
        if (model != null) {
            model.close();
        }

        // Load the model (this is what we're benchmarking)
        model = WarpForge.model(modelId);

        // Optionally access all tensors to ensure they're memory-mapped
        if (accessAllTensors) {
            for (String tensorName : model.tensorNames()) {
                TensorView tensor = model.tensor(tensorName);
                if (warmupTensorAccess && tensor.info().elementCount() > 0) {
                    // Access first element to trigger page fault
                    tensor.getFloatFlat(0);
                }
            }
        }

        // Return model info as "outputs" for validation
        return Map.of();
    }

    @Override
    public List<String> outputsToValidate() {
        // No tensor outputs to validate
        return List.of();
    }

    @Override
    public void teardown() {
        if (model != null) {
            model.close();
            model = null;
        }
    }

    /**
     * Get the loaded model (available after runInference).
     */
    public ModelSource getLoadedModel() {
        return model;
    }

    /**
     * Create a builder for ModelLoadBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for ModelLoadBenchmark.
     */
    public static final class Builder {
        private final String name;
        private String modelId;
        private boolean accessAllTensors = false;
        private boolean warmupTensorAccess = false;
        private Path cacheDir = null;

        Builder(String name) {
            this.name = name;
        }

        /**
         * Set the HuggingFace model ID or local path.
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Whether to iterate through all tensors after loading.
         * This ensures all memory mappings are established.
         */
        public Builder accessAllTensors(boolean access) {
            this.accessAllTensors = access;
            return this;
        }

        /**
         * Whether to read the first element of each tensor.
         * This triggers actual memory page faults for mmap'd files.
         */
        public Builder warmupTensorAccess(boolean warmup) {
            this.warmupTensorAccess = warmup;
            return this;
        }

        /**
         * Set a custom cache directory.
         */
        public Builder cacheDir(Path cacheDir) {
            this.cacheDir = cacheDir;
            return this;
        }

        public ModelLoadBenchmark build() {
            if (modelId == null) {
                throw new IllegalStateException("modelId must be set");
            }
            return new ModelLoadBenchmark(this);
        }
    }
}
