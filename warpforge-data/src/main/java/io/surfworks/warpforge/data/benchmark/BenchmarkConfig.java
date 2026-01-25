package io.surfworks.warpforge.data.benchmark;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for a model benchmark run.
 *
 * <p>Example:
 * <pre>{@code
 * BenchmarkConfig config = BenchmarkConfig.builder("bert-base-uncased")
 *     .backend("cpu")
 *     .goldenDir(Path.of("goldens/bert"))
 *     .warmupIterations(5)
 *     .measurementIterations(20)
 *     .tolerance(1e-5)
 *     .build();
 * }</pre>
 */
public record BenchmarkConfig(
        String modelId,
        String backend,
        Path goldenDir,
        int warmupIterations,
        int measurementIterations,
        double tolerance,
        boolean validateOutputs,
        boolean collectMemoryStats,
        int batchSize,
        int sequenceLength
) {

    public BenchmarkConfig {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(backend, "backend must not be null");
        if (warmupIterations < 0) {
            throw new IllegalArgumentException("warmupIterations must be >= 0");
        }
        if (measurementIterations < 1) {
            throw new IllegalArgumentException("measurementIterations must be >= 1");
        }
        if (tolerance <= 0) {
            throw new IllegalArgumentException("tolerance must be > 0");
        }
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be >= 1");
        }
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be >= 1");
        }
    }

    /**
     * Default warmup iterations.
     */
    public static final int DEFAULT_WARMUP = 5;

    /**
     * Default measurement iterations.
     */
    public static final int DEFAULT_MEASUREMENT = 20;

    /**
     * Default tolerance for output comparison.
     */
    public static final double DEFAULT_TOLERANCE = 1e-5;

    /**
     * Default batch size.
     */
    public static final int DEFAULT_BATCH_SIZE = 1;

    /**
     * Default sequence length for transformer models.
     */
    public static final int DEFAULT_SEQUENCE_LENGTH = 128;

    /**
     * Create a builder for benchmark configuration.
     */
    public static Builder builder(String modelId) {
        return new Builder(modelId);
    }

    /**
     * Create a minimal configuration with defaults.
     */
    public static BenchmarkConfig defaults(String modelId) {
        return builder(modelId).build();
    }

    /**
     * Builder for BenchmarkConfig.
     */
    public static final class Builder {
        private final String modelId;
        private String backend = "cpu";
        private Path goldenDir = null;
        private int warmupIterations = DEFAULT_WARMUP;
        private int measurementIterations = DEFAULT_MEASUREMENT;
        private double tolerance = DEFAULT_TOLERANCE;
        private boolean validateOutputs = true;
        private boolean collectMemoryStats = false;
        private int batchSize = DEFAULT_BATCH_SIZE;
        private int sequenceLength = DEFAULT_SEQUENCE_LENGTH;

        Builder(String modelId) {
            this.modelId = Objects.requireNonNull(modelId);
        }

        /**
         * Set the backend to use (e.g., "cpu", "nvidia", "amd").
         */
        public Builder backend(String backend) {
            this.backend = backend;
            return this;
        }

        /**
         * Set the directory containing golden outputs.
         */
        public Builder goldenDir(Path goldenDir) {
            this.goldenDir = goldenDir;
            return this;
        }

        /**
         * Set the number of warmup iterations.
         */
        public Builder warmupIterations(int warmupIterations) {
            this.warmupIterations = warmupIterations;
            return this;
        }

        /**
         * Set the number of measurement iterations.
         */
        public Builder measurementIterations(int measurementIterations) {
            this.measurementIterations = measurementIterations;
            return this;
        }

        /**
         * Set the tolerance for output comparison.
         */
        public Builder tolerance(double tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        /**
         * Enable or disable output validation against golden outputs.
         */
        public Builder validateOutputs(boolean validateOutputs) {
            this.validateOutputs = validateOutputs;
            return this;
        }

        /**
         * Enable memory statistics collection.
         */
        public Builder collectMemoryStats(boolean collectMemoryStats) {
            this.collectMemoryStats = collectMemoryStats;
            return this;
        }

        /**
         * Set the batch size for inference.
         */
        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * Set the sequence length for transformer models.
         */
        public Builder sequenceLength(int sequenceLength) {
            this.sequenceLength = sequenceLength;
            return this;
        }

        public BenchmarkConfig build() {
            return new BenchmarkConfig(
                    modelId, backend, goldenDir, warmupIterations, measurementIterations,
                    tolerance, validateOutputs, collectMemoryStats, batchSize, sequenceLength
            );
        }
    }
}
