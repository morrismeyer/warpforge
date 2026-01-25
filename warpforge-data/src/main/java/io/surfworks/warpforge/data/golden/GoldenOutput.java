package io.surfworks.warpforge.data.golden;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.Instant;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * Represents a golden (expected) output for benchmark validation.
 *
 * <p>Golden outputs capture the expected result of a model inference or operation,
 * along with metadata about how it was generated. This allows:
 * <ul>
 *   <li>Verifying inference correctness across different backends</li>
 *   <li>Detecting regressions when PyTorch or models are updated</li>
 *   <li>Comparing outputs with configurable tolerance</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Create a golden output
 * GoldenOutput golden = GoldenOutput.builder("bert-base/pooler_output")
 *     .tensor(outputTensor)
 *     .pytorchVersion("2.7.0")
 *     .modelId("bert-base-uncased")
 *     .inputHash("abc123")
 *     .build();
 *
 * // Compare actual output
 * ComparisonResult result = golden.compare(actualTensor, 1e-5);
 * if (!result.matches()) {
 *     System.out.println("Mismatch: " + result.summary());
 * }
 * }</pre>
 *
 * @param id Unique identifier for this golden output (e.g., "bert-base/pooler_output")
 * @param tensorInfo Metadata about the tensor (dtype, shape)
 * @param data Raw tensor data
 * @param metadata Additional metadata (pytorch version, model id, input hash, etc.)
 * @param createdAt When this golden output was generated
 */
public record GoldenOutput(
        String id,
        TensorInfo tensorInfo,
        MemorySegment data,
        Map<String, String> metadata,
        Instant createdAt
) {

    public GoldenOutput {
        Objects.requireNonNull(id, "id must not be null");
        Objects.requireNonNull(tensorInfo, "tensorInfo must not be null");
        Objects.requireNonNull(data, "data must not be null");
        Objects.requireNonNull(metadata, "metadata must not be null");
        Objects.requireNonNull(createdAt, "createdAt must not be null");
        metadata = Map.copyOf(metadata);
    }

    /**
     * Standard metadata keys.
     */
    public static final String KEY_PYTORCH_VERSION = "pytorch_version";
    public static final String KEY_MODEL_ID = "model_id";
    public static final String KEY_INPUT_HASH = "input_hash";
    public static final String KEY_DESCRIPTION = "description";
    public static final String KEY_TOLERANCE = "tolerance";

    /**
     * Create a new builder for constructing golden outputs.
     */
    public static Builder builder(String id) {
        return new Builder(id);
    }

    /**
     * Get the tensor as a TensorView for inspection.
     */
    public TensorView toTensorView() {
        return new TensorView(data, tensorInfo);
    }

    /**
     * Get the PyTorch version used to generate this golden output.
     */
    public String pytorchVersion() {
        return metadata.get(KEY_PYTORCH_VERSION);
    }

    /**
     * Get the model ID this golden output was generated from.
     */
    public String modelId() {
        return metadata.get(KEY_MODEL_ID);
    }

    /**
     * Get the hash of the input used to generate this output.
     */
    public String inputHash() {
        return metadata.get(KEY_INPUT_HASH);
    }

    /**
     * Get the recommended tolerance for comparison.
     */
    public double tolerance() {
        String tol = metadata.get(KEY_TOLERANCE);
        return tol != null ? Double.parseDouble(tol) : 1e-5;
    }

    /**
     * Compare this golden output against an actual tensor.
     *
     * @param actual The actual output to compare
     * @param tolerance Maximum allowed difference (absolute)
     * @return Comparison result with match status and statistics
     */
    public ComparisonResult compare(TensorView actual, double tolerance) {
        return GoldenComparison.compare(this.toTensorView(), actual, tolerance);
    }

    /**
     * Compare this golden output against an actual tensor using the stored tolerance.
     */
    public ComparisonResult compare(TensorView actual) {
        return compare(actual, tolerance());
    }

    /**
     * Builder for constructing GoldenOutput instances.
     */
    public static final class Builder {
        private final String id;
        private TensorInfo tensorInfo;
        private MemorySegment data;
        private final java.util.HashMap<String, String> metadata = new java.util.HashMap<>();
        private Instant createdAt = Instant.now();

        Builder(String id) {
            this.id = Objects.requireNonNull(id);
        }

        /**
         * Set the tensor data from a TensorView.
         */
        public Builder tensor(TensorView view) {
            this.tensorInfo = view.info();
            this.data = view.data();
            return this;
        }

        /**
         * Set the tensor data from raw components.
         */
        public Builder tensor(DType dtype, long[] shape, MemorySegment data) {
            this.tensorInfo = new TensorInfo(id, dtype, shape, 0, data.byteSize());
            this.data = data;
            return this;
        }

        /**
         * Set the tensor data from a float array.
         */
        public Builder tensor(long[] shape, float[] data, Arena arena) {
            long count = 1;
            for (long dim : shape) count *= dim;
            if (count != data.length) {
                throw new IllegalArgumentException(
                        "Shape " + Arrays.toString(shape) + " requires " + count +
                                " elements, but got " + data.length);
            }

            MemorySegment segment = arena.allocate(data.length * 4L);
            for (int i = 0; i < data.length; i++) {
                segment.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, data[i]);
            }

            this.tensorInfo = new TensorInfo(id, DType.F32, shape, 0, segment.byteSize());
            this.data = segment;
            return this;
        }

        /**
         * Set PyTorch version metadata.
         */
        public Builder pytorchVersion(String version) {
            metadata.put(KEY_PYTORCH_VERSION, version);
            return this;
        }

        /**
         * Set model ID metadata.
         */
        public Builder modelId(String modelId) {
            metadata.put(KEY_MODEL_ID, modelId);
            return this;
        }

        /**
         * Set input hash metadata.
         */
        public Builder inputHash(String hash) {
            metadata.put(KEY_INPUT_HASH, hash);
            return this;
        }

        /**
         * Set description metadata.
         */
        public Builder description(String description) {
            metadata.put(KEY_DESCRIPTION, description);
            return this;
        }

        /**
         * Set recommended tolerance for comparison.
         */
        public Builder tolerance(double tolerance) {
            metadata.put(KEY_TOLERANCE, String.valueOf(tolerance));
            return this;
        }

        /**
         * Add custom metadata.
         */
        public Builder metadata(String key, String value) {
            metadata.put(key, value);
            return this;
        }

        /**
         * Set creation timestamp (default is now).
         */
        public Builder createdAt(Instant instant) {
            this.createdAt = Objects.requireNonNull(instant);
            return this;
        }

        /**
         * Build the GoldenOutput.
         */
        public GoldenOutput build() {
            if (tensorInfo == null || data == null) {
                throw new IllegalStateException("Tensor data must be set");
            }
            return new GoldenOutput(id, tensorInfo, data, metadata, createdAt);
        }
    }
}
