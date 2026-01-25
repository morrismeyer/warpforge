package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * A synthetic benchmark for testing the benchmark infrastructure.
 *
 * <p>This benchmark doesn't use a real model - it generates random inputs
 * and produces deterministic outputs based on a simple computation.
 * Useful for testing the runner, validation, and reporting.
 *
 * <p>Example:
 * <pre>{@code
 * SyntheticBenchmark benchmark = SyntheticBenchmark.builder("test-synthetic")
 *     .inputShape(1, 128, 768)  // [batch, seq, hidden]
 *     .outputShape(1, 768)       // [batch, hidden]
 *     .simulatedLatencyMs(10)    // ~10ms per inference
 *     .build();
 *
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class SyntheticBenchmark implements ModelBenchmark {

    private final String name;
    private final String modelId;
    private final long[] inputShape;
    private final long[] outputShape;
    private final long simulatedLatencyNanos;
    private final float outputMultiplier;

    private Arena arena;
    private MemorySegment inputData;
    private MemorySegment outputData;

    private SyntheticBenchmark(Builder builder) {
        this.name = builder.name;
        this.modelId = builder.modelId;
        this.inputShape = builder.inputShape;
        this.outputShape = builder.outputShape;
        this.simulatedLatencyNanos = builder.simulatedLatencyNanos;
        this.outputMultiplier = builder.outputMultiplier;
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
    public void setup(BenchmarkConfig config) {
        arena = Arena.ofShared();

        // Allocate input tensor
        long inputElements = elementCount(inputShape);
        inputData = arena.allocate(inputElements * 4);

        // Fill with random data
        Random random = new Random(42); // Fixed seed for reproducibility
        for (long i = 0; i < inputElements; i++) {
            inputData.setAtIndex(ValueLayout.JAVA_FLOAT, i, random.nextFloat());
        }

        // Allocate output tensor
        long outputElements = elementCount(outputShape);
        outputData = arena.allocate(outputElements * 4);
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        TensorInfo inputInfo = new TensorInfo("input", DType.F32, inputShape, 0, inputData.byteSize());
        return Map.of("input", new TensorView(inputData, inputInfo));
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        // Simulate computation time
        if (simulatedLatencyNanos > 0) {
            long startNanos = System.nanoTime();
            while (System.nanoTime() - startNanos < simulatedLatencyNanos) {
                // Busy wait to simulate work
                Thread.onSpinWait();
            }
        }

        // Compute a deterministic output based on input
        TensorView input = inputs.get("input");
        long outputElements = elementCount(outputShape);

        // Simple reduction: mean across all but the last dimension, scaled
        float sum = 0;
        long inputElements = input.info().elementCount();
        for (long i = 0; i < Math.min(inputElements, 1000); i++) {
            sum += input.getFloatFlat(i);
        }
        float mean = sum / Math.min(inputElements, 1000);

        // Fill output with scaled mean
        for (long i = 0; i < outputElements; i++) {
            float value = mean * outputMultiplier + (i * 0.001f);
            outputData.setAtIndex(ValueLayout.JAVA_FLOAT, i, value);
        }

        TensorInfo outputInfo = new TensorInfo("output", DType.F32, outputShape, 0, outputData.byteSize());
        Map<String, TensorView> outputs = new HashMap<>();
        outputs.put("output", new TensorView(outputData, outputInfo));
        return outputs;
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("output");
    }

    @Override
    public void teardown() {
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    private static long elementCount(long[] shape) {
        long count = 1;
        for (long dim : shape) count *= dim;
        return count;
    }

    /**
     * Create a builder for SyntheticBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for SyntheticBenchmark.
     */
    public static final class Builder {
        private final String name;
        private String modelId = "synthetic";
        private long[] inputShape = {1, 128, 768};
        private long[] outputShape = {1, 768};
        private long simulatedLatencyNanos = 0;
        private float outputMultiplier = 1.0f;

        Builder(String name) {
            this.name = name;
        }

        /**
         * Set the model ID.
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Set the input tensor shape.
         */
        public Builder inputShape(long... shape) {
            this.inputShape = shape.clone();
            return this;
        }

        /**
         * Set the output tensor shape.
         */
        public Builder outputShape(long... shape) {
            this.outputShape = shape.clone();
            return this;
        }

        /**
         * Set simulated latency in milliseconds.
         */
        public Builder simulatedLatencyMs(long ms) {
            this.simulatedLatencyNanos = ms * 1_000_000;
            return this;
        }

        /**
         * Set simulated latency in nanoseconds.
         */
        public Builder simulatedLatencyNanos(long nanos) {
            this.simulatedLatencyNanos = nanos;
            return this;
        }

        /**
         * Set the output multiplier (for deterministic output generation).
         */
        public Builder outputMultiplier(float multiplier) {
            this.outputMultiplier = multiplier;
            return this;
        }

        public SyntheticBenchmark build() {
            return new SyntheticBenchmark(this);
        }
    }
}
