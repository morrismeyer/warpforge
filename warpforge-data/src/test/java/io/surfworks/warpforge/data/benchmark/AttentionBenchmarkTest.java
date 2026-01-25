package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AttentionBenchmarkTest {

    @Nested
    class BuilderTests {

        @Test
        void testBuilderDefaults() {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("test")
                    .build();

            assertEquals("test", benchmark.name());
            assertEquals("synthetic-attention", benchmark.modelId());
        }

        @Test
        void testBuilderWithCustomConfig() {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("custom")
                    .hiddenSize(1024)
                    .numHeads(16)
                    .causal(false)
                    .useFlashAttention(true)
                    .build();

            assertEquals("custom", benchmark.name());
        }

        @Test
        void testBuilderValidatesHiddenSizeDivisibility() {
            assertThrows(IllegalArgumentException.class, () ->
                    AttentionBenchmark.builder("invalid")
                            .hiddenSize(768)
                            .numHeads(11) // 768 not divisible by 11
                            .build());
        }
    }

    @Nested
    class UnitTests {

        @Test
        void testOutputsToValidate() {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("test")
                    .build();

            assertFalse(benchmark.outputsToValidate().isEmpty());
            assertTrue(benchmark.outputsToValidate().contains("attention_output"));
        }

        @Test
        void testPrepareInputs() throws Exception {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("test")
                    .hiddenSize(64)
                    .numHeads(4)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .batchSize(2)
                    .sequenceLength(16)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);

            assertTrue(inputs.containsKey("hidden_states"));
            var input = inputs.get("hidden_states");
            assertEquals(3, input.shape().length);
            assertEquals(2, input.shape()[0]); // batch
            assertEquals(16, input.shape()[1]); // seq
            assertEquals(64, input.shape()[2]); // hidden

            benchmark.teardown();
        }
    }

    @Nested
    class InferenceTests {

        private BenchmarkRunner runner;
        private ByteArrayOutputStream outputCapture;

        @BeforeEach
        void setUp() {
            outputCapture = new ByteArrayOutputStream();
            runner = new BenchmarkRunner()
                    .progressOutput(new PrintStream(outputCapture));
        }

        @Test
        void testStandardAttention() {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("standard-attn")
                    .hiddenSize(64)
                    .numHeads(4)
                    .causal(true)
                    .useFlashAttention(false)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .batchSize(2)
                    .sequenceLength(32)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
            assertTrue(result.meanLatencyMs() > 0);
        }

        @Test
        void testFlashAttention() {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("flash-attn")
                    .hiddenSize(64)
                    .numHeads(4)
                    .causal(true)
                    .useFlashAttention(true)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .batchSize(2)
                    .sequenceLength(32)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
        }

        @Test
        void testNonCausalAttention() {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("non-causal")
                    .hiddenSize(64)
                    .numHeads(4)
                    .causal(false)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .batchSize(2)
                    .sequenceLength(32)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
        }

        @Test
        void testOutputShape() throws Exception {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("shape-test")
                    .hiddenSize(128)
                    .numHeads(8)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .batchSize(4)
                    .sequenceLength(64)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            assertTrue(outputs.containsKey("attention_output"));
            var output = outputs.get("attention_output");

            // Shape should match input: [batch_size, seq_len, hidden_size]
            assertEquals(3, output.shape().length);
            assertEquals(4, output.shape()[0]); // batch_size
            assertEquals(64, output.shape()[1]); // seq_len
            assertEquals(128, output.shape()[2]); // hidden_size

            benchmark.teardown();
        }

        @Test
        void testOutputValues() throws Exception {
            AttentionBenchmark benchmark = AttentionBenchmark.builder("values-test")
                    .hiddenSize(64)
                    .numHeads(4)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .batchSize(1)
                    .sequenceLength(8)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            var output = outputs.get("attention_output");

            // Check that values are valid (not NaN or Inf)
            boolean hasValidValues = true;
            for (int i = 0; i < Math.min(100, output.info().elementCount()); i++) {
                float val = output.getFloatFlat(i);
                if (Float.isNaN(val) || Float.isInfinite(val)) {
                    hasValidValues = false;
                    break;
                }
            }

            assertTrue(hasValidValues, "Attention output should not contain NaN or Inf");

            benchmark.teardown();
        }

        @Test
        void testLargerModel() {
            // Test with BERT-base size attention
            AttentionBenchmark benchmark = AttentionBenchmark.builder("bert-base-attn")
                    .hiddenSize(768)
                    .numHeads(12)
                    .causal(false) // BERT uses bidirectional attention
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .batchSize(1)
                    .sequenceLength(64) // Keep sequence short for speed
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
        }
    }
}
