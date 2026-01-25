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

class TransformerBlockBenchmarkTest {

    @Nested
    class BuilderTests {

        @Test
        void testBuilderDefaults() {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("test")
                    .build();

            assertEquals("test", benchmark.name());
            assertEquals("synthetic-transformer-block", benchmark.modelId());
        }

        @Test
        void testBuilderWithCustomConfig() {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("custom")
                    .hiddenSize(1024)
                    .intermediateSize(4096)
                    .numHeads(16)
                    .preNorm(false)
                    .causal(false)
                    .layerNormEps(1e-6f)
                    .build();

            assertEquals("custom", benchmark.name());
        }

        @Test
        void testBuilderValidatesHiddenSizeDivisibility() {
            assertThrows(IllegalArgumentException.class, () ->
                    TransformerBlockBenchmark.builder("invalid")
                            .hiddenSize(768)
                            .numHeads(11)
                            .build());
        }

        @Test
        void testGPT2BaseConfig() {
            // GPT-2 base: 768 hidden, 3072 intermediate, 12 heads
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("gpt2-base")
                    .hiddenSize(768)
                    .intermediateSize(3072)
                    .numHeads(12)
                    .preNorm(true)
                    .causal(true)
                    .build();

            assertEquals("gpt2-base", benchmark.name());
        }

        @Test
        void testBertBaseConfig() {
            // BERT base: 768 hidden, 3072 intermediate, 12 heads, post-norm, bidirectional
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("bert-base")
                    .hiddenSize(768)
                    .intermediateSize(3072)
                    .numHeads(12)
                    .preNorm(false)
                    .causal(false)
                    .build();

            assertEquals("bert-base", benchmark.name());
        }
    }

    @Nested
    class UnitTests {

        @Test
        void testOutputsToValidate() {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("test")
                    .build();

            assertFalse(benchmark.outputsToValidate().isEmpty());
            assertTrue(benchmark.outputsToValidate().contains("hidden_states"));
        }

        @Test
        void testPrepareInputs() throws Exception {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("test")
                    .hiddenSize(64)
                    .intermediateSize(256)
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
            assertEquals(2, input.shape()[0]);
            assertEquals(16, input.shape()[1]);
            assertEquals(64, input.shape()[2]);

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
        void testPreNormTransformer() {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("pre-norm")
                    .hiddenSize(64)
                    .intermediateSize(256)
                    .numHeads(4)
                    .preNorm(true)
                    .causal(true)
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
        void testPostNormTransformer() {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("post-norm")
                    .hiddenSize(64)
                    .intermediateSize(256)
                    .numHeads(4)
                    .preNorm(false)
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
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("shape-test")
                    .hiddenSize(128)
                    .intermediateSize(512)
                    .numHeads(8)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .batchSize(4)
                    .sequenceLength(64)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            assertTrue(outputs.containsKey("hidden_states"));
            var output = outputs.get("hidden_states");

            assertEquals(3, output.shape().length);
            assertEquals(4, output.shape()[0]);
            assertEquals(64, output.shape()[1]);
            assertEquals(128, output.shape()[2]);

            benchmark.teardown();
        }

        @Test
        void testOutputValues() throws Exception {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("values-test")
                    .hiddenSize(64)
                    .intermediateSize(256)
                    .numHeads(4)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .batchSize(1)
                    .sequenceLength(8)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            var output = outputs.get("hidden_states");

            boolean hasValidValues = true;
            for (int i = 0; i < Math.min(100, output.info().elementCount()); i++) {
                float val = output.getFloatFlat(i);
                if (Float.isNaN(val) || Float.isInfinite(val)) {
                    hasValidValues = false;
                    break;
                }
            }

            assertTrue(hasValidValues, "Transformer output should not contain NaN or Inf");

            benchmark.teardown();
        }

        @Test
        void testLargerModel() {
            // Test with GPT-2 small size
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("gpt2-small")
                    .hiddenSize(768)
                    .intermediateSize(3072)
                    .numHeads(12)
                    .preNorm(true)
                    .causal(true)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .batchSize(1)
                    .sequenceLength(32)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
        }

        @Test
        void testBidirectionalAttention() {
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("bidirectional")
                    .hiddenSize(64)
                    .intermediateSize(256)
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
        void testResidualConnections() throws Exception {
            // Verify residual connections work by checking output is not zero
            TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("residual-test")
                    .hiddenSize(64)
                    .intermediateSize(256)
                    .numHeads(4)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .batchSize(1)
                    .sequenceLength(4)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            var output = outputs.get("hidden_states");

            // Output should have non-zero values (residual + attention + mlp)
            boolean hasNonZero = false;
            for (int i = 0; i < output.info().elementCount(); i++) {
                if (Math.abs(output.getFloatFlat(i)) > 1e-6f) {
                    hasNonZero = true;
                    break;
                }
            }

            assertTrue(hasNonZero, "Transformer output should have non-zero values");

            benchmark.teardown();
        }
    }
}
