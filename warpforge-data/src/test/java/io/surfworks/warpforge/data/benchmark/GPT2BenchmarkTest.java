package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GPT2BenchmarkTest {

    @Nested
    class BuilderTests {

        @Test
        void testBuilderDefaults() {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("test")
                    .build();

            assertEquals("test", benchmark.name());
            assertEquals("hf-internal-testing/tiny-random-gpt2", benchmark.modelId());
        }

        @Test
        void testBuilderWithCustomModel() {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("custom")
                    .modelId("gpt2")
                    .build();

            assertEquals("gpt2", benchmark.modelId());
        }
    }

    @Nested
    class UnitTests {

        @Test
        void testOutputsToValidate() {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("test")
                    .build();

            assertFalse(benchmark.outputsToValidate().isEmpty());
            assertTrue(benchmark.outputsToValidate().contains("hidden_states"));
        }
    }

    @Nested
    @Tag("integration")
    class IntegrationTests {

        private BenchmarkRunner runner;
        private ByteArrayOutputStream outputCapture;

        @BeforeEach
        void setUp() {
            outputCapture = new ByteArrayOutputStream();
            runner = new BenchmarkRunner()
                    .progressOutput(new PrintStream(outputCapture))
                    .verbose(true);
        }

        @Test
        void testTinyGPT2Inference() {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("tiny-gpt2")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-gpt2")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .batchSize(1)
                    .sequenceLength(16)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
            assertTrue(result.meanLatencyMs() > 0);
        }

        @Test
        void testBatchedGPT2() {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("batched-gpt2")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-gpt2")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .batchSize(4)
                    .sequenceLength(32)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
        }

        @Test
        void testGPT2OutputShape() throws Exception {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("shape-test")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-gpt2")
                    .batchSize(2)
                    .sequenceLength(8)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            assertTrue(outputs.containsKey("hidden_states"));
            var hiddenStates = outputs.get("hidden_states");

            // Shape should be [batch_size, seq_len, hidden_size]
            assertEquals(3, hiddenStates.shape().length);
            assertEquals(2, hiddenStates.shape()[0]); // batch_size
            assertEquals(8, hiddenStates.shape()[1]); // seq_len

            benchmark.teardown();
        }

        @Test
        void testGPT2OutputValues() throws Exception {
            GPT2Benchmark benchmark = GPT2Benchmark.builder("values-test")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-gpt2")
                    .batchSize(1)
                    .sequenceLength(4)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            var hiddenStates = outputs.get("hidden_states");

            // Check that values are valid (not NaN or Inf)
            boolean hasValidValues = true;
            for (int i = 0; i < Math.min(100, hiddenStates.info().elementCount()); i++) {
                float val = hiddenStates.getFloatFlat(i);
                if (Float.isNaN(val) || Float.isInfinite(val)) {
                    hasValidValues = false;
                    break;
                }
            }

            assertTrue(hasValidValues, "Hidden states should not contain NaN or Inf");

            benchmark.teardown();
        }
    }
}
