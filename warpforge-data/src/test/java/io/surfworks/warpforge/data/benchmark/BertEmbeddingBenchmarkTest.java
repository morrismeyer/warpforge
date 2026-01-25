package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BertEmbeddingBenchmarkTest {

    @TempDir
    Path tempDir;

    @Nested
    class BuilderTests {

        @Test
        void testBuilderDefaults() {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("test")
                    .build();

            assertEquals("test", benchmark.name());
            assertEquals("hf-internal-testing/tiny-random-bert", benchmark.modelId());
        }

        @Test
        void testBuilderWithCustomModel() {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("custom")
                    .modelId("bert-base-uncased")
                    .build();

            assertEquals("bert-base-uncased", benchmark.modelId());
        }
    }

    @Nested
    class UnitTests {

        @Test
        void testOutputsToValidate() {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("test")
                    .build();

            assertFalse(benchmark.outputsToValidate().isEmpty());
            assertTrue(benchmark.outputsToValidate().contains("embeddings"));
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
        void testTinyBertEmbedding() {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("tiny-bert-embed")
                    .modelId("hf-internal-testing/tiny-random-bert")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-bert")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .batchSize(1)
                    .sequenceLength(16) // Short sequence for fast testing
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
            assertTrue(result.meanLatencyMs() > 0);
        }

        @Test
        void testBatchedEmbedding() {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("batched-bert-embed")
                    .modelId("hf-internal-testing/tiny-random-bert")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-bert")
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
        void testEmbeddingOutputShape() throws Exception {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("shape-test")
                    .modelId("hf-internal-testing/tiny-random-bert")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-bert")
                    .batchSize(2)
                    .sequenceLength(8)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            assertTrue(outputs.containsKey("embeddings"));
            var embeddings = outputs.get("embeddings");

            // Shape should be [batch_size, seq_len, hidden_size]
            assertEquals(3, embeddings.shape().length);
            assertEquals(2, embeddings.shape()[0]); // batch_size
            assertEquals(8, embeddings.shape()[1]); // seq_len
            // hidden_size depends on model config

            benchmark.teardown();
        }

        @Test
        void testEmbeddingValues() throws Exception {
            BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("values-test")
                    .modelId("hf-internal-testing/tiny-random-bert")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-bert")
                    .batchSize(1)
                    .sequenceLength(4)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            var embeddings = outputs.get("embeddings");

            // Check that values are normalized (not NaN or Inf)
            boolean hasValidValues = true;
            for (int i = 0; i < Math.min(100, embeddings.info().elementCount()); i++) {
                float val = embeddings.getFloatFlat(i);
                if (Float.isNaN(val) || Float.isInfinite(val)) {
                    hasValidValues = false;
                    break;
                }
            }

            assertTrue(hasValidValues, "Embedding values should not contain NaN or Inf");

            benchmark.teardown();
        }
    }
}
