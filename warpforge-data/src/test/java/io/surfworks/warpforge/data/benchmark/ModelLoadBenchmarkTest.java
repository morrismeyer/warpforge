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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelLoadBenchmarkTest {

    @TempDir
    Path tempDir;

    @Nested
    class BuilderTests {

        @Test
        void testBuilderRequiresModelId() {
            assertThrows(IllegalStateException.class, () ->
                    ModelLoadBenchmark.builder("test").build());
        }

        @Test
        void testBuilderWithModelId() {
            ModelLoadBenchmark benchmark = ModelLoadBenchmark.builder("test")
                    .modelId("gpt2")
                    .build();

            assertEquals("test", benchmark.name());
            assertEquals("gpt2", benchmark.modelId());
        }

        @Test
        void testBuilderWithAllOptions() {
            ModelLoadBenchmark benchmark = ModelLoadBenchmark.builder("full-test")
                    .modelId("bert-base")
                    .accessAllTensors(true)
                    .warmupTensorAccess(true)
                    .cacheDir(tempDir)
                    .build();

            assertEquals("full-test", benchmark.name());
            assertEquals("bert-base", benchmark.modelId());
        }
    }

    @Nested
    class UnitTests {

        @Test
        void testOutputsToValidateIsEmpty() {
            ModelLoadBenchmark benchmark = ModelLoadBenchmark.builder("test")
                    .modelId("test-model")
                    .build();

            assertTrue(benchmark.outputsToValidate().isEmpty());
        }

        @Test
        void testPrepareInputsReturnsEmptyMap() {
            ModelLoadBenchmark benchmark = ModelLoadBenchmark.builder("test")
                    .modelId("test-model")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.defaults("test");
            assertTrue(benchmark.prepareInputs(config).isEmpty());
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
                    .progressOutput(new PrintStream(outputCapture));
        }

        @Test
        void testLoadTinyModel() throws Exception {
            ModelLoadBenchmark benchmark = ModelLoadBenchmark.builder("tiny-gpt2-load")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .accessAllTensors(true)
                    .cacheDir(tempDir)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-gpt2")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status());
            // Note: getLoadedModel() may be null after runner.run() completes
            // because teardown() is called. We verify the result status instead.
            assertTrue(result.latenciesNanos().length > 0);
        }

        @Test
        void testLoadWithCaching() {
            Path cacheDir = tempDir.resolve("cache");

            // First load - downloads
            ModelLoadBenchmark benchmark1 = ModelLoadBenchmark.builder("first-load")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .cacheDir(cacheDir)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-gpt2")
                    .warmupIterations(0)
                    .measurementIterations(1)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result1 = runner.run(benchmark1, config);
            benchmark1.teardown();

            // Second load - from cache (should be faster)
            ModelLoadBenchmark benchmark2 = ModelLoadBenchmark.builder("cached-load")
                    .modelId("hf-internal-testing/tiny-random-gpt2")
                    .cacheDir(cacheDir)
                    .build();

            BenchmarkResult result2 = runner.run(benchmark2, config);
            benchmark2.teardown();

            assertEquals(BenchmarkResult.Status.SUCCESS, result1.status());
            assertEquals(BenchmarkResult.Status.SUCCESS, result2.status());

            // Cached load should generally be faster, but we don't enforce this
            // as it depends on system state
        }
    }
}
