package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ViTBenchmarkTest {

    @Nested
    class BuilderTests {

        @Test
        void testBuilderDefaults() {
            ViTBenchmark benchmark = ViTBenchmark.builder("test")
                    .build();

            assertEquals("test", benchmark.name());
            assertEquals("hf-internal-testing/tiny-random-vit", benchmark.modelId());
        }

        @Test
        void testBuilderWithCustomConfig() {
            ViTBenchmark benchmark = ViTBenchmark.builder("custom")
                    .modelId("google/vit-base-patch16-224")
                    .imageSize(384)
                    .patchSize(16)
                    .numChannels(3)
                    .build();

            assertEquals("google/vit-base-patch16-224", benchmark.modelId());
        }

        @Test
        void testBuilderValidatesImagePatchDivisibility() {
            assertThrows(IllegalArgumentException.class, () ->
                    ViTBenchmark.builder("invalid")
                            .imageSize(224)
                            .patchSize(17) // 224 not divisible by 17
                            .build());
        }

        @Test
        void testViTBaseConfig() {
            ViTBenchmark benchmark = ViTBenchmark.builder("vit-base")
                    .modelId("google/vit-base-patch16-224")
                    .imageSize(224)
                    .patchSize(16)
                    .build();

            assertEquals("vit-base", benchmark.name());
        }

        @Test
        void testViTLargeConfig() {
            ViTBenchmark benchmark = ViTBenchmark.builder("vit-large")
                    .modelId("google/vit-large-patch16-224")
                    .imageSize(224)
                    .patchSize(16)
                    .build();

            assertEquals("vit-large", benchmark.name());
        }
    }

    @Nested
    class UnitTests {

        @Test
        void testOutputsToValidate() {
            ViTBenchmark benchmark = ViTBenchmark.builder("test")
                    .build();

            assertFalse(benchmark.outputsToValidate().isEmpty());
            assertTrue(benchmark.outputsToValidate().contains("logits"));
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
        void testTinyViTInference() {
            ViTBenchmark benchmark = ViTBenchmark.builder("tiny-vit")
                    .modelId("hf-internal-testing/tiny-random-vit")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-vit")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .batchSize(1)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
            assertTrue(result.meanLatencyMs() > 0);
        }

        @Test
        void testBatchedViT() {
            ViTBenchmark benchmark = ViTBenchmark.builder("batched-vit")
                    .modelId("hf-internal-testing/tiny-random-vit")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-vit")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .batchSize(4)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status(),
                    "Benchmark failed: " + result.errorMessage());
        }

        @Test
        void testViTOutputShape() throws Exception {
            ViTBenchmark benchmark = ViTBenchmark.builder("shape-test")
                    .modelId("hf-internal-testing/tiny-random-vit")
                    .imageSize(32) // Tiny for test
                    .patchSize(8)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-vit")
                    .batchSize(2)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            assertTrue(outputs.containsKey("logits"));
            var logits = outputs.get("logits");

            // Shape should be [batch_size, num_classes]
            assertEquals(2, logits.shape().length);
            assertEquals(2, logits.shape()[0]); // batch_size

            benchmark.teardown();
        }

        @Test
        void testViTOutputValues() throws Exception {
            ViTBenchmark benchmark = ViTBenchmark.builder("values-test")
                    .modelId("hf-internal-testing/tiny-random-vit")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-vit")
                    .batchSize(1)
                    .build();

            benchmark.setup(config);
            var inputs = benchmark.prepareInputs(config);
            var outputs = benchmark.runInference(inputs);

            var logits = outputs.get("logits");

            // Check that values are valid
            boolean hasValidValues = true;
            for (int i = 0; i < Math.min(100, logits.info().elementCount()); i++) {
                float val = logits.getFloatFlat(i);
                if (Float.isNaN(val) || Float.isInfinite(val)) {
                    hasValidValues = false;
                    break;
                }
            }

            assertTrue(hasValidValues, "Logits should not contain NaN or Inf");

            benchmark.teardown();
        }
    }
}
