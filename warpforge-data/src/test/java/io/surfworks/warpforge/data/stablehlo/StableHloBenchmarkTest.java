package io.surfworks.warpforge.data.stablehlo;

import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.benchmark.BenchmarkConfig;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StableHloBenchmarkTest {

    @TempDir
    Path tempDir;

    private static final String SIMPLE_ADD_MLIR = """
            module @add_test {
              func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
                stablehlo.return %0 : tensor<4x8xf32>
              }
            }
            """;

    private static final String MATMUL_MLIR = """
            module @matmul_test {
              func.func public @forward(%input: tensor<1x768xf32>, %weight: tensor<768x768xf32>) -> (tensor<1x768xf32>) {
                %0 = stablehlo.dot %input, %weight : tensor<1x768xf32>
                stablehlo.return %0 : tensor<1x768xf32>
              }
            }
            """;

    @Nested
    class BuilderTests {

        @Test
        void testBuilderWithMlirContent() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test-add")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .build();

            assertEquals("test-add", benchmark.name());
            assertEquals("inline-mlir", benchmark.modelId());
        }

        @Test
        void testBuilderWithMlirFile() throws IOException {
            Path mlirFile = tempDir.resolve("add.mlir");
            Files.writeString(mlirFile, SIMPLE_ADD_MLIR);

            StableHloBenchmark benchmark = StableHloBenchmark.builder("test-file")
                    .mlirFile(mlirFile)
                    .build();

            assertEquals("test-file", benchmark.name());
            assertTrue(benchmark.modelId().contains("add.mlir"));
        }

        @Test
        void testBuilderValidation() {
            assertThrows(IllegalStateException.class, () ->
                    StableHloBenchmark.builder("invalid").build()
            );
        }

        @Test
        void testBuilderOptions() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .randomInputs(false)
                    .seed(12345)
                    .build();

            assertNotNull(benchmark);
        }
    }

    @Nested
    class SetupTests {

        @Test
        void testSetupFromContent() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(4)
                    .build();

            benchmark.setup(config);

            assertNotNull(benchmark.module());
            assertEquals("add_test", benchmark.module().name());

            benchmark.teardown();
        }

        @Test
        void testSetupFromFile() throws IOException {
            Path mlirFile = tempDir.resolve("test.mlir");
            Files.writeString(mlirFile, MATMUL_MLIR);

            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirFile(mlirFile)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(1)
                    .build();

            benchmark.setup(config);

            assertNotNull(benchmark.module());
            assertEquals("matmul_test", benchmark.module().name());

            benchmark.teardown();
        }
    }

    @Nested
    class PrepareInputsTests {

        @Test
        void testPrepareInputsSimple() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(4)
                    .build();

            benchmark.setup(config);
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);

            assertEquals(2, inputs.size());
            assertTrue(inputs.containsKey("arg0"));
            assertTrue(inputs.containsKey("arg1"));

            TensorView arg0 = inputs.get("arg0");
            assertEquals(2, arg0.shape().length);
            assertEquals(4, arg0.shape()[0]); // batch size
            assertEquals(8, arg0.shape()[1]);

            benchmark.teardown();
        }

        @Test
        void testPrepareInputsWithDifferentBatchSize() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(MATMUL_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(8)
                    .build();

            benchmark.setup(config);
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);

            TensorView input = inputs.get("input");
            assertEquals(8, input.shape()[0]); // batch dimension adjusted

            benchmark.teardown();
        }

        @Test
        void testPrepareInputsRandomVsDeterministic() throws IOException {
            // Random inputs
            StableHloBenchmark randomBenchmark = StableHloBenchmark.builder("random")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .randomInputs(true)
                    .seed(42)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(1)
                    .build();

            randomBenchmark.setup(config);
            Map<String, TensorView> randomInputs = randomBenchmark.prepareInputs(config);
            TensorView randomArg0 = randomInputs.get("arg0");

            // Check that random inputs have non-zero values
            boolean hasNonZero = false;
            for (int i = 0; i < 8; i++) {
                if (randomArg0.getFloatFlat(i) != 0.0f) {
                    hasNonZero = true;
                    break;
                }
            }
            assertTrue(hasNonZero, "Random inputs should have non-zero values");

            randomBenchmark.teardown();

            // Zero inputs
            StableHloBenchmark zeroBenchmark = StableHloBenchmark.builder("zero")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .randomInputs(false)
                    .build();

            zeroBenchmark.setup(config);
            Map<String, TensorView> zeroInputs = zeroBenchmark.prepareInputs(config);
            TensorView zeroArg0 = zeroInputs.get("arg0");

            // All zeros
            for (int i = 0; i < 8; i++) {
                assertEquals(0.0f, zeroArg0.getFloatFlat(i), 1e-6f);
            }

            zeroBenchmark.teardown();
        }
    }

    @Nested
    class InferenceTests {

        @Test
        void testRunInference() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(4)
                    .build();

            benchmark.setup(config);
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);
            Map<String, TensorView> outputs = benchmark.runInference(inputs);

            assertFalse(outputs.isEmpty());
            assertTrue(outputs.containsKey("output_0"));

            TensorView output = outputs.get("output_0");
            assertEquals(4, output.shape()[0]); // batch dimension matches input

            benchmark.teardown();
        }

        @Test
        void testOutputsToValidate() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(1)
                    .build();

            benchmark.setup(config);

            var outputs = benchmark.outputsToValidate();
            assertEquals(1, outputs.size());
            assertEquals("output_0", outputs.get(0));

            benchmark.teardown();
        }
    }

    @Nested
    class MlirExportTests {

        @Test
        void testToMlir() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(SIMPLE_ADD_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(1)
                    .build();

            benchmark.setup(config);

            String mlir = benchmark.toMlir();
            assertFalse(mlir.isEmpty());
            assertTrue(mlir.contains("module @add_test"));

            benchmark.teardown();
        }
    }

    @Nested
    class MultiOutputTests {

        private static final String MULTI_OUTPUT_MLIR = """
                module @multi_output {
                  func.func public @forward(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.abs %x : tensor<4xf32>
                    %1 = stablehlo.negate %x : tensor<4xf32>
                    stablehlo.return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
                """;

        @Test
        void testMultipleOutputs() throws IOException {
            StableHloBenchmark benchmark = StableHloBenchmark.builder("test")
                    .mlirContent(MULTI_OUTPUT_MLIR)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("test")
                    .batchSize(4)
                    .build();

            benchmark.setup(config);
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);
            Map<String, TensorView> outputs = benchmark.runInference(inputs);

            assertEquals(2, outputs.size());
            assertTrue(outputs.containsKey("output_0"));
            assertTrue(outputs.containsKey("output_1"));

            var outputNames = benchmark.outputsToValidate();
            assertEquals(2, outputNames.size());

            benchmark.teardown();
        }
    }
}
