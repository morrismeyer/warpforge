package io.surfworks.warpforge.runner;

import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelMetadata;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for TestRunner.
 *
 * <p>Tests the RunnerArgs and RunResult records, as well as basic
 * functionality that doesn't require pre-generated model JARs.
 *
 * <p>For full integration tests that use actual model JARs, see
 * ExecutionModeConsistencyTest and EspressoSmokeTest which run
 * via the integrationTest and espressoSmokeTest tasks.
 */
class TestRunnerUnitTest {

    // ========================
    // RunnerArgs record tests
    // ========================

    @Test
    void testRunnerArgsRecord() {
        Path jar = Path.of("test.jar");
        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            jar,
            ExecutionMode.JVM,
            "cpu",
            new int[]{2, 2}
        );

        assertEquals(jar, args.jarPath());
        assertEquals(ExecutionMode.JVM, args.mode());
        assertEquals("cpu", args.backend());
        assertArrayEquals(new int[]{2, 2}, args.inputShape());
    }

    @Test
    void testRunnerArgsWithNullShape() {
        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            Path.of("test.jar"),
            ExecutionMode.JVM,
            "cpu",
            null  // null shape should be allowed
        );

        assertTrue(args.inputShape() == null);
    }

    @Test
    void testRunnerArgsAllModes() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
                Path.of("test.jar"),
                mode,
                "cpu",
                new int[]{1}
            );
            assertEquals(mode, args.mode());
        }
    }

    @Test
    void testRunnerArgsAllBackends() {
        String[] backends = {"cpu", "nvidia", "amd"};
        for (String backend : backends) {
            TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
                Path.of("test.jar"),
                ExecutionMode.JVM,
                backend,
                new int[]{1}
            );
            assertEquals(backend, args.backend());
        }
    }

    @Test
    void testRunnerArgsVariousShapes() {
        int[][] shapes = {
            new int[]{1},
            new int[]{2, 2},
            new int[]{3, 4, 5},
            new int[]{2, 3, 4, 5}
        };

        for (int[] shape : shapes) {
            TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
                Path.of("test.jar"),
                ExecutionMode.JVM,
                "cpu",
                shape
            );
            assertArrayEquals(shape, args.inputShape());
        }
    }

    // ========================
    // RunResult record tests
    // ========================

    @Test
    void testRunResultSuccess() {
        List<float[]> outputs = List.of(new float[]{1, 2, 3});
        TestRunner.RunResult result = new TestRunner.RunResult(true, outputs, 100.0);

        assertTrue(result.success());
        assertEquals(1, result.outputs().size());
        assertArrayEquals(new float[]{1, 2, 3}, result.outputs().get(0));
        assertEquals(100.0, result.durationMs(), 0.001);
    }

    @Test
    void testRunResultFailure() {
        TestRunner.RunResult result = new TestRunner.RunResult(false, List.of(), 0.0);

        assertFalse(result.success());
        assertTrue(result.outputs().isEmpty());
    }

    @Test
    void testRunResultMultipleOutputs() {
        List<float[]> outputs = List.of(
            new float[]{1, 2},
            new float[]{3, 4, 5},
            new float[]{6}
        );
        TestRunner.RunResult result = new TestRunner.RunResult(true, outputs, 50.0);

        assertEquals(3, result.outputs().size());
        assertArrayEquals(new float[]{1, 2}, result.outputs().get(0));
        assertArrayEquals(new float[]{3, 4, 5}, result.outputs().get(1));
        assertArrayEquals(new float[]{6}, result.outputs().get(2));
    }

    @Test
    void testRunResultZeroDuration() {
        TestRunner.RunResult result = new TestRunner.RunResult(true, List.of(), 0.0);

        assertTrue(result.success());
        assertEquals(0.0, result.durationMs(), 0.001);
    }

    @Test
    void testRunResultLargeDuration() {
        TestRunner.RunResult result = new TestRunner.RunResult(true, List.of(), 60000.0);

        assertEquals(60000.0, result.durationMs(), 0.001);
    }

    // ========================
    // Native mode with registry tests
    // ========================

    @Test
    void testRunNativeModeNotRegistered() {
        // Use a name that we know won't be in the registry
        String uniqueName = "nonexistent_" + System.currentTimeMillis() + ".jar";
        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            Path.of(uniqueName),
            ExecutionMode.NATIVE,
            "cpu",
            new int[]{2, 2}
        );

        // Native mode should throw when model not found in registry
        assertThrows(Exception.class, () -> TestRunner.run(args));
    }

    @Test
    void testRunNativeModeWithRegisteredModel() throws Exception {
        // Create and register a mock model
        String modelName = "native_unit_test_" + System.currentTimeMillis();
        CompiledModel mockModel = createMockModel(modelName);

        NativeModelRegistry.register(modelName, mockModel);

        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            Path.of(modelName + ".jar"),  // Name derived from JAR
            ExecutionMode.NATIVE,
            "cpu",
            new int[]{2, 2}
        );

        TestRunner.RunResult result = TestRunner.run(args);

        assertTrue(result.success());
        assertEquals(1, result.outputs().size());
    }

    // ========================
    // Error handling tests
    // ========================

    @Test
    void testRunWithNonexistentJar() {
        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            Path.of("definitely_nonexistent_" + System.currentTimeMillis() + ".jar"),
            ExecutionMode.JVM,
            "cpu",
            new int[]{2, 2}
        );

        assertThrows(Exception.class, () -> TestRunner.run(args));
    }

    @Test
    void testRunWithInvalidBackend() {
        // Even with a nonexistent JAR, invalid backend should fail first for native mode
        String modelName = "backend_test_" + System.currentTimeMillis();
        NativeModelRegistry.register(modelName, createMockModel(modelName));

        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            Path.of(modelName + ".jar"),
            ExecutionMode.NATIVE,
            "invalid_backend_xyz",
            new int[]{2, 2}
        );

        assertThrows(IllegalArgumentException.class, () -> TestRunner.run(args));
    }

    // ========================
    // Helper methods
    // ========================

    /**
     * Creates a mock CompiledModel for testing.
     */
    private CompiledModel createMockModel(String name) {
        return new CompiledModel() {
            @Override
            public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
                // Create a simple output tensor
                Tensor output = Tensor.zeros(new int[]{2, 2});
                float[] data = new float[4];
                for (int i = 0; i < inputs.size(); i++) {
                    float[] inputData = inputs.get(i).toFloatArray();
                    for (int j = 0; j < Math.min(inputData.length, data.length); j++) {
                        data[j] += inputData[j];
                    }
                }
                output.copyFrom(data);
                return List.of(output);
            }

            @Override
            public int inputCount() {
                return 2;
            }

            @Override
            public int outputCount() {
                return 1;
            }

            @Override
            public ModelMetadata metadata() {
                return new ModelMetadata(name, "mockhash", System.currentTimeMillis(), "1.0.0-test");
            }
        };
    }
}
