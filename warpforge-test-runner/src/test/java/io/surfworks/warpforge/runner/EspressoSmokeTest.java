package io.surfworks.warpforge.runner;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Mode 2b: Espresso Smoke Tests
 *
 * <p>These tests validate Espresso JAR loading using pre-built model JARs.
 * They run on GraalVM JVM with Espresso support, NOT Babylon JDK.
 *
 * <p>Prerequisites:
 * <ul>
 *   <li>GraalVM 25 with Espresso support must be installed</li>
 *   <li>Model JARs must be pre-generated via generateTestModelJars task</li>
 * </ul>
 *
 * <p>Run with: ./gradlew :warpforge-test-runner:espressoSmokeTest
 */
@Tag("espresso-smoke")
class EspressoSmokeTest {

    private static Path modelJarsDir;

    @BeforeAll
    static void setup() {
        String dirPath = System.getProperty("warpforge.test.modelJarsDir");
        if (dirPath == null || dirPath.isBlank()) {
            fail("System property 'warpforge.test.modelJarsDir' not set. " +
                 "Run: ./gradlew :warpforge-test-runner:espressoSmokeTest");
        }

        modelJarsDir = Path.of(dirPath);
        if (!Files.isDirectory(modelJarsDir)) {
            fail("Model JARs directory does not exist: " + modelJarsDir + ". " +
                 "Run: ./gradlew :warpforge-test-runner:generateTestModelJars");
        }

        System.out.println("Mode 2b: Espresso Smoke Test");
        System.out.println("Using model JARs from: " + modelJarsDir);
    }

    @Test
    void testEspressoIsAvailable() {
        boolean available = isEspressoAvailable();
        System.out.println("Espresso available: " + available);

        // Skip gracefully if Espresso is not available (e.g., Espresso not installed in GraalVM)
        // This allows local development without Espresso while CI can verify availability
        assumeTrue(available,
            "Espresso not available. To run Espresso tests, install Espresso in GraalVM. " +
            "See: https://www.graalvm.org/latest/reference-manual/espresso/");
    }

    @Test
    void testEspressoLoadAddModel() throws Exception {
        assumeEspressoAvailable();

        Path jarPath = modelJarsDir.resolve("addmodel.jar");
        assertTrue(Files.exists(jarPath), "AddModel JAR should exist: " + jarPath);

        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            jarPath,
            ExecutionMode.ESPRESSO,
            "cpu",
            new int[]{2, 2}
        );

        TestRunner.RunResult result = TestRunner.run(args);

        assertTrue(result.success(), "Espresso mode should succeed for AddModel");
        assertEquals(1, result.outputs().size(), "Should have 1 output");

        // With inputs [1,2,3,4] and [11,12,13,14], output should be [12,14,16,18]
        float[] expected = {12.0f, 14.0f, 16.0f, 18.0f};
        assertArrayEquals(expected, result.outputs().getFirst(), 0.001f,
            "Espresso AddModel output should match expected values");

        System.out.println("SUCCESS: Espresso loaded and executed AddModel");
        System.out.println("  Output: " + Arrays.toString(result.outputs().getFirst()));
    }

    @Test
    void testEspressoLoadMulAddModel() throws Exception {
        assumeEspressoAvailable();

        Path jarPath = modelJarsDir.resolve("muladdmodel.jar");
        assertTrue(Files.exists(jarPath), "MulAddModel JAR should exist: " + jarPath);

        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            jarPath,
            ExecutionMode.ESPRESSO,
            "cpu",
            new int[]{2, 2}
        );

        TestRunner.RunResult result = TestRunner.run(args);

        assertTrue(result.success(), "Espresso mode should succeed for MulAddModel");
        assertEquals(1, result.outputs().size(), "Should have 1 output");

        // With inputs [1,2,3,4], [11,12,13,14], [21,22,23,24]
        // multiply: [1*11, 2*12, 3*13, 4*14] = [11, 24, 39, 56]
        // add: [11+21, 24+22, 39+23, 56+24] = [32, 46, 62, 80]
        float[] expected = {32.0f, 46.0f, 62.0f, 80.0f};
        assertArrayEquals(expected, result.outputs().getFirst(), 0.001f,
            "Espresso MulAddModel output should match expected values");

        System.out.println("SUCCESS: Espresso loaded and executed MulAddModel");
        System.out.println("  Output: " + Arrays.toString(result.outputs().getFirst()));
    }

    @Test
    void testEspressoAndJvmProduceIdenticalOutputs() throws Exception {
        assumeEspressoAvailable();

        List<String> models = List.of("addmodel.jar", "muladdmodel.jar", "submodel.jar", "negatemodel.jar");

        for (String modelFile : models) {
            Path jarPath = modelJarsDir.resolve(modelFile);
            if (!Files.exists(jarPath)) {
                System.out.println("Skipping " + modelFile + " (not found)");
                continue;
            }

            // Determine shape based on model
            int[] shape = modelFile.contains("negate") ? new int[]{3, 3} :
                         modelFile.contains("sub") ? new int[]{4} :
                         new int[]{2, 2};

            // Run in JVM mode
            TestRunner.RunnerArgs jvmArgs = new TestRunner.RunnerArgs(
                jarPath, ExecutionMode.JVM, "cpu", shape);
            TestRunner.RunResult jvmResult = TestRunner.run(jvmArgs);

            // Run in Espresso mode
            TestRunner.RunnerArgs espressoArgs = new TestRunner.RunnerArgs(
                jarPath, ExecutionMode.ESPRESSO, "cpu", shape);
            TestRunner.RunResult espressoResult = TestRunner.run(espressoArgs);

            // Verify both succeeded
            assertTrue(jvmResult.success(), "JVM mode should succeed for " + modelFile);
            assertTrue(espressoResult.success(), "Espresso mode should succeed for " + modelFile);

            // Verify outputs are identical
            assertEquals(jvmResult.outputs().size(), espressoResult.outputs().size(),
                "Output count should match for " + modelFile);

            for (int i = 0; i < jvmResult.outputs().size(); i++) {
                assertArrayEquals(
                    jvmResult.outputs().get(i),
                    espressoResult.outputs().get(i),
                    0.0001f,
                    "Output " + i + " should be identical for " + modelFile
                );
            }

            System.out.println("SUCCESS: JVM and Espresso produce identical outputs for " + modelFile);
        }

        System.out.println("\nAll models verified: JVM and Espresso modes are consistent!");
    }

    /**
     * Check if Espresso is available in the current runtime.
     */
    private static boolean isEspressoAvailable() {
        try {
            Class.forName("org.graalvm.polyglot.Context");
            org.graalvm.polyglot.Context context = org.graalvm.polyglot.Context.newBuilder("java")
                .allowAllAccess(true)
                .option("java.Classpath", System.getProperty("java.class.path"))
                .build();
            context.close();
            return true;
        } catch (Exception | Error e) {
            System.out.println("Espresso not available: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return false;
        }
    }

    private void assumeEspressoAvailable() {
        assumeTrue(isEspressoAvailable(),
            "Espresso not available - this test requires GraalVM JDK with Espresso support");
    }
}
