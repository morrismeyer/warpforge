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

/**
 * Integration test that verifies all execution modes produce identical outputs.
 *
 * <p>This test uses pre-generated model JARs from snakeburger-codegen:generateTestModelJars.
 * The JARs are generated on Babylon JDK with Java 25 bytecode, allowing this test to run
 * on any JDK 25+ including GraalVM for Espresso testing.
 *
 * <p>Test flow:
 * <ol>
 *   <li>Loads pre-generated model JARs from configured directory</li>
 *   <li>Executes each model in JVM mode</li>
 *   <li>Executes each model in Espresso mode (if available)</li>
 *   <li>Verifies outputs are identical across modes</li>
 * </ol>
 *
 * <p>Native mode is stubbed as it requires AOT compilation of the model itself.
 */
@Tag("integration")
class ExecutionModeConsistencyTest {

    private static Path modelJarsDir;
    private static Path addModelJar;
    private static Path muladdModelJar;

    @BeforeAll
    static void setup() {
        String dirPath = System.getProperty("warpforge.test.modelJarsDir");
        if (dirPath == null || dirPath.isBlank()) {
            fail("System property 'warpforge.test.modelJarsDir' not set. " +
                 "Run: ./gradlew :warpforge-test-runner:integrationTest");
        }

        modelJarsDir = Path.of(dirPath);
        if (!Files.isDirectory(modelJarsDir)) {
            fail("Model JARs directory does not exist: " + modelJarsDir + ". " +
                 "Run: ./gradlew :snakeburger-codegen:generateTestModelJars");
        }

        // Locate pre-generated JARs
        addModelJar = modelJarsDir.resolve("addmodel.jar");
        muladdModelJar = modelJarsDir.resolve("muladdmodel.jar");

        assertTrue(Files.exists(addModelJar), "AddModel JAR should exist: " + addModelJar);
        assertTrue(Files.exists(muladdModelJar), "MulAddModel JAR should exist: " + muladdModelJar);

        System.out.println("Integration Test: Execution Mode Consistency");
        System.out.println("Using model JARs from: " + modelJarsDir);
    }

    @Test
    void testJvmModeAddModel() throws Exception {
        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            addModelJar,
            ExecutionMode.JVM,
            "cpu",
            new int[]{2, 2}
        );

        TestRunner.RunResult result = TestRunner.run(args);

        assertTrue(result.success(), "JVM mode should succeed");
        assertEquals(1, result.outputs().size(), "Should have 1 output");

        // With inputs [1,2,3,4] and [11,12,13,14], output should be [12,14,16,18]
        float[] expected = {12.0f, 14.0f, 16.0f, 18.0f};
        assertArrayEquals(expected, result.outputs().getFirst(), 0.001f,
            "JVM mode output should match expected values");
    }

    @Test
    void testJvmModeMulAddModel() throws Exception {
        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            muladdModelJar,
            ExecutionMode.JVM,
            "cpu",
            new int[]{2, 2}
        );

        TestRunner.RunResult result = TestRunner.run(args);

        assertTrue(result.success(), "JVM mode should succeed");
        assertEquals(1, result.outputs().size(), "Should have 1 output");

        // With inputs [1,2,3,4], [11,12,13,14], [21,22,23,24]
        // multiply: [1*11, 2*12, 3*13, 4*14] = [11, 24, 39, 56]
        // add: [11+21, 24+22, 39+23, 56+24] = [32, 46, 62, 80]
        float[] expected = {32.0f, 46.0f, 62.0f, 80.0f};
        assertArrayEquals(expected, result.outputs().getFirst(), 0.001f,
            "JVM mode output should match expected values");
    }

    @Test
    void testEspressoModeAddModel() throws Exception {
        // Skip if Espresso is not available (e.g., not running on GraalVM)
        if (!isEspressoAvailable()) {
            System.out.println("Skipping Espresso test: Espresso not available");
            return;
        }

        TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
            addModelJar,
            ExecutionMode.ESPRESSO,
            "cpu",
            new int[]{2, 2}
        );

        TestRunner.RunResult result = TestRunner.run(args);

        assertTrue(result.success(), "Espresso mode should succeed");
        assertEquals(1, result.outputs().size(), "Should have 1 output");

        // Should match JVM mode exactly
        float[] expected = {12.0f, 14.0f, 16.0f, 18.0f};
        assertArrayEquals(expected, result.outputs().getFirst(), 0.001f,
            "Espresso mode output should match expected values");
    }

    @Test
    void testJvmAndEspressoModesProduceIdenticalOutputs() throws Exception {
        // Skip if Espresso is not available
        if (!isEspressoAvailable()) {
            System.out.println("Skipping consistency test: Espresso not available");
            return;
        }

        // Run in JVM mode
        TestRunner.RunnerArgs jvmArgs = new TestRunner.RunnerArgs(
            addModelJar,
            ExecutionMode.JVM,
            "cpu",
            new int[]{2, 2}
        );
        TestRunner.RunResult jvmResult = TestRunner.run(jvmArgs);

        // Run in Espresso mode
        TestRunner.RunnerArgs espressoArgs = new TestRunner.RunnerArgs(
            addModelJar,
            ExecutionMode.ESPRESSO,
            "cpu",
            new int[]{2, 2}
        );
        TestRunner.RunResult espressoResult = TestRunner.run(espressoArgs);

        // Verify both succeeded
        assertTrue(jvmResult.success(), "JVM mode should succeed");
        assertTrue(espressoResult.success(), "Espresso mode should succeed");

        // Verify output counts match
        assertEquals(jvmResult.outputs().size(), espressoResult.outputs().size(),
            "Output counts should match between modes");

        // Verify all outputs are identical
        for (int i = 0; i < jvmResult.outputs().size(); i++) {
            float[] jvmOutput = jvmResult.outputs().get(i);
            float[] espressoOutput = espressoResult.outputs().get(i);

            assertArrayEquals(jvmOutput, espressoOutput, 0.0001f,
                "Output " + i + " should be identical between JVM and Espresso modes");
        }

        System.out.println("SUCCESS: JVM and Espresso modes produce identical outputs");
        System.out.println("  JVM output: " + Arrays.toString(jvmResult.outputs().getFirst()));
        System.out.println("  Espresso output: " + Arrays.toString(espressoResult.outputs().getFirst()));
    }

    @Test
    void testAllModesProduceIdenticalOutputs() throws Exception {
        List<ExecutionMode> availableModes = getAvailableModes();

        System.out.println("Testing consistency across modes: " + availableModes);

        if (availableModes.size() < 2) {
            System.out.println("Skipping multi-mode consistency test: only " + availableModes.size() + " mode(s) available");
            return;
        }

        // Run model in each available mode
        TestRunner.RunResult referenceResult = null;
        ExecutionMode referenceMode = null;

        for (ExecutionMode mode : availableModes) {
            TestRunner.RunnerArgs args = new TestRunner.RunnerArgs(
                muladdModelJar,
                mode,
                "cpu",
                new int[]{2, 2}
            );

            TestRunner.RunResult result = TestRunner.run(args);
            assertTrue(result.success(), mode + " mode should succeed");

            if (referenceResult == null) {
                referenceResult = result;
                referenceMode = mode;
                System.out.println("Reference mode: " + mode);
                System.out.println("Reference output: " + Arrays.toString(result.outputs().getFirst()));
            } else {
                // Compare with reference
                assertEquals(referenceResult.outputs().size(), result.outputs().size(),
                    "Output count mismatch between " + referenceMode + " and " + mode);

                for (int i = 0; i < referenceResult.outputs().size(); i++) {
                    assertArrayEquals(
                        referenceResult.outputs().get(i),
                        result.outputs().get(i),
                        0.0001f,
                        "Output " + i + " mismatch between " + referenceMode + " and " + mode
                    );
                }

                System.out.println(mode + " mode matches reference (" + referenceMode + ")");
            }
        }

        System.out.println("SUCCESS: All " + availableModes.size() + " modes produce identical outputs");
    }

    /**
     * Check if Espresso is available in the current runtime.
     *
     * <p>Espresso requires GraalVM with the Java language installed.
     * This check verifies both the polyglot API and actual Espresso functionality.
     */
    private static boolean isEspressoAvailable() {
        try {
            Class.forName("org.graalvm.polyglot.Context");
            // Try to actually create a Java context with classpath option
            // This triggers full Espresso initialization which may fail on non-GraalVM JVMs
            org.graalvm.polyglot.Context context = org.graalvm.polyglot.Context.newBuilder("java")
                .allowAllAccess(true)
                .option("java.Classpath", System.getProperty("java.class.path"))
                .build();
            context.close();
            return true;
        } catch (Exception | Error e) {
            // Catch Error as well (includes NoClassDefFoundError, ExceptionInInitializerError)
            System.out.println("Espresso not available: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return false;
        }
    }

    /**
     * Get list of available execution modes in the current environment.
     */
    private static List<ExecutionMode> getAvailableModes() {
        java.util.ArrayList<ExecutionMode> modes = new java.util.ArrayList<>();

        // JVM mode is always available
        modes.add(ExecutionMode.JVM);

        // Check Espresso availability
        if (isEspressoAvailable()) {
            modes.add(ExecutionMode.ESPRESSO);
        }

        // Native mode requires AOT compilation - not tested here

        return modes;
    }
}
