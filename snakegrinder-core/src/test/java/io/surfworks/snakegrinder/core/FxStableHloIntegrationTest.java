package io.surfworks.snakegrinder.core;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests that invoke the snakegrinder native image CLI.
 *
 * These tests verify end-to-end FX tracing functionality by running the actual
 * native-image binary, which includes the full GraalPy runtime and PyTorch.
 *
 * Prerequisites:
 * - Run: ./gradlew :snakegrinder-dist:assembleDist
 * - Or: ./gradlew :snakegrinder-dist:buildApp (for macOS .app bundle)
 *
 * Run: ./gradlew :snakegrinder-core:integrationTest
 */
@Tag("integration")
class FxStableHloIntegrationTest {

    private static Path snakegrinderBinary;
    private static Path tempOutputDir;

    /**
     * Check if the native image binary exists.
     * Searches in order:
     * 1. macOS .app bundle
     * 2. assembleDist output
     * 3. nativeCompile output
     */
    static boolean isNativeImageAvailable() {
        snakegrinderBinary = findSnakegrinderBinary();
        return snakegrinderBinary != null && Files.isExecutable(snakegrinderBinary);
    }

    private static Path findSnakegrinderBinary() {
        Path projectRoot = findProjectRoot();
        if (projectRoot == null) {
            return null;
        }

        // Check macOS .app bundle first
        Path appBinary = projectRoot.resolve("snakegrinder-dist/build/SnakeGrinder.app/Contents/MacOS/snakegrinder");
        if (Files.isExecutable(appBinary)) {
            return appBinary;
        }

        // Check assembleDist output
        Path distBinary = projectRoot.resolve("snakegrinder-dist/build/dist/bin/snakegrinder-bin");
        if (Files.isExecutable(distBinary)) {
            return distBinary;
        }

        // Check direct nativeCompile output
        Path nativeBinary = projectRoot.resolve("snakegrinder-cli/build/native/nativeCompile/snakegrinder");
        if (Files.isExecutable(nativeBinary)) {
            return nativeBinary;
        }

        return null;
    }

    private static Path findProjectRoot() {
        Path current = Paths.get("").toAbsolutePath();

        // Walk up to find the project root (contains settings.gradle)
        while (current != null) {
            if (Files.exists(current.resolve("settings.gradle"))) {
                return current;
            }
            current = current.getParent();
        }

        // Try common locations
        Path userDir = Paths.get(System.getProperty("user.dir"));
        if (Files.exists(userDir.resolve("settings.gradle"))) {
            return userDir;
        }

        return null;
    }

    @BeforeAll
    static void setup() throws IOException {
        tempOutputDir = Files.createTempDirectory("snakegrinder-test");
    }

    @AfterAll
    static void cleanup() throws IOException {
        if (tempOutputDir != null && Files.exists(tempOutputDir)) {
            Files.walk(tempOutputDir)
                    .sorted((a, b) -> b.compareTo(a)) // Reverse order for deletion
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            // Ignore cleanup errors
                        }
                    });
        }
    }

    // ========================================================================
    // CLI Execution Helper
    // ========================================================================

    private CliResult runSnakegrinder(String... args) throws IOException, InterruptedException {
        if (snakegrinderBinary == null) {
            snakegrinderBinary = findSnakegrinderBinary();
        }

        assertNotNull(snakegrinderBinary, "Native image binary not found");

        String[] command = new String[args.length + 1];
        command[0] = snakegrinderBinary.toString();
        System.arraycopy(args, 0, command, 1, args.length);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(false);

        // Set library path for PyTorch native libs
        Path projectRoot = findProjectRoot();
        if (projectRoot != null) {
            Path venvLibs = projectRoot.resolve("snakegrinder-dist/.pytorch-venv/lib/python3.12/site-packages/torch/lib");
            if (Files.exists(venvLibs)) {
                String osName = System.getProperty("os.name", "").toLowerCase();
                if (osName.contains("mac")) {
                    pb.environment().put("DYLD_LIBRARY_PATH", venvLibs.toString());
                } else {
                    pb.environment().put("LD_LIBRARY_PATH", venvLibs.toString());
                }
            }
        }

        Process process = pb.start();

        StringBuilder stdout = new StringBuilder();
        StringBuilder stderr = new StringBuilder();

        Thread stdoutReader = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                reader.lines().forEach(line -> stdout.append(line).append("\n"));
            } catch (IOException e) {
                // Ignore
            }
        });

        Thread stderrReader = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                reader.lines().forEach(line -> stderr.append(line).append("\n"));
            } catch (IOException e) {
                // Ignore
            }
        });

        stdoutReader.start();
        stderrReader.start();

        boolean completed = process.waitFor(120, TimeUnit.SECONDS);
        assertTrue(completed, "Process timed out after 120 seconds");

        stdoutReader.join(5000);
        stderrReader.join(5000);

        return new CliResult(process.exitValue(), stdout.toString(), stderr.toString());
    }

    record CliResult(int exitCode, String stdout, String stderr) {
        boolean success() {
            return exitCode == 0;
        }

        String output() {
            return stdout + stderr;
        }
    }

    // ========================================================================
    // Help and Info Tests
    // ========================================================================

    @Nested
    class HelpTests {

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void helpShowsUsage() throws IOException, InterruptedException {
            CliResult result = runSnakegrinder("--help");

            assertTrue(result.success(), "Exit code should be 0");
            assertTrue(result.stdout().contains("snakegrinder"), "Should show program name");
            assertTrue(result.stdout().contains("--trace"), "Should show --trace option");
            assertTrue(result.stdout().contains("--trace-example"), "Should show --trace-example option");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void pytorchInfoShowsVersion() throws IOException, InterruptedException {
            CliResult result = runSnakegrinder("--pytorch-info");

            assertTrue(result.success(), "Exit code should be 0: " + result.stderr());
            assertTrue(result.stdout().contains("PyTorch version"), "Should show PyTorch version");
            assertTrue(result.stdout().contains("FX available"), "Should show FX availability");
        }
    }

    // ========================================================================
    // Trace Example Tests (Built-in MLP)
    // ========================================================================

    @Nested
    class TraceExampleTests {

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceExampleSucceeds() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("trace-example-1");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());

            assertTrue(result.success(), "Trace should succeed: " + result.stderr());
            assertTrue(result.stdout().contains("SUCCESS"), "Should report success");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceExampleWritesMlirFile() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("trace-example-2");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success(), "Trace should succeed: " + result.stderr());

            Path mlirFile = outputDir.resolve("model.mlir");
            assertTrue(Files.exists(mlirFile), "Should create model.mlir file");

            String mlir = Files.readString(mlirFile);
            assertFalse(mlir.isBlank(), "MLIR file should not be empty");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceExampleWritesManifestFile() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("trace-example-3");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success(), "Trace should succeed: " + result.stderr());

            Path manifestFile = outputDir.resolve("manifest.json");
            assertTrue(Files.exists(manifestFile), "Should create manifest.json file");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceExampleOutputsValidMlir() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("trace-example-4");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success(), "Trace should succeed: " + result.stderr());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            // Verify module structure
            assertTrue(mlir.contains("module"), "Should have module declaration");
            assertTrue(mlir.contains("func.func") || mlir.contains("func @"), "Should have function declaration");
        }
    }

    // ========================================================================
    // MLIR Output Format Tests
    // ========================================================================

    @Nested
    class MlirFormatTests {

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void mlirHasModuleDeclaration() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("mlir-format-1");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            // Pattern: module @name {
            Pattern modulePattern = Pattern.compile("module\\s+@\\w+\\s*\\{");
            Matcher matcher = modulePattern.matcher(mlir);
            assertTrue(matcher.find(), "Should have module @name { declaration");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void mlirHasStableHloOps() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("mlir-format-2");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            assertTrue(mlir.contains("stablehlo."), "Should have stablehlo operations");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void mlirHasTensorTypes() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("mlir-format-3");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            // Pattern: tensor<NxMxf32>
            Pattern tensorPattern = Pattern.compile("tensor<[^>]+>");
            Matcher matcher = tensorPattern.matcher(mlir);
            assertTrue(matcher.find(), "Should have tensor types");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void mlirHasBalancedBraces() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("mlir-format-4");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            long openBraces = mlir.chars().filter(c -> c == '{').count();
            long closeBraces = mlir.chars().filter(c -> c == '}').count();
            assertEquals(openBraces, closeBraces, "Braces should be balanced");

            long openParens = mlir.chars().filter(c -> c == '(').count();
            long closeParens = mlir.chars().filter(c -> c == ')').count();
            assertEquals(openParens, closeParens, "Parentheses should be balanced");
        }
    }

    // ========================================================================
    // StableHLO Operation Tests
    // ========================================================================

    @Nested
    class StableHloOperationTests {

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void hasDotGeneralForMatmul() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("stablehlo-ops-1");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            // MLP has linear layers which use dot_general
            assertTrue(mlir.contains("stablehlo.dot_general"),
                    "MLP should have stablehlo.dot_general for matrix multiplication");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void hasMaximumForRelu() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("stablehlo-ops-2");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            // MLP uses ReLU which becomes max(x, 0)
            assertTrue(mlir.contains("stablehlo.maximum"),
                    "MLP should have stablehlo.maximum for ReLU activation");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void hasConstantOps() throws IOException, InterruptedException {
            Path outputDir = tempOutputDir.resolve("stablehlo-ops-3");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            assertTrue(mlir.contains("stablehlo.constant"),
                    "Should have stablehlo.constant operations");
        }
    }

    // ========================================================================
    // Custom Model Tracing Tests
    // ========================================================================

    @Nested
    class CustomModelTests {

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceCustomModelFromFile() throws IOException, InterruptedException {
            // Create a temporary Python file with a simple model
            Path modelFile = tempOutputDir.resolve("simple_model.py");
            Files.writeString(modelFile, """
                import torch
                import torch.nn as nn

                class SimpleAdd(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        return x + x
                """);

            Path outputDir = tempOutputDir.resolve("custom-model-1");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder(
                    "--trace",
                    "--source", modelFile.toString(),
                    "--class", "SimpleAdd",
                    "--inputs", "[(1,4)]",
                    "--out", outputDir.toString());

            assertTrue(result.success(), "Trace should succeed: " + result.stderr());

            Path mlirFile = outputDir.resolve("model.mlir");
            assertTrue(Files.exists(mlirFile), "Should create MLIR file");

            String mlir = Files.readString(mlirFile);
            assertTrue(mlir.contains("stablehlo.add"), "x + x should produce stablehlo.add");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceLinearLayer() throws IOException, InterruptedException {
            Path modelFile = tempOutputDir.resolve("linear_model.py");
            Files.writeString(modelFile, """
                import torch
                import torch.nn as nn

                class LinearModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc = nn.Linear(8, 4)

                    def forward(self, x):
                        return self.fc(x)
                """);

            Path outputDir = tempOutputDir.resolve("custom-model-2");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder(
                    "--trace",
                    "--source", modelFile.toString(),
                    "--class", "LinearModel",
                    "--inputs", "[(1,8)]",
                    "--out", outputDir.toString());

            assertTrue(result.success(), "Trace should succeed: " + result.stderr());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));
            assertTrue(mlir.contains("stablehlo.dot_general"),
                    "Linear layer should produce dot_general");
        }
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    @Nested
    class ErrorHandlingTests {

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void unknownArgsShowsError() throws IOException, InterruptedException {
            CliResult result = runSnakegrinder("--unknown-flag");

            assertFalse(result.success(), "Should fail with unknown flag");
            assertTrue(result.stderr().contains("Unknown") || result.stderr().contains("--help"),
                    "Should suggest --help");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceWithoutSourceFails() throws IOException, InterruptedException {
            CliResult result = runSnakegrinder("--trace", "--class", "Foo", "--inputs", "[(1,4)]");

            assertFalse(result.success(), "Should fail without --source");
            assertTrue(result.stderr().contains("--source"), "Should mention missing --source");
        }

        @Test
        @EnabledIf("io.surfworks.snakegrinder.core.FxStableHloIntegrationTest#isNativeImageAvailable")
        void traceNonexistentFileFails() throws IOException, InterruptedException {
            CliResult result = runSnakegrinder(
                    "--trace",
                    "--source", "/nonexistent/path/model.py",
                    "--class", "Model",
                    "--inputs", "[(1,4)]");

            assertFalse(result.success(), "Should fail with nonexistent file");
        }
    }
}
