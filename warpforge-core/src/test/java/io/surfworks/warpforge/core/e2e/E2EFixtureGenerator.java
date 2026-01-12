package io.surfworks.warpforge.core.e2e;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;

/**
 * Generator for E2E test fixtures using the snakegrinder-dist native binary.
 *
 * <p>This is NOT a regular test - it's a fixture generator that:
 * <ol>
 *   <li>Invokes snakegrinder-dist/build/dist/bin/snakegrinder (native binary)</li>
 *   <li>Uses --trace-with-values to capture tensor data</li>
 *   <li>Writes fixtures to warpforge-core/src/test/resources/fixtures/e2e/</li>
 * </ol>
 *
 * <p>Run manually with: ./gradlew :warpforge-core:generateE2EFixtures
 *
 * <p>Prerequisites:
 * <ul>
 *   <li>./gradlew :snakegrinder-dist:assembleDist (build the native distribution)</li>
 * </ul>
 */
@Tag("fixture-generator")
@Tag("requires-snakegrinder-dist")
@DisplayName("E2E Fixture Generator")
class E2EFixtureGenerator {

    private static final Path PROJECT_ROOT = findProjectRoot();
    private static final Path SNAKEGRINDER_BINARY = PROJECT_ROOT.resolve(
        "snakegrinder-dist/build/dist/bin/snakegrinder"
    );
    private static final Path FIXTURES_OUTPUT_DIR = PROJECT_ROOT.resolve(
        "warpforge-core/src/test/resources/fixtures/e2e"
    );
    private static final long DEFAULT_SEED = 42;
    private static final int TIMEOUT_SECONDS = 120;

    @TempDir
    Path tempDir;

    private static Path findProjectRoot() {
        Path current = Paths.get("").toAbsolutePath();
        while (current != null) {
            if (Files.exists(current.resolve("settings.gradle")) ||
                Files.exists(current.resolve("settings.gradle.kts"))) {
                return current;
            }
            current = current.getParent();
        }
        return Paths.get("").toAbsolutePath();
    }

    @BeforeAll
    static void checkPrerequisites() {
        assumeTrue(Files.exists(SNAKEGRINDER_BINARY),
            "snakegrinder binary not found at " + SNAKEGRINDER_BINARY +
            ". Run: ./gradlew :snakegrinder-dist:assembleDist");
        assumeTrue(Files.isExecutable(SNAKEGRINDER_BINARY),
            "snakegrinder binary is not executable");
    }

    // ==================== Tier 1: Elementwise Operations ====================

    @Test
    @DisplayName("Generate: add")
    void generateAdd() throws Exception {
        generateFixture("add", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x, y):
                    return x + y
            """, "[(4,), (4,)]");
    }

    @Test
    @DisplayName("Generate: subtract")
    void generateSubtract() throws Exception {
        generateFixture("subtract", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x, y):
                    return x - y
            """, "[(4,), (4,)]");
    }

    @Test
    @DisplayName("Generate: multiply")
    void generateMultiply() throws Exception {
        generateFixture("multiply", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x, y):
                    return x * y
            """, "[(4,), (4,)]");
    }

    @Test
    @DisplayName("Generate: negate")
    void generateNegate() throws Exception {
        generateFixture("negate", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return -x
            """, "[(4,)]");
    }

    @Test
    @DisplayName("Generate: abs")
    void generateAbs() throws Exception {
        generateFixture("abs", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.abs(x)
            """, "[(4,)]");
    }

    // ==================== Tier 2: Transcendental Operations ====================

    @Test
    @DisplayName("Generate: exp")
    void generateExp() throws Exception {
        generateFixture("exp", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.exp(x)
            """, "[(4,)]");
    }

    @Test
    @DisplayName("Generate: tanh")
    void generateTanh() throws Exception {
        generateFixture("tanh", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.tanh(x)
            """, "[(4,)]");
    }

    @Test
    @DisplayName("Generate: sigmoid")
    void generateSigmoid() throws Exception {
        generateFixture("sigmoid", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.sigmoid(x)
            """, "[(4,)]");
    }

    // ==================== Tier 3: ReLU (composite) ====================

    @Test
    @DisplayName("Generate: relu")
    void generateRelu() throws Exception {
        generateFixture("relu", """
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.relu(x)
            """, "[(4,)]");
    }

    // ==================== Helper Methods ====================

    private void generateFixture(String name, String modelSource, String inputSpecs) throws Exception {
        Path outputDir = tempDir.resolve(name);
        Files.createDirectories(outputDir);

        // Write model source to temp file
        Path modelFile = tempDir.resolve(name + "_model.py");
        Files.writeString(modelFile, modelSource);

        // Invoke snakegrinder --trace-with-values
        ProcessBuilder pb = new ProcessBuilder(
            SNAKEGRINDER_BINARY.toString(),
            "--trace-with-values",
            "--source", modelFile.toString(),
            "--class", "Model",
            "--inputs", inputSpecs,
            "--seed", String.valueOf(DEFAULT_SEED),
            "--out", outputDir.toString()
        );

        pb.redirectErrorStream(true);

        Process process = pb.start();
        StringBuilder output = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
        }

        boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            fail("Timeout generating fixture: " + name);
        }

        int exitCode = process.exitValue();
        if (exitCode != 0) {
            fail("Failed to generate fixture '" + name + "' (exit " + exitCode + "):\n" + output);
        }

        // Verify output files exist
        assertTrue(Files.exists(outputDir.resolve("model.mlir")),
            "model.mlir not created for " + name);
        assertTrue(Files.exists(outputDir.resolve("inputs")),
            "inputs/ directory not created for " + name);
        assertTrue(Files.exists(outputDir.resolve("outputs")),
            "outputs/ directory not created for " + name);

        // Copy to fixtures directory
        Path fixtureDir = FIXTURES_OUTPUT_DIR.resolve(name);
        copyDirectory(outputDir, fixtureDir);

        System.out.println("Generated fixture: " + name + " -> " + fixtureDir);
    }

    private void copyDirectory(Path source, Path target) throws IOException {
        Files.createDirectories(target);

        Files.walk(source).forEach(sourcePath -> {
            try {
                Path targetPath = target.resolve(source.relativize(sourcePath));
                if (Files.isDirectory(sourcePath)) {
                    Files.createDirectories(targetPath);
                } else {
                    Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to copy: " + sourcePath, e);
            }
        });
    }
}
