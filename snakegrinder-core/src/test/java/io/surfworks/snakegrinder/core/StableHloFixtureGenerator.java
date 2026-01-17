package io.surfworks.snakegrinder.core;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * StableHLO Fixture Generator
 *
 * Generates MLIR fixture files by tracing PyTorch operations through snakegrinder.
 * These fixtures can be used by both snakegrinder (output validation) and
 * snakeburger (parser input testing).
 *
 * Run with: ./gradlew :snakegrinder-core:generateFixtures
 *
 * The generated fixtures are saved to:
 *   snakegrinder-core/src/test/resources/fixtures/stablehlo/
 *
 * This structure allows the fixtures to be:
 * 1. Committed to the repository as test resources
 * 2. Used by snakeburger tests via file copy or symlink
 * 3. Validated for correctness by snakegrinder tests
 */
@Tag("fixture-generator")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class StableHloFixtureGenerator {

    private static Path snakegrinderBinary;
    private static Path fixturesDir;
    private static Path tempDir;

    // Operation definitions: name -> (python source, input specs, expected ops)
    record OpDefinition(
            String name,
            String category,
            String pythonSource,
            String className,
            String inputSpecs,
            List<String> expectedOps
    ) {}

    static boolean isNativeImageAvailable() {
        snakegrinderBinary = findSnakegrinderBinary();
        return snakegrinderBinary != null && Files.isExecutable(snakegrinderBinary);
    }

    private static Path findSnakegrinderBinary() {
        Path projectRoot = findProjectRoot();
        if (projectRoot == null) return null;

        Path appBinary = projectRoot.resolve("snakegrinder-dist/build/SnakeGrinder.app/Contents/MacOS/snakegrinder");
        if (Files.isExecutable(appBinary)) return appBinary;

        Path distBinary = projectRoot.resolve("snakegrinder-dist/build/dist/bin/snakegrinder-bin");
        if (Files.isExecutable(distBinary)) return distBinary;

        Path nativeBinary = projectRoot.resolve("snakegrinder-cli/build/native/nativeCompile/snakegrinder");
        if (Files.isExecutable(nativeBinary)) return nativeBinary;

        return null;
    }

    private static Path findProjectRoot() {
        Path current = Paths.get("").toAbsolutePath();
        while (current != null) {
            if (Files.exists(current.resolve("settings.gradle"))) return current;
            current = current.getParent();
        }
        return Paths.get(System.getProperty("user.dir"));
    }

    @BeforeAll
    static void setup() throws IOException {
        Path projectRoot = findProjectRoot();
        fixturesDir = projectRoot.resolve("snakegrinder-core/src/test/resources/fixtures/stablehlo");
        tempDir = Files.createTempDirectory("stablehlo-fixtures");

        // Create fixture directories
        Files.createDirectories(fixturesDir.resolve("elementwise"));
        Files.createDirectories(fixturesDir.resolve("transcendental"));
        Files.createDirectories(fixturesDir.resolve("comparison"));
        Files.createDirectories(fixturesDir.resolve("selection"));
        Files.createDirectories(fixturesDir.resolve("matrix"));
        Files.createDirectories(fixturesDir.resolve("shape"));
        Files.createDirectories(fixturesDir.resolve("slicing"));
        Files.createDirectories(fixturesDir.resolve("reduction"));
        Files.createDirectories(fixturesDir.resolve("activation"));
        Files.createDirectories(fixturesDir.resolve("convolution"));
        Files.createDirectories(fixturesDir.resolve("pooling"));
        Files.createDirectories(fixturesDir.resolve("normalization"));
        Files.createDirectories(fixturesDir.resolve("padding"));
        Files.createDirectories(fixturesDir.resolve("constant"));
        Files.createDirectories(fixturesDir.resolve("composite"));
    }

    @AfterAll
    static void cleanup() throws IOException {
        if (tempDir != null && Files.exists(tempDir)) {
            Files.walk(tempDir)
                    .sorted((a, b) -> b.compareTo(a))
                    .forEach(path -> {
                        try { Files.deleteIfExists(path); } catch (IOException e) { }
                    });
        }
    }

    // ========================================================================
    // Operation Definitions
    // ========================================================================

    static Stream<OpDefinition> elementwiseOps() {
        return Stream.of(
                new OpDefinition("add", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class AddOp(nn.Module):
                            def forward(self, x, y):
                                return x + y
                        """,
                        "AddOp", "[(1,8),(1,8)]",
                        List.of("stablehlo.add")),

                new OpDefinition("subtract", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class SubtractOp(nn.Module):
                            def forward(self, x, y):
                                return x - y
                        """,
                        "SubtractOp", "[(1,8),(1,8)]",
                        List.of("stablehlo.subtract")),

                new OpDefinition("multiply", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class MultiplyOp(nn.Module):
                            def forward(self, x, y):
                                return x * y
                        """,
                        "MultiplyOp", "[(1,8),(1,8)]",
                        List.of("stablehlo.multiply")),

                new OpDefinition("divide", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class DivideOp(nn.Module):
                            def forward(self, x, y):
                                return x / y
                        """,
                        "DivideOp", "[(1,8),(1,8)]",
                        List.of("stablehlo.divide")),

                new OpDefinition("maximum", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class MaximumOp(nn.Module):
                            def forward(self, x, y):
                                return torch.maximum(x, y)
                        """,
                        "MaximumOp", "[(1,8),(1,8)]",
                        List.of("stablehlo.maximum")),

                new OpDefinition("minimum", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class MinimumOp(nn.Module):
                            def forward(self, x, y):
                                return torch.minimum(x, y)
                        """,
                        "MinimumOp", "[(1,8),(1,8)]",
                        List.of("stablehlo.minimum")),

                new OpDefinition("negate", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class NegateOp(nn.Module):
                            def forward(self, x):
                                return -x
                        """,
                        "NegateOp", "[(1,8)]",
                        List.of("stablehlo.negate")),

                new OpDefinition("abs", "elementwise",
                        """
                        import torch
                        import torch.nn as nn
                        class AbsOp(nn.Module):
                            def forward(self, x):
                                return torch.abs(x)
                        """,
                        "AbsOp", "[(1,8)]",
                        List.of("stablehlo.abs"))
        );
    }

    static Stream<OpDefinition> transcendentalOps() {
        return Stream.of(
                new OpDefinition("exp", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class ExpOp(nn.Module):
                            def forward(self, x):
                                return torch.exp(x)
                        """,
                        "ExpOp", "[(1,8)]",
                        List.of("stablehlo.exponential")),

                new OpDefinition("log", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class LogOp(nn.Module):
                            def forward(self, x):
                                return torch.log(x)
                        """,
                        "LogOp", "[(1,8)]",
                        List.of("stablehlo.log")),

                new OpDefinition("sqrt", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class SqrtOp(nn.Module):
                            def forward(self, x):
                                return torch.sqrt(x)
                        """,
                        "SqrtOp", "[(1,8)]",
                        List.of("stablehlo.sqrt")),

                new OpDefinition("sin", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class SinOp(nn.Module):
                            def forward(self, x):
                                return torch.sin(x)
                        """,
                        "SinOp", "[(1,8)]",
                        List.of("stablehlo.sine")),

                new OpDefinition("cos", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class CosOp(nn.Module):
                            def forward(self, x):
                                return torch.cos(x)
                        """,
                        "CosOp", "[(1,8)]",
                        List.of("stablehlo.cosine")),

                new OpDefinition("tanh", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class TanhOp(nn.Module):
                            def forward(self, x):
                                return torch.tanh(x)
                        """,
                        "TanhOp", "[(1,8)]",
                        List.of("stablehlo.tanh")),

                new OpDefinition("sigmoid", "transcendental",
                        """
                        import torch
                        import torch.nn as nn
                        class SigmoidOp(nn.Module):
                            def forward(self, x):
                                return torch.sigmoid(x)
                        """,
                        "SigmoidOp", "[(1,8)]",
                        List.of("stablehlo.logistic"))
        );
    }

    static Stream<OpDefinition> matrixOps() {
        return Stream.of(
                new OpDefinition("matmul", "matrix",
                        """
                        import torch
                        import torch.nn as nn
                        class MatmulOp(nn.Module):
                            def forward(self, x, y):
                                return torch.matmul(x, y)
                        """,
                        "MatmulOp", "[(1,4,8),(1,8,4)]",
                        List.of("stablehlo.dot_general")),

                new OpDefinition("linear", "matrix",
                        """
                        import torch
                        import torch.nn as nn
                        class LinearOp(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.fc = nn.Linear(8, 4)
                            def forward(self, x):
                                return self.fc(x)
                        """,
                        "LinearOp", "[(1,8)]",
                        List.of("stablehlo.dot_general", "stablehlo.add"))
        );
    }

    static Stream<OpDefinition> shapeOps() {
        return Stream.of(
                new OpDefinition("reshape", "shape",
                        """
                        import torch
                        import torch.nn as nn
                        class ReshapeOp(nn.Module):
                            def forward(self, x):
                                return x.reshape(-1, 4)
                        """,
                        "ReshapeOp", "[(2,8)]",
                        List.of("stablehlo.reshape")),

                new OpDefinition("transpose", "shape",
                        """
                        import torch
                        import torch.nn as nn
                        class TransposeOp(nn.Module):
                            def forward(self, x):
                                return x.transpose(0, 1)
                        """,
                        "TransposeOp", "[(4,8)]",
                        List.of("stablehlo.transpose")),

                new OpDefinition("concat", "shape",
                        """
                        import torch
                        import torch.nn as nn
                        class ConcatOp(nn.Module):
                            def forward(self, x, y):
                                return torch.cat([x, y], dim=1)
                        """,
                        "ConcatOp", "[(1,4),(1,4)]",
                        List.of("stablehlo.concatenate")),

                new OpDefinition("broadcast", "shape",
                        """
                        import torch
                        import torch.nn as nn
                        class BroadcastOp(nn.Module):
                            def forward(self, x):
                                return x.expand(4, -1, -1)
                        """,
                        "BroadcastOp", "[(1,4,8)]",
                        List.of("stablehlo.broadcast_in_dim"))
        );
    }

    static Stream<OpDefinition> reductionOps() {
        return Stream.of(
                new OpDefinition("sum_reduce", "reduction",
                        """
                        import torch
                        import torch.nn as nn
                        class SumReduceOp(nn.Module):
                            def forward(self, x):
                                return torch.sum(x, dim=1)
                        """,
                        "SumReduceOp", "[(2,8)]",
                        List.of("stablehlo.reduce")),

                new OpDefinition("max_reduce", "reduction",
                        """
                        import torch
                        import torch.nn as nn
                        class MaxReduceOp(nn.Module):
                            def forward(self, x):
                                return torch.max(x, dim=1)[0]
                        """,
                        "MaxReduceOp", "[(2,8)]",
                        List.of("stablehlo.reduce"))
        );
    }

    static Stream<OpDefinition> activationOps() {
        return Stream.of(
                new OpDefinition("relu", "activation",
                        """
                        import torch
                        import torch.nn as nn
                        import torch.nn.functional as F
                        class ReLUOp(nn.Module):
                            def forward(self, x):
                                return F.relu(x)
                        """,
                        "ReLUOp", "[(1,8)]",
                        List.of("stablehlo.maximum")),

                new OpDefinition("softmax", "activation",
                        """
                        import torch
                        import torch.nn as nn
                        import torch.nn.functional as F
                        class SoftmaxOp(nn.Module):
                            def forward(self, x):
                                return F.softmax(x, dim=-1)
                        """,
                        "SoftmaxOp", "[(1,8)]",
                        List.of("stablehlo.exponential", "stablehlo.reduce", "stablehlo.divide"))
        );
    }

    static Stream<OpDefinition> convolutionOps() {
        return Stream.of(
                new OpDefinition("conv2d", "convolution",
                        """
                        import torch
                        import torch.nn as nn
                        class Conv2dOp(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                            def forward(self, x):
                                return self.conv(x)
                        """,
                        "Conv2dOp", "[(1,3,8,8)]",
                        List.of("stablehlo.convolution"))
        );
    }

    static Stream<OpDefinition> poolingOps() {
        return Stream.of(
                new OpDefinition("max_pool2d", "pooling",
                        """
                        import torch
                        import torch.nn as nn
                        class MaxPool2dOp(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                            def forward(self, x):
                                return self.pool(x)
                        """,
                        "MaxPool2dOp", "[(1,4,8,8)]",
                        List.of("stablehlo.reduce_window"))
        );
    }

    static Stream<OpDefinition> slicingOps() {
        return Stream.of(
                new OpDefinition("slice", "slicing",
                        """
                        import torch
                        import torch.nn as nn
                        class SliceOp(nn.Module):
                            def forward(self, x):
                                return x[:, 1:3]
                        """,
                        "SliceOp", "[(1,8)]",
                        List.of("stablehlo.slice"))
        );
    }

    static Stream<OpDefinition> constantOps() {
        return Stream.of(
                new OpDefinition("constant", "constant",
                        """
                        import torch
                        import torch.nn as nn
                        class ConstantOp(nn.Module):
                            def forward(self, x):
                                ones = torch.ones_like(x)
                                return x + ones
                        """,
                        "ConstantOp", "[(1,8)]",
                        List.of("stablehlo.constant"))
        );
    }

    static Stream<OpDefinition> compositeOps() {
        return Stream.of(
                new OpDefinition("simple_mlp", "composite",
                        """
                        import torch
                        import torch.nn as nn
                        import torch.nn.functional as F
                        class SimpleMLP(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.fc1 = nn.Linear(8, 16)
                                self.fc2 = nn.Linear(16, 4)
                            def forward(self, x):
                                x = F.relu(self.fc1(x))
                                return self.fc2(x)
                        """,
                        "SimpleMLP", "[(1,8)]",
                        List.of("stablehlo.dot_general", "stablehlo.maximum", "stablehlo.add")),

                new OpDefinition("residual_block", "composite",
                        """
                        import torch
                        import torch.nn as nn
                        import torch.nn.functional as F
                        class ResidualBlock(nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.fc1 = nn.Linear(8, 8)
                                self.fc2 = nn.Linear(8, 8)
                            def forward(self, x):
                                residual = x
                                x = F.relu(self.fc1(x))
                                x = self.fc2(x)
                                return x + residual
                        """,
                        "ResidualBlock", "[(1,8)]",
                        List.of("stablehlo.dot_general", "stablehlo.maximum", "stablehlo.add"))
        );
    }

    static Stream<OpDefinition> allOps() {
        return Stream.of(
                elementwiseOps(),
                transcendentalOps(),
                matrixOps(),
                shapeOps(),
                reductionOps(),
                activationOps(),
                convolutionOps(),
                poolingOps(),
                slicingOps(),
                constantOps(),
                compositeOps()
        ).flatMap(s -> s);
    }

    // ========================================================================
    // Fixture Generation
    // ========================================================================

    @ParameterizedTest(name = "{0}")
    @MethodSource("allOps")
    @EnabledIf("io.surfworks.snakegrinder.core.StableHloFixtureGenerator#isNativeImageAvailable")
    @Order(1)
    void generateFixture(OpDefinition op) throws Exception {
        // Write Python source to temp file
        Path pythonFile = tempDir.resolve(op.name + ".py");
        Files.writeString(pythonFile, op.pythonSource);

        // Create output directory
        Path outputDir = tempDir.resolve("output-" + op.name);
        Files.createDirectories(outputDir);

        // Run snakegrinder
        CliResult result = runSnakegrinder(
                "--trace",
                "--source", pythonFile.toString(),
                "--class", op.className,
                "--inputs", op.inputSpecs,
                "--out", outputDir.toString()
        );

        // Check for success
        if (!result.success()) {
            System.err.println("Failed to generate fixture for: " + op.name);
            System.err.println("Exit code: " + result.exitCode());
            System.err.println("Stderr: " + result.stderr());
            System.err.println("Stdout: " + result.stdout());

            // Write error file instead
            Path errorFile = fixturesDir.resolve(op.category).resolve(op.name + ".error");
            Files.writeString(errorFile, "Generation failed:\n" + result.stderr());
            return;
        }

        // Read MLIR output
        Path mlirFile = outputDir.resolve("model.mlir");
        assertTrue(Files.exists(mlirFile), "MLIR file should exist for " + op.name);

        String mlir = Files.readString(mlirFile);

        // Save to fixtures directory
        Path fixtureFile = fixturesDir.resolve(op.category).resolve(op.name + ".mlir");
        Files.writeString(fixtureFile, mlir);

        System.out.println("Generated fixture: " + fixtureFile);

        // Check which expected operations are present
        List<String> foundOps = new ArrayList<>();
        List<String> missingOps = new ArrayList<>();
        for (String expectedOp : op.expectedOps) {
            if (mlir.contains(expectedOp)) {
                foundOps.add(expectedOp);
            } else {
                missingOps.add(expectedOp);
            }
        }

        // Report coverage (don't fail - just report)
        if (!missingOps.isEmpty()) {
            System.out.println("  WARNING: Missing operations for " + op.name + ": " + missingOps);
            System.out.println("  (This indicates the converter doesn't support these ops yet)");
        }
        if (!foundOps.isEmpty()) {
            System.out.println("  Found operations: " + foundOps);
        }

        // Check for "Unsupported function" comments which indicate gaps
        if (mlir.contains("// Unsupported")) {
            System.out.println("  NOTE: Contains unsupported function markers - converter needs enhancement");
        }
    }

    // ========================================================================
    // CLI Helper
    // ========================================================================

    private CliResult runSnakegrinder(String... args) throws IOException, InterruptedException {
        String[] command = new String[args.length + 1];
        command[0] = snakegrinderBinary.toString();
        System.arraycopy(args, 0, command, 1, args.length);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(false);

        // Set library path for PyTorch native libs
        Path projectRoot = findProjectRoot();
        Path venvLibs = projectRoot.resolve("snakegrinder-dist/.pytorch-venv/lib/python3.12/site-packages/torch/lib");
        if (Files.exists(venvLibs)) {
            String osName = System.getProperty("os.name", "").toLowerCase();
            if (osName.contains("mac")) {
                pb.environment().put("DYLD_LIBRARY_PATH", venvLibs.toString());
            } else {
                pb.environment().put("LD_LIBRARY_PATH", venvLibs.toString());
            }
        }

        Process process = pb.start();

        StringBuilder stdout = new StringBuilder();
        StringBuilder stderr = new StringBuilder();

        Thread stdoutReader = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                reader.lines().forEach(line -> stdout.append(line).append("\n"));
            } catch (IOException e) { }
        });

        Thread stderrReader = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                reader.lines().forEach(line -> stderr.append(line).append("\n"));
            } catch (IOException e) { }
        });

        stdoutReader.start();
        stderrReader.start();

        boolean completed = process.waitFor(120, TimeUnit.SECONDS);
        assertTrue(completed, "Process timed out");

        stdoutReader.join(5000);
        stderrReader.join(5000);

        return new CliResult(process.exitValue(), stdout.toString(), stderr.toString());
    }

    record CliResult(int exitCode, String stdout, String stderr) {
        boolean success() { return exitCode == 0; }
    }
}
