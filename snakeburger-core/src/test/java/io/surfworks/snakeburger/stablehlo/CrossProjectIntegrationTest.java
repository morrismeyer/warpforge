package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConcatenateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConstantOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConvolutionOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotGeneralOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogisticOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MaximumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NegateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceWindowOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanhOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TransposeOp;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end cross-project integration tests.
 *
 * These tests exercise the full PyTorch developer workflow:
 * 1. Define a PyTorch model (inline or from file)
 * 2. Trace it with snakegrinder native image
 * 3. Parse the MLIR output with snakeburger
 * 4. Validate the structure and operations
 *
 * This is the primary integration point for PyTorch developers
 * getting started with WarpForge.
 *
 * Prerequisites:
 * - Run: ./gradlew :snakegrinder-dist:assembleDist
 *
 * Run: ./gradlew :snakeburger-core:crossProjectTest
 */
@Tag("integration")
@Tag("cross-project")
class CrossProjectIntegrationTest {

    private static Path snakegrinderBinary;
    private static Path tempDir;
    private static Path projectRoot;

    /**
     * Check if the native image binary exists.
     */
    static boolean isNativeImageAvailable() {
        snakegrinderBinary = findSnakegrinderBinary();
        return snakegrinderBinary != null && Files.isExecutable(snakegrinderBinary);
    }

    private static Path findSnakegrinderBinary() {
        projectRoot = findProjectRoot();
        if (projectRoot == null) {
            return null;
        }

        // Check macOS .app bundle first
        Path appBinary = projectRoot.resolve("snakegrinder-dist/build/SnakeGrinder.app/Contents/MacOS/snakegrinder");
        if (Files.isExecutable(appBinary)) {
            return appBinary;
        }

        // Check assembleDist output - use wrapper script (not binary directly) to get env setup
        Path distWrapper = projectRoot.resolve("snakegrinder-dist/build/dist/bin/snakegrinder");
        if (Files.isExecutable(distWrapper)) {
            return distWrapper;
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
        while (current != null) {
            if (Files.exists(current.resolve("settings.gradle"))) {
                return current;
            }
            current = current.getParent();
        }
        return Paths.get(System.getProperty("user.dir"));
    }

    @BeforeAll
    static void setup() throws IOException {
        tempDir = Files.createTempDirectory("cross-project-test");
    }

    @AfterAll
    static void cleanup() throws IOException {
        if (tempDir != null && Files.exists(tempDir)) {
            Files.walk(tempDir)
                    .sorted((a, b) -> b.compareTo(a))
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
    }

    /**
     * Trace a model and return the parsed Module.
     */
    private Module traceAndParse(String modelCode, String className, String inputSpecs, String testName)
            throws IOException, InterruptedException {

        Path modelFile = tempDir.resolve(testName + "_model.py");
        Files.writeString(modelFile, modelCode);

        Path outputDir = tempDir.resolve(testName + "_output");
        Files.createDirectories(outputDir);

        CliResult result = runSnakegrinder(
                "--trace",
                "--source", modelFile.toString(),
                "--class", className,
                "--inputs", inputSpecs,
                "--out", outputDir.toString());

        assertTrue(result.success(),
                "Trace should succeed for " + className + ": " + result.stderr());

        Path mlirFile = outputDir.resolve("model.mlir");
        assertTrue(Files.exists(mlirFile),
                "MLIR file should be created for " + className);

        String mlir = Files.readString(mlirFile);
        assertFalse(mlir.isBlank(), "MLIR should not be empty for " + className);

        Module module = StableHloParser.parse(mlir);
        assertNotNull(module, "Parser should return a module for " + className);

        return module;
    }

    // ========================================================================
    // Simple Model Tests - Basic Operations
    // ========================================================================

    @Nested
    @DisplayName("Simple Models")
    class SimpleModelTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Add: x + y → stablehlo.add")
        void addModel() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class AddModel(nn.Module):
                    def forward(self, x, y):
                        return x + y
                """;

            Module module = traceAndParse(model, "AddModel", "[(1,8),(1,8)]", "add");

            assertEquals("main", module.name());
            Function forward = module.functions().get(0);
            assertEquals(2, forward.arguments().size(), "Should have 2 inputs");

            boolean hasAdd = forward.body().stream()
                    .anyMatch(op -> op instanceof AddOp);
            assertTrue(hasAdd, "x + y should produce AddOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Multiply: x * y → stablehlo.multiply")
        void multiplyModel() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class MulModel(nn.Module):
                    def forward(self, x, y):
                        return x * y
                """;

            Module module = traceAndParse(model, "MulModel", "[(1,8),(1,8)]", "mul");

            Function forward = module.functions().get(0);
            boolean hasMul = forward.body().stream()
                    .anyMatch(op -> op instanceof MultiplyOp);
            assertTrue(hasMul, "x * y should produce MultiplyOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Negate: -x → stablehlo.negate")
        void negateModel() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class NegModel(nn.Module):
                    def forward(self, x):
                        return -x
                """;

            Module module = traceAndParse(model, "NegModel", "[(1,8)]", "neg");

            Function forward = module.functions().get(0);
            boolean hasNeg = forward.body().stream()
                    .anyMatch(op -> op instanceof NegateOp);
            assertTrue(hasNeg, "-x should produce NegateOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Exp: torch.exp(x) → stablehlo.exponential")
        void expModel() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class ExpModel(nn.Module):
                    def forward(self, x):
                        return torch.exp(x)
                """;

            Module module = traceAndParse(model, "ExpModel", "[(1,8)]", "exp");

            Function forward = module.functions().get(0);
            boolean hasExp = forward.body().stream()
                    .anyMatch(op -> op instanceof ExpOp);
            assertTrue(hasExp, "torch.exp should produce ExpOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Tanh: torch.tanh(x) → stablehlo.tanh")
        void tanhModel() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class TanhModel(nn.Module):
                    def forward(self, x):
                        return torch.tanh(x)
                """;

            Module module = traceAndParse(model, "TanhModel", "[(1,8)]", "tanh");

            Function forward = module.functions().get(0);
            boolean hasTanh = forward.body().stream()
                    .anyMatch(op -> op instanceof TanhOp);
            assertTrue(hasTanh, "torch.tanh should produce TanhOp");
        }
    }

    // ========================================================================
    // Linear Layer Tests
    // ========================================================================

    @Nested
    @DisplayName("Linear Layers")
    class LinearLayerTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Linear → stablehlo.dot_general + add")
        void linearLayer() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class LinearModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc = nn.Linear(8, 4)

                    def forward(self, x):
                        return self.fc(x)
                """;

            Module module = traceAndParse(model, "LinearModel", "[(1,8)]", "linear");

            Function forward = module.functions().get(0);
            assertEquals(1, forward.arguments().size(), "Should have 1 input");

            boolean hasDotGeneral = forward.body().stream()
                    .anyMatch(op -> op instanceof DotGeneralOp);
            assertTrue(hasDotGeneral, "nn.Linear should produce DotGeneralOp");

            boolean hasAdd = forward.body().stream()
                    .anyMatch(op -> op instanceof AddOp);
            assertTrue(hasAdd, "nn.Linear with bias should produce AddOp");

            long constantCount = forward.body().stream()
                    .filter(op -> op instanceof ConstantOp).count();
            assertTrue(constantCount >= 2, "Should have constants for weight and bias");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Linear(bias=False) → stablehlo.dot_general only")
        void linearLayerNoBias() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class LinearNoBias(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc = nn.Linear(8, 4, bias=False)

                    def forward(self, x):
                        return self.fc(x)
                """;

            Module module = traceAndParse(model, "LinearNoBias", "[(1,8)]", "linear_no_bias");

            Function forward = module.functions().get(0);

            boolean hasDotGeneral = forward.body().stream()
                    .anyMatch(op -> op instanceof DotGeneralOp);
            assertTrue(hasDotGeneral, "nn.Linear should produce DotGeneralOp");

            // Without bias, should have only weight constant
            long constantCount = forward.body().stream()
                    .filter(op -> op instanceof ConstantOp).count();
            assertEquals(1, constantCount, "Linear without bias should have 1 constant (weight)");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("torch.matmul → stablehlo.dot_general")
        void matmul() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class MatmulModel(nn.Module):
                    def forward(self, a, b):
                        return torch.matmul(a, b)
                """;

            Module module = traceAndParse(model, "MatmulModel", "[(2,4,8),(2,8,4)]", "matmul");

            Function forward = module.functions().get(0);
            assertEquals(2, forward.arguments().size(), "Should have 2 matrix inputs");

            boolean hasDotGeneral = forward.body().stream()
                    .anyMatch(op -> op instanceof DotGeneralOp);
            assertTrue(hasDotGeneral, "torch.matmul should produce DotGeneralOp");
        }
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    @Nested
    @DisplayName("Activation Functions")
    class ActivationTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.ReLU → stablehlo.maximum")
        void reluActivation() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class ReLUModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.relu = nn.ReLU()

                    def forward(self, x):
                        return self.relu(x)
                """;

            Module module = traceAndParse(model, "ReLUModel", "[(1,8)]", "relu");

            Function forward = module.functions().get(0);

            boolean hasMaximum = forward.body().stream()
                    .anyMatch(op -> op instanceof MaximumOp);
            assertTrue(hasMaximum, "ReLU should produce MaximumOp (max(x, 0))");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("F.relu → stablehlo.maximum")
        void functionalRelu() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn
                import torch.nn.functional as F

                class FunctionalReLU(nn.Module):
                    def forward(self, x):
                        return F.relu(x)
                """;

            Module module = traceAndParse(model, "FunctionalReLU", "[(1,8)]", "frelu");

            Function forward = module.functions().get(0);

            boolean hasMaximum = forward.body().stream()
                    .anyMatch(op -> op instanceof MaximumOp);
            assertTrue(hasMaximum, "F.relu should produce MaximumOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Sigmoid → stablehlo.logistic")
        void sigmoidActivation() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class SigmoidModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.sigmoid = nn.Sigmoid()

                    def forward(self, x):
                        return self.sigmoid(x)
                """;

            Module module = traceAndParse(model, "SigmoidModel", "[(1,8)]", "sigmoid");

            Function forward = module.functions().get(0);

            boolean hasLogistic = forward.body().stream()
                    .anyMatch(op -> op instanceof LogisticOp);
            assertTrue(hasLogistic, "Sigmoid should produce LogisticOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Tanh module → stablehlo.tanh")
        void tanhModule() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class TanhModule(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.tanh = nn.Tanh()

                    def forward(self, x):
                        return self.tanh(x)
                """;

            Module module = traceAndParse(model, "TanhModule", "[(1,8)]", "tanh_mod");

            Function forward = module.functions().get(0);

            boolean hasTanh = forward.body().stream()
                    .anyMatch(op -> op instanceof TanhOp);
            assertTrue(hasTanh, "nn.Tanh should produce TanhOp");
        }
    }

    // ========================================================================
    // Convolution Operations
    // ========================================================================

    @Nested
    @DisplayName("Convolution")
    class ConvolutionTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Conv2d → stablehlo.convolution")
        void conv2d() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class Conv2dModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

                    def forward(self, x):
                        return self.conv(x)
                """;

            Module module = traceAndParse(model, "Conv2dModel", "[(1,3,8,8)]", "conv2d");

            Function forward = module.functions().get(0);

            boolean hasConv = forward.body().stream()
                    .anyMatch(op -> op instanceof ConvolutionOp);
            assertTrue(hasConv, "nn.Conv2d should produce ConvolutionOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Conv2d(bias=False) → stablehlo.convolution only")
        void conv2dNoBias() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class Conv2dNoBias(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

                    def forward(self, x):
                        return self.conv(x)
                """;

            Module module = traceAndParse(model, "Conv2dNoBias", "[(1,3,8,8)]", "conv2d_no_bias");

            Function forward = module.functions().get(0);

            boolean hasConv = forward.body().stream()
                    .anyMatch(op -> op instanceof ConvolutionOp);
            assertTrue(hasConv, "nn.Conv2d should produce ConvolutionOp");

            // Without bias, should have only kernel constant
            long constantCount = forward.body().stream()
                    .filter(op -> op instanceof ConstantOp).count();
            assertEquals(1, constantCount, "Conv2d without bias should have 1 constant (kernel)");
        }
    }

    // ========================================================================
    // Pooling Operations
    // ========================================================================

    @Nested
    @DisplayName("Pooling")
    class PoolingTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.MaxPool2d → stablehlo.reduce_window")
        void maxPool2d() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class MaxPoolModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

                    def forward(self, x):
                        return self.pool(x)
                """;

            Module module = traceAndParse(model, "MaxPoolModel", "[(1,4,8,8)]", "maxpool");

            Function forward = module.functions().get(0);

            boolean hasReduceWindow = forward.body().stream()
                    .anyMatch(op -> op instanceof ReduceWindowOp);
            assertTrue(hasReduceWindow, "MaxPool2d should produce ReduceWindowOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.AvgPool2d → stablehlo.reduce_window")
        void avgPool2d() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class AvgPoolModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

                    def forward(self, x):
                        return self.pool(x)
                """;

            Module module = traceAndParse(model, "AvgPoolModel", "[(1,4,8,8)]", "avgpool");

            Function forward = module.functions().get(0);

            // AvgPool becomes reduce_window + divide
            boolean hasReduceWindow = forward.body().stream()
                    .anyMatch(op -> op instanceof ReduceWindowOp);
            assertTrue(hasReduceWindow, "AvgPool2d should produce ReduceWindowOp");
        }
    }

    // ========================================================================
    // Shape Operations
    // ========================================================================

    @Nested
    @DisplayName("Shape Operations")
    class ShapeTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("tensor.reshape → stablehlo.reshape")
        void reshape() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class ReshapeModel(nn.Module):
                    def forward(self, x):
                        return x.reshape(1, 16)
                """;

            Module module = traceAndParse(model, "ReshapeModel", "[(2,8)]", "reshape");

            Function forward = module.functions().get(0);

            boolean hasReshape = forward.body().stream()
                    .anyMatch(op -> op instanceof ReshapeOp);
            assertTrue(hasReshape, "tensor.reshape should produce ReshapeOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("tensor.transpose → stablehlo.transpose")
        void transpose() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class TransposeModel(nn.Module):
                    def forward(self, x):
                        return x.transpose(0, 1)
                """;

            Module module = traceAndParse(model, "TransposeModel", "[(4,8)]", "transpose");

            Function forward = module.functions().get(0);

            boolean hasTranspose = forward.body().stream()
                    .anyMatch(op -> op instanceof TransposeOp);
            assertTrue(hasTranspose, "tensor.transpose should produce TransposeOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("torch.cat → stablehlo.concatenate")
        void concatenate() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class CatModel(nn.Module):
                    def forward(self, x, y):
                        return torch.cat([x, y], dim=1)
                """;

            Module module = traceAndParse(model, "CatModel", "[(1,4),(1,4)]", "concat");

            Function forward = module.functions().get(0);

            boolean hasConcatenate = forward.body().stream()
                    .anyMatch(op -> op instanceof ConcatenateOp);
            assertTrue(hasConcatenate, "torch.cat should produce ConcatenateOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("nn.Flatten → stablehlo.reshape")
        void flatten() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class FlattenModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.flatten = nn.Flatten()

                    def forward(self, x):
                        return self.flatten(x)
                """;

            Module module = traceAndParse(model, "FlattenModel", "[(1,4,8,8)]", "flatten");

            Function forward = module.functions().get(0);

            boolean hasReshape = forward.body().stream()
                    .anyMatch(op -> op instanceof ReshapeOp);
            assertTrue(hasReshape, "nn.Flatten should produce ReshapeOp");
        }
    }

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    @Nested
    @DisplayName("Reductions")
    class ReductionTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("torch.sum → stablehlo.reduce with add")
        void sumReduction() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class SumModel(nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)
                """;

            Module module = traceAndParse(model, "SumModel", "[(2,8)]", "sum");

            Function forward = module.functions().get(0);

            boolean hasReduce = forward.body().stream()
                    .anyMatch(op -> op instanceof ReduceOp);
            assertTrue(hasReduce, "torch.sum should produce ReduceOp");
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("torch.amax(dim=...) → stablehlo.reduce with max")
        void maxReduction() throws IOException, InterruptedException {
            // Note: torch.max(dim=...).values involves getattr which isn't supported,
            // so we use torch.amax which directly returns the reduced values
            String model = """
                import torch
                import torch.nn as nn

                class MaxModel(nn.Module):
                    def forward(self, x):
                        return torch.amax(x, dim=1)
                """;

            Module module = traceAndParse(model, "MaxModel", "[(2,8)]", "max");

            Function forward = module.functions().get(0);

            boolean hasReduce = forward.body().stream()
                    .anyMatch(op -> op instanceof ReduceOp);
            assertTrue(hasReduce, "torch.amax should produce ReduceOp");
        }
    }

    // ========================================================================
    // Composite Architectures - The Main Workflow
    // ========================================================================

    @Nested
    @DisplayName("Common Architectures")
    class ArchitectureTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Simple MLP: Linear → ReLU → Linear")
        void simpleMLP() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class SimpleMLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = nn.Linear(8, 16)
                        self.relu = nn.ReLU()
                        self.fc2 = nn.Linear(16, 4)

                    def forward(self, x):
                        x = self.fc1(x)
                        x = self.relu(x)
                        x = self.fc2(x)
                        return x
                """;

            Module module = traceAndParse(model, "SimpleMLP", "[(1,8)]", "mlp");

            Function forward = module.functions().get(0);

            long dotGeneralCount = forward.body().stream()
                    .filter(op -> op instanceof DotGeneralOp).count();
            assertEquals(2, dotGeneralCount, "MLP should have 2 dot_general ops (2 linear layers)");

            boolean hasMaximum = forward.body().stream()
                    .anyMatch(op -> op instanceof MaximumOp);
            assertTrue(hasMaximum, "MLP should have maximum op (ReLU)");

            System.out.println("SimpleMLP ops: " + forward.body().size());
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Deep MLP: 4 layers with ReLU")
        void deepMLP() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class DeepMLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = nn.Linear(8, 32)
                        self.fc2 = nn.Linear(32, 32)
                        self.fc3 = nn.Linear(32, 16)
                        self.fc4 = nn.Linear(16, 4)

                    def forward(self, x):
                        x = torch.relu(self.fc1(x))
                        x = torch.relu(self.fc2(x))
                        x = torch.relu(self.fc3(x))
                        x = self.fc4(x)
                        return x
                """;

            Module module = traceAndParse(model, "DeepMLP", "[(1,8)]", "deep_mlp");

            Function forward = module.functions().get(0);

            long dotGeneralCount = forward.body().stream()
                    .filter(op -> op instanceof DotGeneralOp).count();
            assertEquals(4, dotGeneralCount, "Deep MLP should have 4 dot_general ops");

            long maximumCount = forward.body().stream()
                    .filter(op -> op instanceof MaximumOp).count();
            assertEquals(3, maximumCount, "Deep MLP should have 3 ReLU activations");

            System.out.println("DeepMLP ops: " + forward.body().size());
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Residual Block: x + fc2(relu(fc1(x)))")
        void residualBlock() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class ResidualBlock(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.fc1 = nn.Linear(8, 8)
                        self.fc2 = nn.Linear(8, 8)

                    def forward(self, x):
                        residual = x
                        out = torch.relu(self.fc1(x))
                        out = self.fc2(out)
                        return out + residual
                """;

            Module module = traceAndParse(model, "ResidualBlock", "[(1,8)]", "residual");

            Function forward = module.functions().get(0);

            long dotGeneralCount = forward.body().stream()
                    .filter(op -> op instanceof DotGeneralOp).count();
            assertEquals(2, dotGeneralCount, "Residual block should have 2 linear layers");

            // Should have adds for: fc1 bias, fc2 bias, residual connection
            long addCount = forward.body().stream()
                    .filter(op -> op instanceof AddOp).count();
            assertTrue(addCount >= 3, "Residual block should have at least 3 adds");

            System.out.println("ResidualBlock ops: " + forward.body().size());
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Simple CNN: Conv → ReLU → Pool → Flatten → Linear")
        void simpleCNN() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class SimpleCNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                        self.relu = nn.ReLU()
                        self.pool = nn.MaxPool2d(2, 2)
                        self.flatten = nn.Flatten()
                        self.fc = nn.Linear(8 * 4 * 4, 10)

                    def forward(self, x):
                        x = self.conv(x)
                        x = self.relu(x)
                        x = self.pool(x)
                        x = self.flatten(x)
                        x = self.fc(x)
                        return x
                """;

            Module module = traceAndParse(model, "SimpleCNN", "[(1,3,8,8)]", "cnn");

            Function forward = module.functions().get(0);

            boolean hasConv = forward.body().stream()
                    .anyMatch(op -> op instanceof ConvolutionOp);
            assertTrue(hasConv, "CNN should have convolution");

            boolean hasMaximum = forward.body().stream()
                    .anyMatch(op -> op instanceof MaximumOp);
            assertTrue(hasMaximum, "CNN should have ReLU");

            boolean hasReduceWindow = forward.body().stream()
                    .anyMatch(op -> op instanceof ReduceWindowOp);
            assertTrue(hasReduceWindow, "CNN should have pooling");

            boolean hasReshape = forward.body().stream()
                    .anyMatch(op -> op instanceof ReshapeOp);
            assertTrue(hasReshape, "CNN should have flatten (reshape)");

            boolean hasDotGeneral = forward.body().stream()
                    .anyMatch(op -> op instanceof DotGeneralOp);
            assertTrue(hasDotGeneral, "CNN should have linear layer");

            System.out.println("SimpleCNN ops: " + forward.body().size());
        }

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("Autoencoder: Encoder + Decoder")
        void autoencoder() throws IOException, InterruptedException {
            String model = """
                import torch
                import torch.nn as nn

                class Autoencoder(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # Encoder
                        self.enc1 = nn.Linear(16, 8)
                        self.enc2 = nn.Linear(8, 4)
                        # Decoder
                        self.dec1 = nn.Linear(4, 8)
                        self.dec2 = nn.Linear(8, 16)

                    def forward(self, x):
                        # Encode
                        x = torch.relu(self.enc1(x))
                        x = torch.relu(self.enc2(x))
                        # Decode
                        x = torch.relu(self.dec1(x))
                        x = self.dec2(x)
                        return x
                """;

            Module module = traceAndParse(model, "Autoencoder", "[(1,16)]", "autoencoder");

            Function forward = module.functions().get(0);

            long dotGeneralCount = forward.body().stream()
                    .filter(op -> op instanceof DotGeneralOp).count();
            assertEquals(4, dotGeneralCount, "Autoencoder should have 4 linear layers");

            long maximumCount = forward.body().stream()
                    .filter(op -> op instanceof MaximumOp).count();
            assertEquals(3, maximumCount, "Autoencoder should have 3 ReLU activations");

            System.out.println("Autoencoder ops: " + forward.body().size());
        }
    }

    // ========================================================================
    // Built-in Example Test
    // ========================================================================

    @Nested
    @DisplayName("Built-in Examples")
    class BuiltInExampleTests {

        @Test
        @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
        @DisplayName("--trace-example produces parseable MLIR")
        void traceExampleIsParseable() throws IOException, InterruptedException {
            Path outputDir = tempDir.resolve("builtin_example");
            Files.createDirectories(outputDir);

            CliResult result = runSnakegrinder("--trace-example", "--out", outputDir.toString());
            assertTrue(result.success(), "Built-in example should succeed: " + result.stderr());

            String mlir = Files.readString(outputDir.resolve("model.mlir"));

            Module module = StableHloParser.parse(mlir);
            assertNotNull(module, "Built-in example should parse successfully");
            assertEquals("main", module.name());
            assertFalse(module.functions().isEmpty(), "Should have functions");

            System.out.println("Built-in example module: " + module.name());
            System.out.println("Functions: " + module.functions().size());
            System.out.println("Ops in forward: " + module.functions().get(0).body().size());
        }
    }

    // ========================================================================
    // Cross-Project Compatibility Report
    // ========================================================================

    @Test
    @EnabledIf("io.surfworks.snakeburger.stablehlo.CrossProjectIntegrationTest#isNativeImageAvailable")
    @DisplayName("Full Workflow Report")
    void fullWorkflowReport() throws IOException, InterruptedException {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║     Cross-Project Integration Test: Full Workflow Report         ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.println("║ snakegrinder binary: " + snakegrinderBinary);
        System.out.println("╟──────────────────────────────────────────────────────────────────╢");

        // Trace and report
        Path outputDir = tempDir.resolve("full_workflow_test");
        Files.createDirectories(outputDir);

        String model = """
            import torch
            import torch.nn as nn

            class FullWorkflowModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(8, 16)
                    self.fc2 = nn.Linear(16, 4)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    return self.fc2(x)
            """;

        Path modelFile = tempDir.resolve("full_workflow_model.py");
        Files.writeString(modelFile, model);

        CliResult result = runSnakegrinder(
                "--trace",
                "--source", modelFile.toString(),
                "--class", "FullWorkflowModel",
                "--inputs", "[(1,8)]",
                "--out", outputDir.toString());

        System.out.println("║ Step 1: Trace PyTorch model");
        System.out.println("║   Exit code: " + result.exitCode());
        System.out.println("║   Status: " + (result.success() ? "✓ SUCCESS" : "✗ FAILED"));

        if (result.success()) {
            String mlir = Files.readString(outputDir.resolve("model.mlir"));
            System.out.println("║   MLIR size: " + mlir.length() + " bytes");

            System.out.println("╟──────────────────────────────────────────────────────────────────╢");
            System.out.println("║ Step 2: Parse with SnakeBurger");

            Module module = StableHloParser.parse(mlir);
            System.out.println("║   Module name: " + module.name());
            System.out.println("║   Functions: " + module.functions().size());

            Function forward = module.functions().get(0);
            System.out.println("║   Operations: " + forward.body().size());

            System.out.println("╟──────────────────────────────────────────────────────────────────╢");
            System.out.println("║ Step 3: Analyze Operations");

            var opCounts = forward.body().stream()
                    .collect(java.util.stream.Collectors.groupingBy(
                            op -> op.getClass().getSimpleName(),
                            java.util.stream.Collectors.counting()));

            opCounts.forEach((name, count) ->
                    System.out.println("║   " + name + ": " + count));

            System.out.println("╟──────────────────────────────────────────────────────────────────╢");
            System.out.println("║ Step 4: Type Check");

            StableHloTypeChecker checker = new StableHloTypeChecker();
            var errors = checker.validate(module);
            System.out.println("║   Type errors: " + errors.size());
            System.out.println("║   Status: " + (errors.isEmpty() ? "✓ VALID" : "⚠ WARNINGS"));
        }

        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }
}
