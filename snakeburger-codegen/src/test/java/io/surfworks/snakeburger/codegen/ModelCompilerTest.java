package io.surfworks.snakeburger.codegen;

import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelLoader;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.backend.cpu.CpuBackend;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for ModelCompiler.
 *
 * <p>Tests the programmatic compile() and compileToBytes() APIs,
 * as well as end-to-end compilation from MLIR to executable JAR.
 */
class ModelCompilerTest {

    private static final String SIMPLE_ADD_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
            stablehlo.return %0 : tensor<2x2xf32>
          }
        }
        """;

    private static final String MULTIPLY_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> (tensor<3x3xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<3x3xf32>
            stablehlo.return %0 : tensor<3x3xf32>
          }
        }
        """;

    private static final String MULTI_OP_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
            %1 = stablehlo.add %0, %arg2 : tensor<4xf32>
            stablehlo.return %1 : tensor<4xf32>
          }
        }
        """;

    private static final String MULTI_FUNCTION_MLIR = """
        module @main {
          func.func @helper(%x: tensor<2xf32>) -> (tensor<2xf32>) {
            %0 = stablehlo.negate %x : tensor<2xf32>
            stablehlo.return %0 : tensor<2xf32>
          }
          func.func public @forward(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
            stablehlo.return %0 : tensor<2xf32>
          }
        }
        """;

    @TempDir
    Path tempDir;

    @Test
    void testCompileToJar() throws Exception {
        Path jarPath = tempDir.resolve("model.jar");
        String className = "io.surfworks.warpforge.generated.CompiledAdd";

        ModelCompiler.compile(SIMPLE_ADD_MLIR, jarPath, className);

        assertTrue(Files.exists(jarPath));
        assertTrue(Files.size(jarPath) > 0);

        // Load and verify the model
        CompiledModel model = ModelLoader.load(jarPath);
        assertNotNull(model);
        assertEquals(2, model.inputCount());
        assertEquals(1, model.outputCount());
    }

    @Test
    void testCompileAndExecute() throws Exception {
        Path jarPath = tempDir.resolve("add_model.jar");
        String className = "io.surfworks.warpforge.generated.AddModel";

        ModelCompiler.compile(SIMPLE_ADD_MLIR, jarPath, className);

        CompiledModel model = ModelLoader.load(jarPath);

        try (var backend = new CpuBackend()) {
            Tensor input1 = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2);
            Tensor input2 = Tensor.fromFloatArray(new float[]{10, 20, 30, 40}, 2, 2);

            List<Tensor> outputs = model.forward(List.of(input1, input2), backend);

            assertEquals(1, outputs.size());
            float[] result = outputs.getFirst().toFloatArray();
            assertArrayEquals(new float[]{11, 22, 33, 44}, result, 0.001f);

            input1.close();
            input2.close();
            outputs.getFirst().close();
        }
    }

    @Test
    void testCompileMultiply() throws Exception {
        Path jarPath = tempDir.resolve("multiply_model.jar");
        String className = "io.surfworks.warpforge.generated.MulModel";

        ModelCompiler.compile(MULTIPLY_MLIR, jarPath, className);

        CompiledModel model = ModelLoader.load(jarPath);

        try (var backend = new CpuBackend()) {
            Tensor input1 = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
            Tensor input2 = Tensor.fromFloatArray(new float[]{2, 2, 2, 2, 2, 2, 2, 2, 2}, 3, 3);

            List<Tensor> outputs = model.forward(List.of(input1, input2), backend);

            assertEquals(1, outputs.size());
            float[] result = outputs.getFirst().toFloatArray();
            assertArrayEquals(new float[]{2, 4, 6, 8, 10, 12, 14, 16, 18}, result, 0.001f);

            input1.close();
            input2.close();
            outputs.getFirst().close();
        }
    }

    @Test
    void testCompileMultiOpModel() throws Exception {
        Path jarPath = tempDir.resolve("muladd_model.jar");
        String className = "io.surfworks.warpforge.generated.MulAddModel";

        ModelCompiler.compile(MULTI_OP_MLIR, jarPath, className);

        CompiledModel model = ModelLoader.load(jarPath);

        assertEquals(3, model.inputCount());
        assertEquals(1, model.outputCount());

        try (var backend = new CpuBackend()) {
            Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
            Tensor b = Tensor.fromFloatArray(new float[]{2, 3, 4, 5}, 4);
            Tensor c = Tensor.fromFloatArray(new float[]{10, 10, 10, 10}, 4);

            List<Tensor> outputs = model.forward(List.of(a, b, c), backend);

            // (1*2+10, 2*3+10, 3*4+10, 4*5+10) = (12, 16, 22, 30)
            float[] result = outputs.getFirst().toFloatArray();
            assertArrayEquals(new float[]{12, 16, 22, 30}, result, 0.001f);

            a.close();
            b.close();
            c.close();
            outputs.getFirst().close();
        }
    }

    @Test
    void testCompileToBytes() throws Exception {
        String className = "io.surfworks.warpforge.generated.BytesModel";

        var generated = ModelCompiler.compileToBytes(SIMPLE_ADD_MLIR, className);

        assertNotNull(generated);
        assertEquals(className, generated.className());
        assertTrue(generated.bytecode().length > 0);
        assertNotNull(generated.metadata());
        assertEquals("forward", generated.metadata().name());
    }

    @Test
    void testCompileToBytesAndLoad() throws Exception {
        String className = "io.surfworks.warpforge.generated.LoadableModel";

        var generated = ModelCompiler.compileToBytes(SIMPLE_ADD_MLIR, className);

        // Load model directly from bytecode
        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);

        assertNotNull(model);
        assertEquals(2, model.inputCount());
        assertEquals(1, model.outputCount());

        try (var backend = new CpuBackend()) {
            Tensor input1 = Tensor.fromFloatArray(new float[]{5, 5, 5, 5}, 2, 2);
            Tensor input2 = Tensor.fromFloatArray(new float[]{3, 3, 3, 3}, 2, 2);

            List<Tensor> outputs = model.forward(List.of(input1, input2), backend);

            float[] result = outputs.getFirst().toFloatArray();
            assertArrayEquals(new float[]{8, 8, 8, 8}, result, 0.001f);

            input1.close();
            input2.close();
            outputs.getFirst().close();
        }
    }

    @Test
    void testCompileSelectsPublicFunction() throws Exception {
        Path jarPath = tempDir.resolve("public_func_model.jar");
        String className = "io.surfworks.warpforge.generated.PublicFuncModel";

        ModelCompiler.compile(MULTI_FUNCTION_MLIR, jarPath, className);

        CompiledModel model = ModelLoader.load(jarPath);

        // Should select the public "forward" function, not "helper"
        assertEquals("forward", model.metadata().name());
        assertEquals(2, model.inputCount());
        assertEquals(1, model.outputCount());
    }

    @Test
    void testCompileFromFile() throws Exception {
        // Write MLIR to file
        Path mlirFile = tempDir.resolve("input.mlir");
        Files.writeString(mlirFile, SIMPLE_ADD_MLIR, StandardCharsets.UTF_8);

        Path jarPath = tempDir.resolve("from_file.jar");
        String className = "io.surfworks.warpforge.generated.FromFile";

        // Use run() with CompilerArgs
        ModelCompiler.CompilerArgs args = new ModelCompiler.CompilerArgs(
            mlirFile, jarPath, className, null);

        ModelCompiler.run(args);

        assertTrue(Files.exists(jarPath));
        CompiledModel model = ModelLoader.load(jarPath);
        assertEquals(2, model.inputCount());
    }

    @Test
    void testCompileWithSpecificFunction() throws Exception {
        Path mlirFile = tempDir.resolve("multi_func.mlir");
        Files.writeString(mlirFile, MULTI_FUNCTION_MLIR, StandardCharsets.UTF_8);

        Path jarPath = tempDir.resolve("helper_func.jar");
        String className = "io.surfworks.warpforge.generated.HelperModel";

        // Explicitly compile the "helper" function
        ModelCompiler.CompilerArgs args = new ModelCompiler.CompilerArgs(
            mlirFile, jarPath, className, "helper");

        ModelCompiler.run(args);

        CompiledModel model = ModelLoader.load(jarPath);
        assertEquals("helper", model.metadata().name());
        assertEquals(1, model.inputCount());  // helper has only 1 input
    }

    @Test
    void testCompileInvalidMlirThrows() {
        String invalidMlir = "this is not valid MLIR";
        Path jarPath = tempDir.resolve("invalid.jar");

        assertThrows(Exception.class, () -> {
            ModelCompiler.compile(invalidMlir, jarPath, "io.test.Invalid");
        });
    }

    @Test
    void testCompileEmptyModuleThrows() {
        String emptyModule = "module @empty {}";
        Path jarPath = tempDir.resolve("empty.jar");

        assertThrows(IllegalArgumentException.class, () -> {
            ModelCompiler.compile(emptyModule, jarPath, "io.test.Empty");
        });
    }

    @Test
    void testCompileMissingFunctionThrows() throws Exception {
        Path mlirFile = tempDir.resolve("test.mlir");
        Files.writeString(mlirFile, SIMPLE_ADD_MLIR, StandardCharsets.UTF_8);

        Path jarPath = tempDir.resolve("missing_func.jar");

        ModelCompiler.CompilerArgs args = new ModelCompiler.CompilerArgs(
            mlirFile, jarPath, "io.test.Missing", "nonexistent_function");

        assertThrows(IllegalArgumentException.class, () -> {
            ModelCompiler.run(args);
        });
    }

    @Test
    void testMetadataPreserved() throws Exception {
        Path jarPath = tempDir.resolve("metadata_test.jar");
        String className = "io.surfworks.warpforge.generated.MetadataModel";

        ModelCompiler.compile(SIMPLE_ADD_MLIR, jarPath, className);

        CompiledModel model = ModelLoader.load(jarPath);

        assertNotNull(model.metadata());
        assertEquals("forward", model.metadata().name());
        assertNotNull(model.metadata().sourceHash());
        assertTrue(model.metadata().sourceHash().length() == 64); // SHA-256 hex
        assertTrue(model.metadata().generatedAt() > 0);
        assertEquals(ModelClassGenerator.GENERATOR_VERSION, model.metadata().generatorVersion());
    }

    @Test
    void testDifferentClassNames() throws Exception {
        String[] classNames = {
            "io.surfworks.warpforge.generated.Model1",
            "com.example.MyModel",
            "TestModel"
        };

        for (String className : classNames) {
            Path jarPath = tempDir.resolve(className.replace('.', '_') + ".jar");

            ModelCompiler.compile(SIMPLE_ADD_MLIR, jarPath, className);

            CompiledModel model = ModelLoader.load(jarPath);
            assertNotNull(model);
            assertEquals(2, model.inputCount());
        }
    }

    @Test
    void testCompiledModelIsExecutableMultipleTimes() throws Exception {
        Path jarPath = tempDir.resolve("reusable.jar");
        String className = "io.surfworks.warpforge.generated.Reusable";

        ModelCompiler.compile(SIMPLE_ADD_MLIR, jarPath, className);
        CompiledModel model = ModelLoader.load(jarPath);

        try (var backend = new CpuBackend()) {
            // Execute multiple times with different inputs
            for (int i = 0; i < 5; i++) {
                Tensor input1 = Tensor.fromFloatArray(new float[]{i, i, i, i}, 2, 2);
                Tensor input2 = Tensor.fromFloatArray(new float[]{1, 1, 1, 1}, 2, 2);

                List<Tensor> outputs = model.forward(List.of(input1, input2), backend);

                float[] result = outputs.getFirst().toFloatArray();
                float expected = i + 1;
                assertArrayEquals(new float[]{expected, expected, expected, expected}, result, 0.001f);

                input1.close();
                input2.close();
                outputs.getFirst().close();
            }
        }
    }
}
