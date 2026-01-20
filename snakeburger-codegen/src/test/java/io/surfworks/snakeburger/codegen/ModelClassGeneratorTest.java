package io.surfworks.snakeburger.codegen;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelLoader;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.backend.cpu.CpuBackend;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelClassGeneratorTest {

    // Simple MLIR for adding two 2x2 tensors
    private static final String SIMPLE_ADD_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
            stablehlo.return %0 : tensor<2x2xf32>
          }
        }
        """;

    @Test
    void testGenerateSimpleAddModel() throws Exception {
        // Parse MLIR
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        // Generate bytecode
        String className = "io.surfworks.warpforge.generated.TestAddModel";
        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        // Verify generation
        assertNotNull(generated);
        assertEquals(className, generated.className());
        assertTrue(generated.bytecode().length > 0);
        assertNotNull(generated.metadata());
        assertEquals("forward", generated.metadata().name());
    }

    @Test
    void testLoadAndExecuteGeneratedModel() throws Exception {
        // Parse MLIR
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        // Generate bytecode
        String className = "io.surfworks.warpforge.generated.TestModel";
        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        // Load the model from bytecode
        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);

        // Verify model properties
        assertNotNull(model);
        assertEquals(2, model.inputCount());
        assertEquals(1, model.outputCount());
        assertEquals("forward", model.metadata().name());

        // Execute with test inputs
        try (Backend backend = new CpuBackend()) {
            Tensor input1 = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2);
            Tensor input2 = Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 2, 2);

            List<Tensor> outputs = model.forward(List.of(input1, input2), backend);

            assertEquals(1, outputs.size());
            Tensor output = outputs.getFirst();

            // Verify result: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
            float[] result = output.toFloatArray();
            assertEquals(4, result.length);
            assertEquals(6.0f, result[0], 0.001f);
            assertEquals(8.0f, result[1], 0.001f);
            assertEquals(10.0f, result[2], 0.001f);
            assertEquals(12.0f, result[3], 0.001f);

            input1.close();
            input2.close();
            output.close();
        }
    }

    @Test
    void testGeneratedMetadata() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.MetadataTest";
        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        assertNotNull(generated.metadata());
        assertEquals("forward", generated.metadata().name());
        assertTrue(generated.metadata().generatedAt() > 0);
        assertEquals(ModelClassGenerator.GENERATOR_VERSION, generated.metadata().generatorVersion());
        assertNotNull(generated.metadata().sourceHash());
        assertEquals(64, generated.metadata().sourceHash().length()); // SHA-256 hex
    }

    // ========================
    // Multi-operation tests
    // ========================

    private static final String MULTI_OP_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
            %1 = stablehlo.add %0, %arg2 : tensor<4xf32>
            stablehlo.return %1 : tensor<4xf32>
          }
        }
        """;

    private static final String CHAIN_OP_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<3xf32>) -> (tensor<3xf32>) {
            %0 = stablehlo.negate %arg0 : tensor<3xf32>
            %1 = stablehlo.abs %0 : tensor<3xf32>
            %2 = stablehlo.exponential %1 : tensor<3xf32>
            stablehlo.return %2 : tensor<3xf32>
          }
        }
        """;

    private static final String FIVE_OP_MLIR = """
        module @main {
          func.func public @forward(%a: tensor<2xf32>, %b: tensor<2xf32>, %c: tensor<2xf32>) -> (tensor<2xf32>) {
            %0 = stablehlo.add %a, %b : tensor<2xf32>
            %1 = stablehlo.multiply %0, %c : tensor<2xf32>
            %2 = stablehlo.subtract %1, %a : tensor<2xf32>
            %3 = stablehlo.negate %2 : tensor<2xf32>
            %4 = stablehlo.abs %3 : tensor<2xf32>
            stablehlo.return %4 : tensor<2xf32>
          }
        }
        """;

    @Test
    void testGenerateMultiOpModel() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(MULTI_OP_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.MultiOpModel";
        var generated = ModelClassGenerator.generate(className, function, MULTI_OP_MLIR);

        assertNotNull(generated);
        assertTrue(generated.bytecode().length > 0);

        // Load and execute
        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);
        assertEquals(3, model.inputCount());
        assertEquals(1, model.outputCount());

        try (Backend backend = new CpuBackend()) {
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
    void testGenerateChainedUnaryOps() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(CHAIN_OP_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.ChainModel";
        var generated = ModelClassGenerator.generate(className, function, CHAIN_OP_MLIR);

        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);
        assertEquals(1, model.inputCount());
        assertEquals(1, model.outputCount());

        try (Backend backend = new CpuBackend()) {
            Tensor input = Tensor.fromFloatArray(new float[]{-1, 0, 1}, 3);

            List<Tensor> outputs = model.forward(List.of(input), backend);

            // negate: [1, 0, -1]
            // abs: [1, 0, 1]
            // exp: [e, 1, e]
            float[] result = outputs.getFirst().toFloatArray();
            assertEquals(Math.exp(1), result[0], 0.001f);
            assertEquals(1.0f, result[1], 0.001f);
            assertEquals(Math.exp(1), result[2], 0.001f);

            input.close();
            outputs.getFirst().close();
        }
    }

    @Test
    void testGenerateFiveOpModel() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(FIVE_OP_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.FiveOpModel";
        var generated = ModelClassGenerator.generate(className, function, FIVE_OP_MLIR);

        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);
        assertEquals(3, model.inputCount());

        try (Backend backend = new CpuBackend()) {
            Tensor a = Tensor.fromFloatArray(new float[]{2, 4}, 2);
            Tensor b = Tensor.fromFloatArray(new float[]{3, 5}, 2);
            Tensor c = Tensor.fromFloatArray(new float[]{2, 2}, 2);

            List<Tensor> outputs = model.forward(List.of(a, b, c), backend);

            // (a+b) = [5, 9]
            // (a+b)*c = [10, 18]
            // ((a+b)*c)-a = [8, 14]
            // negate: [-8, -14]
            // abs: [8, 14]
            float[] result = outputs.getFirst().toFloatArray();
            assertArrayEquals(new float[]{8, 14}, result, 0.001f);

            a.close();
            b.close();
            c.close();
            outputs.getFirst().close();
        }
    }

    // ========================
    // Different data types
    // ========================

    private static final String INT32_ADD_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xi32>
            stablehlo.return %0 : tensor<4xi32>
          }
        }
        """;

    @Test
    void testGenerateInt32Model() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(INT32_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.Int32Model";
        var generated = ModelClassGenerator.generate(className, function, INT32_ADD_MLIR);

        assertNotNull(generated);
        assertTrue(generated.bytecode().length > 0);

        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);
        assertEquals(2, model.inputCount());
    }

    // ========================
    // Different tensor shapes
    // ========================

    private static final String SCALAR_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
            stablehlo.return %0 : tensor<f32>
          }
        }
        """;

    private static final String RANK3_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x3x4xf32>
            stablehlo.return %0 : tensor<2x3x4xf32>
          }
        }
        """;

    @Test
    void testGenerateScalarModel() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SCALAR_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.ScalarModel";
        var generated = ModelClassGenerator.generate(className, function, SCALAR_MLIR);

        assertNotNull(generated);

        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);
        assertEquals(2, model.inputCount());
        assertEquals(1, model.outputCount());
    }

    @Test
    void testGenerateRank3Model() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(RANK3_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated.Rank3Model";
        var generated = ModelClassGenerator.generate(className, function, RANK3_MLIR);

        CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);

        try (Backend backend = new CpuBackend()) {
            float[] data1 = new float[24];
            float[] data2 = new float[24];
            for (int i = 0; i < 24; i++) {
                data1[i] = i;
                data2[i] = 1;
            }

            Tensor input1 = Tensor.fromFloatArray(data1, 2, 3, 4);
            Tensor input2 = Tensor.fromFloatArray(data2, 2, 3, 4);

            List<Tensor> outputs = model.forward(List.of(input1, input2), backend);

            float[] result = outputs.getFirst().toFloatArray();
            for (int i = 0; i < 24; i++) {
                assertEquals(i + 1, result[i], 0.001f);
            }

            input1.close();
            input2.close();
            outputs.getFirst().close();
        }
    }

    // ========================
    // Different class names
    // ========================

    @Test
    void testGenerateWithDifferentPackages() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        String[] classNames = {
            "io.surfworks.warpforge.generated.Model",
            "com.example.MyModel",
            "SinglePackage.Model",
            "TopLevel"
        };

        for (String className : classNames) {
            var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

            assertNotNull(generated, "Should generate for class: " + className);
            assertEquals(className, generated.className());
            assertTrue(generated.bytecode().length > 0);

            // Verify it can be loaded
            CompiledModel model = ModelLoader.loadFromBytes(generated.bytecode(), className);
            assertEquals(2, model.inputCount());
        }
    }

    // ========================
    // Source hash consistency
    // ========================

    @Test
    void testSourceHashConsistency() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        // Generate twice with same source
        var generated1 = ModelClassGenerator.generate("io.test.Model1", function, SIMPLE_ADD_MLIR);
        var generated2 = ModelClassGenerator.generate("io.test.Model2", function, SIMPLE_ADD_MLIR);

        // Source hash should be the same (based on MLIR source, not class name)
        assertEquals(generated1.metadata().sourceHash(), generated2.metadata().sourceHash());
    }

    @Test
    void testSourceHashDiffersForDifferentSource() throws Exception {
        StableHloAst.Module module1 = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Module module2 = StableHloParser.parse(MULTI_OP_MLIR);

        var generated1 = ModelClassGenerator.generate("io.test.Add",
            module1.functions().getFirst(), SIMPLE_ADD_MLIR);
        var generated2 = ModelClassGenerator.generate("io.test.MulAdd",
            module2.functions().getFirst(), MULTI_OP_MLIR);

        // Source hashes should differ
        assertTrue(!generated1.metadata().sourceHash().equals(generated2.metadata().sourceHash()));
    }

    // ========================
    // Bytecode validity
    // ========================

    @Test
    void testBytecodeStartsWithClassMagic() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();

        var generated = ModelClassGenerator.generate("io.test.MagicTest", function, SIMPLE_ADD_MLIR);

        byte[] bytecode = generated.bytecode();

        // Java class files start with magic number: 0xCAFEBABE
        assertEquals((byte) 0xCA, bytecode[0]);
        assertEquals((byte) 0xFE, bytecode[1]);
        assertEquals((byte) 0xBA, bytecode[2]);
        assertEquals((byte) 0xBE, bytecode[3]);
    }
}
