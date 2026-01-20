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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
}
