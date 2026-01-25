package io.surfworks.warpforge.data.stablehlo;

import io.surfworks.warpforge.data.stablehlo.StableHloModule.StableHloFunction;
import io.surfworks.warpforge.data.stablehlo.StableHloOps.AddOp;
import io.surfworks.warpforge.data.stablehlo.StableHloOps.DotOp;
import io.surfworks.warpforge.data.stablehlo.StableHloOps.MultiplyOp;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.ScalarType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.TensorType;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StableHloModuleTest {

    @TempDir
    Path tempDir;

    @Nested
    class BuilderTests {

        @Test
        void testEmptyModule() {
            StableHloModule module = StableHloModule.builder("test")
                    .build();

            assertEquals("test", module.name());
            assertTrue(module.functions().isEmpty());
        }

        @Test
        void testModuleWithFunction() {
            TensorType inputType = TensorType.of(ScalarType.F32, 1, 8);
            TensorType outputType = TensorType.of(ScalarType.F32, 1, 8);

            StableHloFunction func = StableHloFunction.builder("forward")
                    .addArgument("arg0", inputType)
                    .addArgument("arg1", inputType)
                    .addReturn(outputType)
                    .build();

            StableHloModule module = StableHloModule.builder("add_model")
                    .addFunction(func)
                    .build();

            assertEquals("add_model", module.name());
            assertEquals(1, module.functions().size());
            assertNotNull(module.function("forward"));
        }

        @Test
        void testFunctionWithOperations() {
            TensorType type = TensorType.of(ScalarType.F32, 4, 8);

            StableHloFunction.Builder funcBuilder = StableHloFunction.builder("forward")
                    .addArgument("arg0", type)
                    .addArgument("arg1", type);

            List<String> addResult = funcBuilder.addOperation(
                    new AddOp(type),
                    List.of("arg0", "arg1")
            );

            funcBuilder.addReturn(type);
            funcBuilder.addReturnOp(addResult);

            StableHloFunction func = funcBuilder.build();

            assertEquals("forward", func.name());
            assertEquals(2, func.arguments().size());
            assertEquals(2, func.operations().size()); // add + return
        }
    }

    @Nested
    class MlirEmissionTests {

        @Test
        void testSimpleModuleToMlir() {
            TensorType type = TensorType.of(ScalarType.F32, 1, 8);

            StableHloFunction func = StableHloFunction.builder("forward")
                    .addArgument("arg0", type)
                    .addArgument("arg1", type)
                    .addReturn(type)
                    .build();

            StableHloModule module = StableHloModule.builder("main")
                    .addFunction(func)
                    .build();

            String mlir = module.toMlir();

            assertTrue(mlir.contains("module @main"));
            assertTrue(mlir.contains("func.func public @forward"));
            assertTrue(mlir.contains("%arg0: tensor<1x8xf32>"));
            assertTrue(mlir.contains("%arg1: tensor<1x8xf32>"));
            assertTrue(mlir.contains("-> (tensor<1x8xf32>)"));
        }

        @Test
        void testOperationsToMlir() {
            TensorType type = TensorType.of(ScalarType.F32, 4, 8);

            StableHloFunction.Builder funcBuilder = StableHloFunction.builder("forward")
                    .addArgument("arg0", type)
                    .addArgument("arg1", type);

            List<String> addResult = funcBuilder.addOperation(
                    new AddOp(type),
                    List.of("arg0", "arg1")
            );

            List<String> mulResult = funcBuilder.addOperation(
                    new MultiplyOp(type),
                    List.of(addResult.get(0), "arg1")
            );

            funcBuilder.addReturn(type);
            funcBuilder.addReturnOp(mulResult);

            StableHloFunction func = funcBuilder.build();
            String mlir = func.toMlir(2);

            assertTrue(mlir.contains("stablehlo.add"));
            assertTrue(mlir.contains("stablehlo.multiply"));
            assertTrue(mlir.contains("stablehlo.return"));
        }

        @Test
        void testDotOpToMlir() {
            TensorType inputA = TensorType.of(ScalarType.F32, 1, 768);
            TensorType inputB = TensorType.of(ScalarType.F32, 768, 768);
            TensorType output = TensorType.of(ScalarType.F32, 1, 768);

            StableHloFunction.Builder funcBuilder = StableHloFunction.builder("linear")
                    .addArgument("input", inputA)
                    .addArgument("weight", inputB);

            List<String> dotResult = funcBuilder.addOperation(
                    new DotOp(output),
                    List.of("input", "weight")
            );

            funcBuilder.addReturn(output);
            funcBuilder.addReturnOp(dotResult);

            StableHloFunction func = funcBuilder.build();
            String mlir = func.toMlir(2);

            assertTrue(mlir.contains("stablehlo.dot"));
            assertTrue(mlir.contains("%input: tensor<1x768xf32>"));
            assertTrue(mlir.contains("%weight: tensor<768x768xf32>"));
        }
    }

    @Nested
    class FileIOTests {

        @Test
        void testWriteAndLoad() throws IOException {
            TensorType type = TensorType.of(ScalarType.F32, 1, 8);

            StableHloFunction func = StableHloFunction.builder("forward")
                    .addArgument("arg0", type)
                    .addArgument("arg1", type)
                    .addReturn(type)
                    .build();

            StableHloModule original = StableHloModule.builder("test_model")
                    .addFunction(func)
                    .build();

            Path mlirFile = tempDir.resolve("test.mlir");
            original.writeTo(mlirFile);

            assertTrue(Files.exists(mlirFile));
            String content = Files.readString(mlirFile);
            assertTrue(content.contains("module @test_model"));

            StableHloModule loaded = StableHloModule.loadFrom(mlirFile);
            assertEquals("test_model", loaded.name());
            assertNotNull(loaded.function("forward"));
        }

        @Test
        void testParseBasicMlir() throws IOException {
            String mlir = """
                    module @simple {
                      func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                        %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
                        stablehlo.return %0 : tensor<4x8xf32>
                      }
                    }
                    """;

            StableHloModule module = StableHloModule.parse(mlir);

            assertEquals("simple", module.name());
            assertEquals(1, module.functions().size());

            StableHloFunction func = module.function("forward");
            assertNotNull(func);
            assertTrue(func.isPublic());
            assertEquals(2, func.arguments().size());
            assertEquals(1, func.returnTypes().size());

            assertEquals("arg0", func.arguments().get(0).name());
            assertEquals(ScalarType.F32, func.arguments().get(0).type().elementType());
        }
    }

    @Nested
    class FunctionTests {

        @Test
        void testMainFunction() {
            TensorType type = TensorType.of(ScalarType.F32, 4);

            StableHloModule module = StableHloModule.builder("test")
                    .addFunction(StableHloFunction.builder("forward")
                            .addArgument("x", type)
                            .addReturn(type)
                            .build())
                    .addFunction(StableHloFunction.builder("helper")
                            .addArgument("y", type)
                            .addReturn(type)
                            .build())
                    .build();

            // "forward" should be returned as main
            StableHloFunction main = module.mainFunction();
            assertEquals("forward", main.name());
        }

        @Test
        void testMainFunctionFallback() {
            TensorType type = TensorType.of(ScalarType.F32, 4);

            StableHloModule module = StableHloModule.builder("test")
                    .addFunction(StableHloFunction.builder("main")
                            .addArgument("x", type)
                            .addReturn(type)
                            .build())
                    .build();

            StableHloFunction main = module.mainFunction();
            assertEquals("main", main.name());
        }

        @Test
        void testFunctionType() {
            TensorType inputA = TensorType.of(ScalarType.F32, 1, 8);
            TensorType inputB = TensorType.of(ScalarType.F32, 8, 8);
            TensorType output = TensorType.of(ScalarType.F32, 1, 8);

            StableHloFunction func = StableHloFunction.builder("matmul")
                    .addArgument("a", inputA)
                    .addArgument("b", inputB)
                    .addReturn(output)
                    .build();

            assertEquals("(tensor<1x8xf32>, tensor<8x8xf32>) -> (tensor<1x8xf32>)", func.type().toMlir());
        }

        @Test
        void testPrivateFunction() {
            TensorType type = TensorType.of(ScalarType.F32, 4);

            StableHloFunction func = StableHloFunction.builder("helper")
                    .setPublic(false)
                    .addArgument("x", type)
                    .addReturn(type)
                    .build();

            assertFalse(func.isPublic());
            assertFalse(func.toMlir(0).contains("public"));
        }
    }

    @Nested
    class LookupTests {

        @Test
        void testFunctionByName() {
            TensorType type = TensorType.of(ScalarType.F32, 4);

            StableHloModule module = StableHloModule.builder("test")
                    .addFunction(StableHloFunction.builder("forward")
                            .addArgument("x", type)
                            .addReturn(type)
                            .build())
                    .addFunction(StableHloFunction.builder("backward")
                            .addArgument("grad", type)
                            .addReturn(type)
                            .build())
                    .build();

            assertNotNull(module.function("forward"));
            assertNotNull(module.function("backward"));
            assertNull(module.function("nonexistent"));
        }
    }
}
