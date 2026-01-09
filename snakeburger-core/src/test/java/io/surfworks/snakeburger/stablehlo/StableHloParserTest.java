package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.*;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StableHloParserTest {

    @Nested
    class ModuleParsingTests {

        @Test
        void parseSimpleModule() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    stablehlo.return %arg0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);

            assertEquals("test", module.name());
            assertEquals(1, module.functions().size());
        }

        @Test
        void parseModuleWithMultipleFunctions() {
            String mlir = """
                module @multi {
                  func.func public @func1(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                  func.func public @func2(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
                    stablehlo.return %arg0 : tensor<8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);

            assertEquals("multi", module.name());
            assertEquals(2, module.functions().size());
            assertEquals("func1", module.functions().get(0).name());
            assertEquals("func2", module.functions().get(1).name());
        }

        @Test
        void parseModuleWithUnderscoreInName() {
            String mlir = """
                module @my_module {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            assertEquals("my_module", module.name());
        }

        @Test
        void getNamedFunction() {
            String mlir = """
                module @test {
                  func.func public @first(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                  func.func public @second(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
                    stablehlo.return %arg0 : tensor<8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            assertTrue(module.getFunction("first").isPresent());
            assertTrue(module.getFunction("second").isPresent());
            assertFalse(module.getFunction("nonexistent").isPresent());
        }
    }

    @Nested
    class FunctionParsingTests {

        @Test
        void parsePublicFunction() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertTrue(func.isPublic());
            assertEquals("main", func.name());
        }

        @Test
        void parsePrivateFunction() {
            String mlir = """
                module @test {
                  func.func @helper(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertFalse(func.isPublic());
        }

        @Test
        void parseFunctionWithNoArguments() {
            String mlir = """
                module @test {
                  func.func public @noargs() -> (tensor<4xf32>) {
                    %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertTrue(func.arguments().isEmpty());
            assertEquals(1, func.resultTypes().size());
        }

        @Test
        void parseFunctionWithMultipleArguments() {
            String mlir = """
                module @test {
                  func.func public @multi(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %a : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals(3, func.arguments().size());
            assertEquals("a", func.arguments().get(0).name());
            assertEquals("b", func.arguments().get(1).name());
            assertEquals("c", func.arguments().get(2).name());
        }

        @Test
        void parseFunctionWithMultipleReturnTypes() {
            String mlir = """
                module @test {
                  func.func public @multi_return(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
                    %0 = stablehlo.constant dense<1.0> : tensor<8xf32>
                    stablehlo.return %arg0, %0 : tensor<4xf32>, tensor<8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals(2, func.resultTypes().size());
        }

        @Test
        void functionType() {
            String mlir = """
                module @test {
                  func.func public @typed(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<16xf32>) {
                    %0 = stablehlo.constant dense<1.0> : tensor<16xf32>
                    stablehlo.return %0 : tensor<16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            FunctionType funcType = func.type();

            assertEquals(2, funcType.inputTypes().size());
            assertEquals(1, funcType.resultTypes().size());
        }
    }

    @Nested
    class TensorTypeParsingTests {

        @Test
        void parse1DTensor() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Type argType = module.functions().get(0).arguments().get(0).type();

            assertInstanceOf(TensorType.class, argType);
            TensorType tensor = (TensorType) argType;
            assertEquals(List.of(4), tensor.shape());
            assertEquals("f32", tensor.elementType().name());
            assertEquals(1, tensor.rank());
        }

        @Test
        void parse2DTensor() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    stablehlo.return %arg0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TensorType tensor = (TensorType) module.functions().get(0).arguments().get(0).type();

            assertEquals(List.of(4, 8), tensor.shape());
            assertEquals(2, tensor.rank());
            assertEquals(4, tensor.dim(0));
            assertEquals(8, tensor.dim(1));
        }

        @Test
        void parse3DTensor() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<2x4x8xf32>) -> (tensor<2x4x8xf32>) {
                    stablehlo.return %arg0 : tensor<2x4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TensorType tensor = (TensorType) module.functions().get(0).arguments().get(0).type();

            assertEquals(List.of(2, 4, 8), tensor.shape());
            assertEquals(3, tensor.rank());
        }

        @Test
        void parse4DTensor() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>) {
                    stablehlo.return %arg0 : tensor<1x2x3x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TensorType tensor = (TensorType) module.functions().get(0).arguments().get(0).type();

            assertEquals(List.of(1, 2, 3, 4), tensor.shape());
            assertEquals(4, tensor.rank());
        }

        @Test
        void tensorElementCount() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>) {
                    stablehlo.return %arg0 : tensor<4x8x16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TensorType tensor = (TensorType) module.functions().get(0).arguments().get(0).type();

            assertEquals(4 * 8 * 16, tensor.elementCount());
        }

        @ParameterizedTest
        @ValueSource(strings = {"f16", "f32", "f64", "bf16", "i8", "i16", "i32", "i64"})
        void parseVariousElementTypes(String dtype) {
            String mlir = String.format("""
                module @test {
                  func.func public @main(%%arg0: tensor<4x%s>) -> (tensor<4x%s>) {
                    stablehlo.return %%arg0 : tensor<4x%s>
                  }
                }
                """, dtype, dtype, dtype);

            Module module = StableHloParser.parse(mlir);
            TensorType tensor = (TensorType) module.functions().get(0).arguments().get(0).type();

            assertEquals(dtype, tensor.elementType().name());
        }

        @Test
        void tensorTypeToMlirString() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    stablehlo.return %arg0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TensorType tensor = (TensorType) module.functions().get(0).arguments().get(0).type();

            assertEquals("tensor<4x8xf32>", tensor.toMlirString());
        }

        @Test
        void scalarTypeProperties() {
            ScalarType f32 = ScalarType.of("f32");
            ScalarType i32 = ScalarType.of("i32");
            ScalarType bf16 = ScalarType.of("bf16");

            assertTrue(f32.isFloatingPoint());
            assertFalse(f32.isInteger());

            assertTrue(i32.isInteger());
            assertFalse(i32.isFloatingPoint());

            assertTrue(bf16.isFloatingPoint());
        }
    }

    @Nested
    class OperationParsingTests {

        @Test
        void parseDotGeneral() {
            String mlir = """
                module @test {
                  func.func public @matmul(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DotGeneralOp dotOp = (DotGeneralOp) module.functions().get(0).body().get(0);

            assertEquals("arg0", dotOp.lhs().name());
            assertEquals("arg1", dotOp.rhs().name());
            assertEquals(List.of(1L), dotOp.dimensionNumbers().lhsContractingDimensions());
            assertEquals(List.of(0L), dotOp.dimensionNumbers().rhsContractingDimensions());
            assertTrue(dotOp.dimensionNumbers().lhsBatchingDimensions().isEmpty());
            assertTrue(dotOp.dimensionNumbers().rhsBatchingDimensions().isEmpty());
            assertEquals("0", dotOp.result().name());
        }

        @Test
        void parseDotGeneralWithBatching() {
            String mlir = """
                module @test {
                  func.func public @batched(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> (tensor<2x4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]> : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
                    stablehlo.return %0 : tensor<2x4x16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DotGeneralOp dotOp = (DotGeneralOp) module.functions().get(0).body().get(0);

            assertEquals(List.of(0L), dotOp.dimensionNumbers().lhsBatchingDimensions());
            assertEquals(List.of(0L), dotOp.dimensionNumbers().rhsBatchingDimensions());
            assertEquals(List.of(2L), dotOp.dimensionNumbers().lhsContractingDimensions());
            assertEquals(List.of(1L), dotOp.dimensionNumbers().rhsContractingDimensions());
        }

        @Test
        void parseConstantFloat() {
            String mlir = """
                module @test {
                  func.func public @zeros() -> (tensor<4x4xf32>) {
                    %0 = stablehlo.constant dense<0.0> : tensor<4x4xf32>
                    stablehlo.return %0 : tensor<4x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ConstantOp constOp = (ConstantOp) module.functions().get(0).body().get(0);

            assertEquals(0.0, constOp.value().value());
            assertEquals(List.of(4, 4), constOp.tensorResultType().shape());
        }

        @Test
        void parseConstantInteger() {
            String mlir = """
                module @test {
                  func.func public @ones() -> (tensor<4xi32>) {
                    %0 = stablehlo.constant dense<1> : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ConstantOp constOp = (ConstantOp) module.functions().get(0).body().get(0);

            assertEquals(1L, constOp.value().value());
        }

        @Test
        void parseConstantNegative() {
            String mlir = """
                module @test {
                  func.func public @neg() -> (tensor<4xf32>) {
                    %0 = stablehlo.constant dense<-1.5> : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ConstantOp constOp = (ConstantOp) module.functions().get(0).body().get(0);

            assertEquals(-1.5, constOp.value().value());
        }

        @Test
        void parseAdd() {
            String mlir = """
                module @test {
                  func.func public @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            AddOp addOp = (AddOp) module.functions().get(0).body().get(0);

            assertEquals("a", addOp.lhs().name());
            assertEquals("b", addOp.rhs().name());
            assertEquals("stablehlo.add", addOp.opName());
        }

        @Test
        void parseMultiply() {
            String mlir = """
                module @test {
                  func.func public @mul(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.multiply %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            MultiplyOp mulOp = (MultiplyOp) module.functions().get(0).body().get(0);

            assertEquals("a", mulOp.lhs().name());
            assertEquals("b", mulOp.rhs().name());
        }

        @Test
        void parseDivide() {
            String mlir = """
                module @test {
                  func.func public @div(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.divide %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DivideOp divOp = (DivideOp) module.functions().get(0).body().get(0);

            assertEquals("a", divOp.lhs().name());
            assertEquals("b", divOp.rhs().name());
        }

        @Test
        void parseMaximum() {
            String mlir = """
                module @test {
                  func.func public @max(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.maximum %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            MaximumOp maxOp = (MaximumOp) module.functions().get(0).body().get(0);

            assertEquals("a", maxOp.lhs().name());
            assertEquals("b", maxOp.rhs().name());
        }

        @Test
        void parseNegate() {
            String mlir = """
                module @test {
                  func.func public @neg(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.negate %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            NegateOp negOp = (NegateOp) module.functions().get(0).body().get(0);

            assertEquals("a", negOp.operand().name());
            assertEquals(List.of(negOp.operand()), negOp.operands());
        }

        @Test
        void parseReshape() {
            String mlir = """
                module @test {
                  func.func public @reshape(%a: tensor<4x8xf32>) -> (tensor<32xf32>) {
                    %0 = stablehlo.reshape %a : tensor<4x8xf32> -> tensor<32xf32>
                    stablehlo.return %0 : tensor<32xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReshapeOp reshapeOp = (ReshapeOp) module.functions().get(0).body().get(0);

            assertEquals("a", reshapeOp.operand().name());
            assertEquals(List.of(32), reshapeOp.tensorResultType().shape());
        }

        @Test
        void parseTranspose() {
            String mlir = """
                module @test {
                  func.func public @transpose(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a, dims = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TransposeOp transposeOp = (TransposeOp) module.functions().get(0).body().get(0);

            assertEquals("a", transposeOp.operand().name());
            assertEquals(List.of(1L, 0L), transposeOp.permutation());
            assertEquals(List.of(8, 4), transposeOp.tensorResultType().shape());
        }

        @Test
        void parseBroadcastInDim() {
            String mlir = """
                module @test {
                  func.func public @broadcast(%a: tensor<4xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            BroadcastInDimOp broadcastOp = (BroadcastInDimOp) module.functions().get(0).body().get(0);

            assertEquals("a", broadcastOp.operand().name());
            assertEquals(List.of(0L), broadcastOp.broadcastDimensions());
            assertEquals(List.of(4, 8), broadcastOp.tensorResultType().shape());
        }

        @Test
        void parseReturnSingleValue() {
            String mlir = """
                module @test {
                  func.func public @single(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %a : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReturnOp returnOp = (ReturnOp) module.functions().get(0).body().get(0);

            assertEquals(1, returnOp.operands().size());
            assertEquals("a", returnOp.operands().get(0).name());
        }

        @Test
        void parseReturnMultipleValues() {
            String mlir = """
                module @test {
                  func.func public @multi(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
                    stablehlo.return %a, %b : tensor<4xf32>, tensor<8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReturnOp returnOp = (ReturnOp) module.functions().get(0).body().get(0);

            assertEquals(2, returnOp.operands().size());
            assertEquals("a", returnOp.operands().get(0).name());
            assertEquals("b", returnOp.operands().get(1).name());
        }
    }

    @Nested
    class ChainedOperationsTests {

        @Test
        void parseOperationChain() {
            String mlir = """
                module @test {
                  func.func public @chain(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    %1 = stablehlo.multiply %0, %b : tensor<4xf32>
                    %2 = stablehlo.negate %1 : tensor<4xf32>
                    stablehlo.return %2 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals(4, func.body().size());
            assertInstanceOf(AddOp.class, func.body().get(0));
            assertInstanceOf(MultiplyOp.class, func.body().get(1));
            assertInstanceOf(NegateOp.class, func.body().get(2));
            assertInstanceOf(ReturnOp.class, func.body().get(3));

            // Verify chaining
            MultiplyOp mulOp = (MultiplyOp) func.body().get(1);
            assertEquals("0", mulOp.lhs().name());

            NegateOp negOp = (NegateOp) func.body().get(2);
            assertEquals("1", negOp.operand().name());
        }

        @Test
        void parseMlpPattern() {
            String mlir = """
                module @main {
                  func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    %1 = stablehlo.constant dense<0.0> : tensor<4x16xf32>
                    %2 = stablehlo.maximum %0, %1 : tensor<4x16xf32>
                    stablehlo.return %2 : tensor<4x16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals(4, func.body().size());
            assertInstanceOf(DotGeneralOp.class, func.body().get(0));
            assertInstanceOf(ConstantOp.class, func.body().get(1));
            assertInstanceOf(MaximumOp.class, func.body().get(2));
            assertInstanceOf(ReturnOp.class, func.body().get(3));

            // Verify ReLU pattern: max(matmul_result, 0)
            MaximumOp maxOp = (MaximumOp) func.body().get(2);
            assertEquals("0", maxOp.lhs().name());  // matmul result
            assertEquals("1", maxOp.rhs().name());  // constant zero
        }
    }

    @Nested
    class ValueReferenceTests {

        @Test
        void argumentsToValue() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Argument arg = module.functions().get(0).arguments().get(0);
            Value val = arg.toValue();

            assertEquals("arg0", val.name());
            assertEquals(arg.type(), val.type());
            assertEquals("%arg0", val.toMlirString());
        }

        @Test
        void valueToString() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Value val = module.functions().get(0).arguments().get(0).toValue();

            assertTrue(val.toString().contains("%arg0"));
            assertTrue(val.toString().contains("tensor<4xf32>"));
        }
    }

    @Nested
    class ErrorHandlingTests {

        @Test
        void undefinedValueThrows() {
            String mlir = """
                module @test {
                  func.func public @bad(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %undefined : tensor<4xf32>
                  }
                }
                """;

            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void unsupportedOperationThrows() {
            String mlir = """
                module @test {
                  func.func public @bad(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.unsupported_op %arg0 : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void parseExceptionHasLocation() {
            try {
                String mlir = "module @test { invalid }";
                StableHloParser.parse(mlir);
                fail("Expected exception");
            } catch (StableHloParseException e) {
                assertTrue(e.getMessage().contains("line") || e.getLine() > 0);
            }
        }
    }

    @Nested
    class AttributeTests {

        @Test
        void dotDimensionNumbersToMlirString() {
            DotDimensionNumbers dims = new DotDimensionNumbers(
                    List.of(0L),
                    List.of(0L),
                    List.of(2L),
                    List.of(1L)
            );

            String mlir = dims.toMlirString();
            assertTrue(mlir.contains("lhs_batching_dimensions = [0]"));
            assertTrue(mlir.contains("rhs_batching_dimensions = [0]"));
            assertTrue(mlir.contains("lhs_contracting_dimensions = [2]"));
            assertTrue(mlir.contains("rhs_contracting_dimensions = [1]"));
        }

        @Test
        void denseAttrToMlirString() {
            TensorType type = new TensorType(List.of(4, 4), ScalarType.F32);
            DenseAttr attr = new DenseAttr(0.0, type);

            String mlir = attr.toMlirString();
            assertTrue(mlir.contains("dense<0.0>"));
            assertTrue(mlir.contains("tensor<4x4xf32>"));
        }

        @Test
        void integerAttr() {
            IntegerAttr attr = new IntegerAttr(42);
            assertEquals("42", attr.toMlirString());
        }

        @Test
        void floatAttr() {
            FloatAttr attr = new FloatAttr(3.14);
            assertTrue(attr.toMlirString().contains("3.14"));
        }

        @Test
        void stringAttr() {
            StringAttr attr = new StringAttr("hello");
            assertEquals("\"hello\"", attr.toMlirString());
        }

        @Test
        void arrayAttr() {
            ArrayAttr attr = new ArrayAttr(List.of(
                    new IntegerAttr(1),
                    new IntegerAttr(2),
                    new IntegerAttr(3)
            ));
            assertEquals("[1, 2, 3]", attr.toMlirString());
            assertEquals(List.of(1L, 2L, 3L), attr.asIntegerList());
        }
    }

    @Nested
    class OperationInterfaceTests {

        @Test
        void operationOpName() {
            String mlir = """
                module @test {
                  func.func public @ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            AddOp addOp = (AddOp) module.functions().get(0).body().get(0);
            ReturnOp returnOp = (ReturnOp) module.functions().get(0).body().get(1);

            assertEquals("stablehlo.add", addOp.opName());
            assertEquals("stablehlo.return", returnOp.opName());
        }

        @Test
        void operationResults() {
            String mlir = """
                module @test {
                  func.func public @ops(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.negate %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            NegateOp negOp = (NegateOp) module.functions().get(0).body().get(0);
            ReturnOp returnOp = (ReturnOp) module.functions().get(0).body().get(1);

            assertEquals(1, negOp.results().size());
            assertEquals("0", negOp.results().get(0).name());

            assertTrue(returnOp.results().isEmpty());
        }

        @Test
        void returnOpHasNullResultType() {
            String mlir = """
                module @test {
                  func.func public @ops(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %a : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReturnOp returnOp = (ReturnOp) module.functions().get(0).body().get(0);

            assertNull(returnOp.tensorResultType());
        }
    }
}
