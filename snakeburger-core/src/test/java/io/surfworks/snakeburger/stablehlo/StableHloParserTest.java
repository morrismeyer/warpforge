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

        // ==================== New 80/20 Operations ====================

        @Test
        void parseSubtract() {
            String mlir = """
                module @test {
                  func.func public @sub(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.subtract %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SubtractOp subOp = (SubtractOp) module.functions().get(0).body().get(0);

            assertEquals("a", subOp.lhs().name());
            assertEquals("b", subOp.rhs().name());
            assertEquals("stablehlo.subtract", subOp.opName());
        }

        @Test
        void parseMinimum() {
            String mlir = """
                module @test {
                  func.func public @min(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.minimum %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            MinimumOp minOp = (MinimumOp) module.functions().get(0).body().get(0);

            assertEquals("a", minOp.lhs().name());
            assertEquals("b", minOp.rhs().name());
        }

        @Test
        void parseAbs() {
            String mlir = """
                module @test {
                  func.func public @abs(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.abs %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            AbsOp absOp = (AbsOp) module.functions().get(0).body().get(0);

            assertEquals("a", absOp.operand().name());
            assertEquals("stablehlo.abs", absOp.opName());
        }

        @Test
        void parseExp() {
            String mlir = """
                module @test {
                  func.func public @exp(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.exponential %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ExpOp expOp = (ExpOp) module.functions().get(0).body().get(0);

            assertEquals("a", expOp.operand().name());
            assertEquals("stablehlo.exponential", expOp.opName());
        }

        @Test
        void parseLog() {
            String mlir = """
                module @test {
                  func.func public @log(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.log %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            LogOp logOp = (LogOp) module.functions().get(0).body().get(0);

            assertEquals("a", logOp.operand().name());
        }

        @Test
        void parseTanh() {
            String mlir = """
                module @test {
                  func.func public @tanh(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.tanh %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TanhOp tanhOp = (TanhOp) module.functions().get(0).body().get(0);

            assertEquals("a", tanhOp.operand().name());
        }

        @Test
        void parseSqrt() {
            String mlir = """
                module @test {
                  func.func public @sqrt(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.sqrt %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SqrtOp sqrtOp = (SqrtOp) module.functions().get(0).body().get(0);

            assertEquals("a", sqrtOp.operand().name());
        }

        @Test
        void parseRsqrt() {
            String mlir = """
                module @test {
                  func.func public @rsqrt(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.rsqrt %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            RsqrtOp rsqrtOp = (RsqrtOp) module.functions().get(0).body().get(0);

            assertEquals("a", rsqrtOp.operand().name());
        }

        @Test
        void parseCompare() {
            String mlir = """
                module @test {
                  func.func public @cmp(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xi1>) {
                    %0 = stablehlo.compare %a, %b, direction = GT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    stablehlo.return %0 : tensor<4xi1>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            CompareOp compareOp = (CompareOp) module.functions().get(0).body().get(0);

            assertEquals("a", compareOp.lhs().name());
            assertEquals("b", compareOp.rhs().name());
            assertEquals(ComparisonDirection.GT, compareOp.direction());
            assertEquals(ScalarType.I1, compareOp.tensorResultType().elementType());
        }

        @Test
        void parseSelect() {
            String mlir = """
                module @test {
                  func.func public @sel(%pred: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.select %pred, %a, %b : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SelectOp selectOp = (SelectOp) module.functions().get(0).body().get(0);

            assertEquals("pred", selectOp.pred().name());
            assertEquals("a", selectOp.onTrue().name());
            assertEquals("b", selectOp.onFalse().name());
        }

        @Test
        void parseConcatenate() {
            String mlir = """
                module @test {
                  func.func public @concat(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> (tensor<8x8xf32>) {
                    %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
                    stablehlo.return %0 : tensor<8x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ConcatenateOp concatOp = (ConcatenateOp) module.functions().get(0).body().get(0);

            assertEquals(2, concatOp.inputs().size());
            assertEquals("a", concatOp.inputs().get(0).name());
            assertEquals("b", concatOp.inputs().get(1).name());
            assertEquals(0, concatOp.dimension());
            assertEquals(List.of(8, 8), concatOp.tensorResultType().shape());
        }

        @Test
        void parseSlice() {
            String mlir = """
                module @test {
                  func.func public @slice(%a: tensor<8x8xf32>) -> (tensor<4x4xf32>) {
                    %0 = stablehlo.slice %a, starts = [0, 0], limits = [4, 4], strides = [1, 1] : tensor<8x8xf32> -> tensor<4x4xf32>
                    stablehlo.return %0 : tensor<4x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SliceOp sliceOp = (SliceOp) module.functions().get(0).body().get(0);

            assertEquals("a", sliceOp.operand().name());
            assertEquals(List.of(0L, 0L), sliceOp.startIndices());
            assertEquals(List.of(4L, 4L), sliceOp.limitIndices());
            assertEquals(List.of(1L, 1L), sliceOp.strides());
            assertEquals(List.of(4, 4), sliceOp.tensorResultType().shape());
        }

        @Test
        void parseClamp() {
            String mlir = """
                module @test {
                  func.func public @clamp(%min: tensor<4xf32>, %x: tensor<4xf32>, %max: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.clamp %min, %x, %max : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ClampOp clampOp = (ClampOp) module.functions().get(0).body().get(0);

            assertEquals("min", clampOp.min().name());
            assertEquals("x", clampOp.operand().name());
            assertEquals("max", clampOp.max().name());
        }

        @Test
        void parseConvert() {
            String mlir = """
                module @test {
                  func.func public @convert(%a: tensor<4xf32>) -> (tensor<4xf16>) {
                    %0 = stablehlo.convert %a : tensor<4xf32> -> tensor<4xf16>
                    stablehlo.return %0 : tensor<4xf16>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ConvertOp convertOp = (ConvertOp) module.functions().get(0).body().get(0);

            assertEquals("a", convertOp.operand().name());
            assertEquals(ScalarType.F16, convertOp.tensorResultType().elementType());
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
    class AdditionalBinaryOpsTests {

        @Test
        void parsePower() {
            String mlir = """
                module @test {
                  func.func public @pow(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.power %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            PowerOp powerOp = (PowerOp) module.functions().get(0).body().get(0);

            assertEquals("a", powerOp.lhs().name());
            assertEquals("b", powerOp.rhs().name());
            assertEquals("stablehlo.power", powerOp.opName());
        }

        @Test
        void parseRemainder() {
            String mlir = """
                module @test {
                  func.func public @rem(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.remainder %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            RemainderOp remOp = (RemainderOp) module.functions().get(0).body().get(0);

            assertEquals("a", remOp.lhs().name());
            assertEquals("b", remOp.rhs().name());
        }

        @Test
        void parseAtan2() {
            String mlir = """
                module @test {
                  func.func public @atan2(%y: tensor<4xf32>, %x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.atan2 %y, %x : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Atan2Op atan2Op = (Atan2Op) module.functions().get(0).body().get(0);

            assertEquals("y", atan2Op.lhs().name());
            assertEquals("x", atan2Op.rhs().name());
        }

        @Test
        void parseAnd() {
            String mlir = """
                module @test {
                  func.func public @and_op(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.and %a, %b : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            AndOp andOp = (AndOp) module.functions().get(0).body().get(0);

            assertEquals("a", andOp.lhs().name());
            assertEquals("b", andOp.rhs().name());
            assertEquals("stablehlo.and", andOp.opName());
        }

        @Test
        void parseOr() {
            String mlir = """
                module @test {
                  func.func public @or_op(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.or %a, %b : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            OrOp orOp = (OrOp) module.functions().get(0).body().get(0);

            assertEquals("a", orOp.lhs().name());
            assertEquals("b", orOp.rhs().name());
        }

        @Test
        void parseXor() {
            String mlir = """
                module @test {
                  func.func public @xor_op(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.xor %a, %b : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            XorOp xorOp = (XorOp) module.functions().get(0).body().get(0);

            assertEquals("a", xorOp.lhs().name());
            assertEquals("b", xorOp.rhs().name());
        }

        @Test
        void parseShiftLeft() {
            String mlir = """
                module @test {
                  func.func public @shl(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.shift_left %a, %b : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ShiftLeftOp shlOp = (ShiftLeftOp) module.functions().get(0).body().get(0);

            assertEquals("a", shlOp.lhs().name());
            assertEquals("b", shlOp.rhs().name());
            assertEquals("stablehlo.shift_left", shlOp.opName());
        }

        @Test
        void parseShiftRightArithmetic() {
            String mlir = """
                module @test {
                  func.func public @shra(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.shift_right_arithmetic %a, %b : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ShiftRightArithmeticOp shraOp = (ShiftRightArithmeticOp) module.functions().get(0).body().get(0);

            assertEquals("a", shraOp.lhs().name());
            assertEquals("b", shraOp.rhs().name());
        }

        @Test
        void parseShiftRightLogical() {
            String mlir = """
                module @test {
                  func.func public @shrl(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.shift_right_logical %a, %b : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ShiftRightLogicalOp shrlOp = (ShiftRightLogicalOp) module.functions().get(0).body().get(0);

            assertEquals("a", shrlOp.lhs().name());
            assertEquals("b", shrlOp.rhs().name());
        }
    }

    @Nested
    class AdditionalUnaryOpsTests {

        @Test
        void parseSin() {
            String mlir = """
                module @test {
                  func.func public @sin(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.sine %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SinOp sinOp = (SinOp) module.functions().get(0).body().get(0);

            assertEquals("a", sinOp.operand().name());
            assertEquals("stablehlo.sine", sinOp.opName());
        }

        @Test
        void parseCos() {
            String mlir = """
                module @test {
                  func.func public @cos(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.cosine %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            CosOp cosOp = (CosOp) module.functions().get(0).body().get(0);

            assertEquals("a", cosOp.operand().name());
            assertEquals("stablehlo.cosine", cosOp.opName());
        }

        @Test
        void parseTan() {
            String mlir = """
                module @test {
                  func.func public @tan(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.tan %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TanOp tanOp = (TanOp) module.functions().get(0).body().get(0);

            assertEquals("a", tanOp.operand().name());
        }

        @Test
        void parseCeil() {
            String mlir = """
                module @test {
                  func.func public @ceil(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.ceil %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            CeilOp ceilOp = (CeilOp) module.functions().get(0).body().get(0);

            assertEquals("a", ceilOp.operand().name());
            assertEquals("stablehlo.ceil", ceilOp.opName());
        }

        @Test
        void parseFloor() {
            String mlir = """
                module @test {
                  func.func public @floor(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.floor %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            FloorOp floorOp = (FloorOp) module.functions().get(0).body().get(0);

            assertEquals("a", floorOp.operand().name());
        }

        @Test
        void parseSign() {
            String mlir = """
                module @test {
                  func.func public @sign(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.sign %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SignOp signOp = (SignOp) module.functions().get(0).body().get(0);

            assertEquals("a", signOp.operand().name());
        }

        @Test
        void parseLogistic() {
            String mlir = """
                module @test {
                  func.func public @logistic(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.logistic %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            LogisticOp logisticOp = (LogisticOp) module.functions().get(0).body().get(0);

            assertEquals("a", logisticOp.operand().name());
            assertEquals("stablehlo.logistic", logisticOp.opName());
        }

        @Test
        void parseExpm1() {
            String mlir = """
                module @test {
                  func.func public @expm1(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.exponential_minus_one %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Expm1Op expm1Op = (Expm1Op) module.functions().get(0).body().get(0);

            assertEquals("a", expm1Op.operand().name());
            assertEquals("stablehlo.exponential_minus_one", expm1Op.opName());
        }

        @Test
        void parseLog1p() {
            String mlir = """
                module @test {
                  func.func public @log1p(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.log_plus_one %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Log1pOp log1pOp = (Log1pOp) module.functions().get(0).body().get(0);

            assertEquals("a", log1pOp.operand().name());
        }

        @Test
        void parseCbrt() {
            String mlir = """
                module @test {
                  func.func public @cbrt(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.cbrt %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            CbrtOp cbrtOp = (CbrtOp) module.functions().get(0).body().get(0);

            assertEquals("a", cbrtOp.operand().name());
        }

        @Test
        void parseIsFinite() {
            String mlir = """
                module @test {
                  func.func public @is_finite(%a: tensor<4xf32>) -> (tensor<4xi1>) {
                    %0 = stablehlo.is_finite %a : tensor<4xf32> -> tensor<4xi1>
                    stablehlo.return %0 : tensor<4xi1>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            IsFiniteOp isFiniteOp = (IsFiniteOp) module.functions().get(0).body().get(0);

            assertEquals("a", isFiniteOp.operand().name());
            assertEquals(ScalarType.I1, isFiniteOp.tensorResultType().elementType());
        }

        @Test
        void parsePopcnt() {
            String mlir = """
                module @test {
                  func.func public @popcnt(%a: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.popcnt %a : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            PopcntOp popcntOp = (PopcntOp) module.functions().get(0).body().get(0);

            assertEquals("a", popcntOp.operand().name());
            assertEquals("stablehlo.popcnt", popcntOp.opName());
        }

        @Test
        void parseClz() {
            String mlir = """
                module @test {
                  func.func public @clz(%a: tensor<4xi32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.count_leading_zeros %a : tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ClzOp clzOp = (ClzOp) module.functions().get(0).body().get(0);

            assertEquals("a", clzOp.operand().name());
        }

        @Test
        void parseRoundNearestEven() {
            String mlir = """
                module @test {
                  func.func public @round(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.round_nearest_even %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            RoundNearestEvenOp roundOp = (RoundNearestEvenOp) module.functions().get(0).body().get(0);

            assertEquals("a", roundOp.operand().name());
        }

        @Test
        void parseRoundNearestAfz() {
            String mlir = """
                module @test {
                  func.func public @round_afz(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.round_nearest_afz %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            RoundNearestAfzOp roundOp = (RoundNearestAfzOp) module.functions().get(0).body().get(0);

            assertEquals("a", roundOp.operand().name());
        }

        @Test
        void parseNot() {
            String mlir = """
                module @test {
                  func.func public @not_op(%a: tensor<4xi1>) -> (tensor<4xi1>) {
                    %0 = stablehlo.not %a : tensor<4xi1>
                    stablehlo.return %0 : tensor<4xi1>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            NotOp notOp = (NotOp) module.functions().get(0).body().get(0);

            assertEquals("a", notOp.operand().name());
            assertEquals("stablehlo.not", notOp.opName());
        }
    }

    @Nested
    class AdditionalShapeOpsTests {

        @Test
        void parseReverse() {
            String mlir = """
                module @test {
                  func.func public @reverse(%a: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.reverse %a, dims = [0, 1] : tensor<4x8xf32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReverseOp reverseOp = (ReverseOp) module.functions().get(0).body().get(0);

            assertEquals("a", reverseOp.operand().name());
            assertEquals(List.of(0L, 1L), reverseOp.dimensions());
            assertEquals("stablehlo.reverse", reverseOp.opName());
        }

        @Test
        void parsePad() {
            String mlir = """
                module @test {
                  func.func public @pad(%a: tensor<4x4xf32>, %padding: tensor<f32>) -> (tensor<6x6xf32>) {
                    %0 = stablehlo.pad %a, %padding, low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<4x4xf32>, tensor<f32>) -> tensor<6x6xf32>
                    stablehlo.return %0 : tensor<6x6xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            PadOp padOp = (PadOp) module.functions().get(0).body().get(0);

            assertEquals("a", padOp.operand().name());
            assertEquals("padding", padOp.paddingValue().name());
            assertEquals(List.of(1L, 1L), padOp.edgePaddingLow());
            assertEquals(List.of(1L, 1L), padOp.edgePaddingHigh());
            assertEquals(List.of(0L, 0L), padOp.interiorPadding());
            assertEquals(List.of(6, 6), padOp.tensorResultType().shape());
        }

        @Test
        void parseIota() {
            String mlir = """
                module @test {
                  func.func public @iota() -> (tensor<4xf32>) {
                    %0 = stablehlo.iota dim = 0 : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            IotaOp iotaOp = (IotaOp) module.functions().get(0).body().get(0);

            assertEquals(0, iotaOp.iotaDimension());
            assertEquals(List.of(4), iotaOp.tensorResultType().shape());
            assertTrue(iotaOp.operands().isEmpty());
            assertEquals("stablehlo.iota", iotaOp.opName());
        }

        @Test
        void parseBitcastConvert() {
            String mlir = """
                module @test {
                  func.func public @bitcast(%a: tensor<4xf32>) -> (tensor<4xi32>) {
                    %0 = stablehlo.bitcast_convert %a : tensor<4xf32> -> tensor<4xi32>
                    stablehlo.return %0 : tensor<4xi32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            BitcastConvertOp bitcastOp = (BitcastConvertOp) module.functions().get(0).body().get(0);

            assertEquals("a", bitcastOp.operand().name());
            assertEquals(ScalarType.I32, bitcastOp.tensorResultType().elementType());
        }
    }

    @Nested
    class MalformedInputTests {

        @Test
        void missingModuleKeyword() {
            String mlir = "@test { }";
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingModuleName() {
            String mlir = "module { }";
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingFunctionBody() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>)
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void malformedTensorType() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingOperandInBinaryOp() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingResultType() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.negate %a
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void malformedDotDimensions() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_contracting_dimensions = [> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingReturnStatement() {
            // Parser allows missing return - type checker would catch this
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.negate %arg0 : tensor<4xf32>
                  }
                }
                """;

            // Parser succeeds - function body just has no return op
            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertEquals(1, func.body().size());
            assertInstanceOf(NegateOp.class, func.body().get(0));
        }

        @Test
        void invalidCompareDirection() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xi1>) {
                    %0 = stablehlo.compare %a, %b, direction = INVALID : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
                    stablehlo.return %0 : tensor<4xi1>
                  }
                }
                """;
            assertThrows(IllegalArgumentException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingSliceAttributes() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<8xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.slice %a, starts = [0] : tensor<8xf32> -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingTransposePermutation() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingBroadcastDims() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.broadcast_in_dim %a : (tensor<4xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingPadAttributes() {
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4xf32>, %p: tensor<f32>) -> (tensor<6xf32>) {
                    %0 = stablehlo.pad %a, %p, low = [1] : (tensor<4xf32>, tensor<f32>) -> tensor<6xf32>
                    stablehlo.return %0 : tensor<6xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void missingIotaDimension() {
            String mlir = """
                module @test {
                  func.func public @main() -> (tensor<4xf32>) {
                    %0 = stablehlo.iota : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse(mlir));
        }

        @Test
        void emptyIntegerList() {
            // Empty lists should be valid
            String mlir = """
                module @test {
                  func.func public @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DotGeneralOp dotOp = (DotGeneralOp) module.functions().get(0).body().get(0);
            assertTrue(dotOp.dimensionNumbers().lhsBatchingDimensions().isEmpty());
        }

        @Test
        void exceptionIncludesLineInfo() {
            String mlir = """
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %undefined : tensor<4xf32>
                  }
                }
                """;

            StableHloParseException ex = assertThrows(StableHloParseException.class,
                    () -> StableHloParser.parse(mlir));
            assertTrue(ex.getLine() > 0 || ex.getMessage().contains("line"));
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
