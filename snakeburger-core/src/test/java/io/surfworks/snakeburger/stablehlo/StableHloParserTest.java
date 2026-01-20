package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.AbsOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AfterAllOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ArrayAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllGatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllToAllOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AndOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Atan2Op;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormGradOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormInferenceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormTrainingOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BitcastConvertOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CaseOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CbrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CeilOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CholeskyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ClzOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ClampOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CollectiveBroadcastOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CollectivePermuteOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CompareOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComparisonDirection;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComplexOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CompositeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConcatenateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConstantOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConvertOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConvolutionOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CosOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DenseAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotDimensionNumbers;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotGeneralOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicUpdateSliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicBroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicGatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicIotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicPadOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicConvOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicSliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Expm1Op;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FftOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FloatAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FloorOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FunctionType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetDimensionSizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetTupleElementOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IfOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ImagOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.InfeedOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IntegerAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IsFiniteOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Log1pOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogisticOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MapOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MaximumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MinimumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NegateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NotOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OptimizationBarrierOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OrOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OutfeedOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PadOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PartitionIdOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PopcntOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PowerOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RealOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReducePrecisionOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RemainderOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RoundNearestAfzOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RoundNearestEvenOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RecvOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceWindowOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReplicaIdOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReturnOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReverseOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngAlgorithm;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngBitGeneratorOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScalarType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SelectAndScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SelectOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SendOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ShiftLeftOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ShiftRightArithmeticOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ShiftRightLogicalOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SignOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SinOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SortOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.StringAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanhOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TransposeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TriangularSolveOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TupleOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Type;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformDequantizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformQuantizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.snakeburger.stablehlo.StableHloAst.WhileOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.XorOp;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

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

    // ==================== Indexing Operations Tests ====================

    @Nested
    class IndexingOpsTests {

        @Test
        void parseGather() {
            String mlir = """
                module @test {
                  func.func public @gather(%operand: tensor<3x4x5xf32>, %indices: tensor<2x3xi32>) -> (tensor<2x3x5xf32>) {
                    %0 = stablehlo.gather %operand, %indices, offset_dims=[2], collapsed_slice_dims=[0], start_index_map=[0, 1], index_vector_dim=2, slice_sizes=[1, 1, 5] : (tensor<3x4x5xf32>, tensor<2x3xi32>) -> tensor<2x3x5xf32>
                    stablehlo.return %0 : tensor<2x3x5xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            GatherOp gatherOp = (GatherOp) module.functions().get(0).body().get(0);

            assertEquals("operand", gatherOp.operand().name());
            assertEquals("indices", gatherOp.startIndices().name());
            assertEquals(List.of(2L), gatherOp.offsetDims());
            assertEquals(List.of(0L), gatherOp.collapsedSliceDims());
            assertEquals(List.of(0L, 1L), gatherOp.startIndexMap());
            assertEquals(2, gatherOp.indexVectorDim());
            assertEquals(List.of(1L, 1L, 5L), gatherOp.sliceSizes());
            assertEquals("stablehlo.gather", gatherOp.opName());
        }

        @Test
        void parseScatter() {
            String mlir = """
                module @test {
                  func.func public @scatter(%operand: tensor<3x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<3x4xf32>) {
                    %0 = stablehlo.scatter %operand, %indices, %updates, update_window_dims=[1], inserted_window_dims=[0], scatter_dims_to_operand_dims=[0], index_vector_dim=1 : (tensor<3x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<3x4xf32>
                    stablehlo.return %0 : tensor<3x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ScatterOp scatterOp = (ScatterOp) module.functions().get(0).body().get(0);

            assertEquals("operand", scatterOp.operand().name());
            assertEquals("indices", scatterOp.scatterIndices().name());
            assertEquals("updates", scatterOp.updates().name());
            assertEquals(List.of(1L), scatterOp.updateWindowDims());
            assertEquals(List.of(0L), scatterOp.insertedWindowDims());
            assertEquals("stablehlo.scatter", scatterOp.opName());
        }

        @Test
        void parseDynamicSlice() {
            String mlir = """
                module @test {
                  func.func public @dslice(%operand: tensor<8x8xf32>, %start0: tensor<i32>, %start1: tensor<i32>) -> (tensor<4x4xf32>) {
                    %0 = stablehlo.dynamic_slice %operand, %start0, %start1, sizes=[4, 4] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
                    stablehlo.return %0 : tensor<4x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DynamicSliceOp dsliceOp = (DynamicSliceOp) module.functions().get(0).body().get(0);

            assertEquals("operand", dsliceOp.operand().name());
            assertEquals(2, dsliceOp.startIndices().size());
            assertEquals(List.of(4L, 4L), dsliceOp.sliceSizes());
            assertEquals("stablehlo.dynamic_slice", dsliceOp.opName());
        }

        @Test
        void parseDynamicUpdateSlice() {
            String mlir = """
                module @test {
                  func.func public @duslice(%operand: tensor<8x8xf32>, %update: tensor<4x4xf32>, %start0: tensor<i32>, %start1: tensor<i32>) -> (tensor<8x8xf32>) {
                    %0 = stablehlo.dynamic_update_slice %operand, %update, %start0, %start1 : (tensor<8x8xf32>, tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>
                    stablehlo.return %0 : tensor<8x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DynamicUpdateSliceOp dusliceOp = (DynamicUpdateSliceOp) module.functions().get(0).body().get(0);

            assertEquals("operand", dusliceOp.operand().name());
            assertEquals("update", dusliceOp.update().name());
            assertEquals(2, dusliceOp.startIndices().size());
            assertEquals("stablehlo.dynamic_update_slice", dusliceOp.opName());
        }
    }

    // ==================== Reduction Operations Tests ====================

    @Nested
    class ReductionOpsTests {

        @Test
        void parseReduce() {
            String mlir = """
                module @test {
                  func.func public @reduce(%operand: tensor<4x8xf32>, %init: tensor<f32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.reduce %operand, %init, dims=[1], reducer=add : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReduceOp reduceOp = (ReduceOp) module.functions().get(0).body().get(0);

            assertEquals("operand", reduceOp.operand().name());
            assertEquals("init", reduceOp.initValue().name());
            assertEquals(List.of(1L), reduceOp.dimensions());
            assertEquals("add", reduceOp.reducer());
            assertEquals("stablehlo.reduce", reduceOp.opName());
        }

        @Test
        void parseReduceWindow() {
            String mlir = """
                module @test {
                  func.func public @reduce_window(%operand: tensor<4x8xf32>, %init: tensor<f32>) -> (tensor<2x4xf32>) {
                    %0 = stablehlo.reduce_window %operand, %init, window=[2, 2], strides=[2, 2], reducer=max : (tensor<4x8xf32>, tensor<f32>) -> tensor<2x4xf32>
                    stablehlo.return %0 : tensor<2x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ReduceWindowOp rwOp = (ReduceWindowOp) module.functions().get(0).body().get(0);

            assertEquals("operand", rwOp.operand().name());
            assertEquals(List.of(2L, 2L), rwOp.windowDimensions());
            assertEquals(List.of(2L, 2L), rwOp.windowStrides());
            assertEquals("max", rwOp.reducer());
            assertEquals("stablehlo.reduce_window", rwOp.opName());
        }
    }

    // ==================== Neural Network Operations Tests ====================

    @Nested
    class NeuralNetworkOpsTests {

        @Test
        void parseConvolution() {
            String mlir = """
                module @test {
                  func.func public @conv(%lhs: tensor<1x28x28x1xf32>, %rhs: tensor<3x3x1x32xf32>) -> (tensor<1x26x26x32xf32>) {
                    %0 = stablehlo.convolution %lhs, %rhs, strides=[1, 1], padding_low=[0, 0], padding_high=[0, 0], feature_group_count=1, batch_group_count=1 : (tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>) -> tensor<1x26x26x32xf32>
                    stablehlo.return %0 : tensor<1x26x26x32xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ConvolutionOp convOp = (ConvolutionOp) module.functions().get(0).body().get(0);

            assertEquals("lhs", convOp.lhs().name());
            assertEquals("rhs", convOp.rhs().name());
            assertEquals(List.of(1L, 1L), convOp.windowStrides());
            assertEquals(1, convOp.featureGroupCount());
            assertEquals(1, convOp.batchGroupCount());
            assertEquals("stablehlo.convolution", convOp.opName());
        }

        @Test
        void parseBatchNormInference() {
            String mlir = """
                module @test {
                  func.func public @bn(%operand: tensor<4x8xf32>, %scale: tensor<8xf32>, %offset: tensor<8xf32>, %mean: tensor<8xf32>, %var: tensor<8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.batch_norm_inference %operand, %scale, %offset, %mean, %var, epsilon=0.00001, feature_index=1 : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            BatchNormInferenceOp bnOp = (BatchNormInferenceOp) module.functions().get(0).body().get(0);

            assertEquals("operand", bnOp.operand().name());
            assertEquals("scale", bnOp.scale().name());
            assertEquals("offset", bnOp.offset().name());
            assertEquals("mean", bnOp.mean().name());
            assertEquals("var", bnOp.variance().name());
            assertEquals(1, bnOp.featureIndex());
            assertEquals("stablehlo.batch_norm_inference", bnOp.opName());
        }
    }

    // ==================== Sort and RNG Tests ====================

    @Nested
    class SortAndRngOpsTests {

        @Test
        void parseSort() {
            String mlir = """
                module @test {
                  func.func public @sort(%input: tensor<8xf32>) -> (tensor<8xf32>) {
                    %0 = stablehlo.sort %input, dim=0, stable=true : (tensor<8xf32>) -> tensor<8xf32>
                    stablehlo.return %0 : tensor<8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            SortOp sortOp = (SortOp) module.functions().get(0).body().get(0);

            assertEquals(1, sortOp.inputs().size());
            assertEquals("input", sortOp.inputs().get(0).name());
            assertEquals(0, sortOp.dimension());
            assertTrue(sortOp.isStable());
            assertEquals("stablehlo.sort", sortOp.opName());
        }

        @Test
        void parseRng() {
            String mlir = """
                module @test {
                  func.func public @rng(%a: tensor<f32>, %b: tensor<f32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.rng %a, %b, distribution=uniform : (tensor<f32>, tensor<f32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            RngOp rngOp = (RngOp) module.functions().get(0).body().get(0);

            assertEquals("a", rngOp.a().name());
            assertEquals("b", rngOp.b().name());
            assertEquals("uniform", rngOp.distribution());
            assertEquals("stablehlo.rng", rngOp.opName());
        }
    }

    // ==================== Linear Algebra Tests ====================

    @Nested
    class LinearAlgebraOpsTests {

        @Test
        void parseDot() {
            String mlir = """
                module @test {
                  func.func public @dot(%lhs: tensor<4x8xf32>, %rhs: tensor<8xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.dot %lhs, %rhs : (tensor<4x8xf32>, tensor<8xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            DotOp dotOp = (DotOp) module.functions().get(0).body().get(0);

            assertEquals("lhs", dotOp.lhs().name());
            assertEquals("rhs", dotOp.rhs().name());
            assertEquals("stablehlo.dot", dotOp.opName());
        }

        @Test
        void parseCholesky() {
            String mlir = """
                module @test {
                  func.func public @cholesky(%a: tensor<4x4xf32>) -> (tensor<4x4xf32>) {
                    %0 = stablehlo.cholesky %a, lower=true : tensor<4x4xf32> -> tensor<4x4xf32>
                    stablehlo.return %0 : tensor<4x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            CholeskyOp cholOp = (CholeskyOp) module.functions().get(0).body().get(0);

            assertEquals("a", cholOp.operand().name());
            assertTrue(cholOp.lower());
            assertEquals("stablehlo.cholesky", cholOp.opName());
        }

        @Test
        void parseTriangularSolve() {
            String mlir = """
                module @test {
                  func.func public @tsolve(%a: tensor<4x4xf32>, %b: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.triangular_solve %a, %b, left_side=true, lower=true, transpose_a=false : (tensor<4x4xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            TriangularSolveOp tsOp = (TriangularSolveOp) module.functions().get(0).body().get(0);

            assertEquals("a", tsOp.a().name());
            assertEquals("b", tsOp.b().name());
            assertTrue(tsOp.leftSide());
            assertTrue(tsOp.lower());
            assertFalse(tsOp.transposeA());
            assertEquals("stablehlo.triangular_solve", tsOp.opName());
        }
    }

    // ==================== Complex Number Tests ====================

    @Nested
    class ComplexNumberOpsTests {

        @Test
        void parseReal() {
            String mlir = """
                module @test {
                  func.func public @real(%c: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.real %c : tensor<4xf32> -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            RealOp realOp = (RealOp) module.functions().get(0).body().get(0);

            assertEquals("c", realOp.operand().name());
            assertEquals("stablehlo.real", realOp.opName());
        }

        @Test
        void parseImag() {
            String mlir = """
                module @test {
                  func.func public @imag(%c: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.imag %c : tensor<4xf32> -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ImagOp imagOp = (ImagOp) module.functions().get(0).body().get(0);

            assertEquals("c", imagOp.operand().name());
            assertEquals("stablehlo.imag", imagOp.opName());
        }

        @Test
        void parseComplex() {
            String mlir = """
                module @test {
                  func.func public @complex(%r: tensor<4xf32>, %i: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.complex %r, %i : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            ComplexOp complexOp = (ComplexOp) module.functions().get(0).body().get(0);

            assertEquals("r", complexOp.real().name());
            assertEquals("i", complexOp.imag().name());
            assertEquals("stablehlo.complex", complexOp.opName());
        }
    }

    // ==================== Signal Processing Tests ====================

    @Nested
    class SignalProcessingOpsTests {

        @Test
        void parseFft() {
            String mlir = """
                module @test {
                  func.func public @fft(%input: tensor<8xf32>) -> (tensor<8xf32>) {
                    %0 = stablehlo.fft %input, type=FFT, length=[8] : tensor<8xf32> -> tensor<8xf32>
                    stablehlo.return %0 : tensor<8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            FftOp fftOp = (FftOp) module.functions().get(0).body().get(0);

            assertEquals("input", fftOp.operand().name());
            assertEquals("FFT", fftOp.fftType());
            assertEquals(List.of(8L), fftOp.fftLength());
            assertEquals("stablehlo.fft", fftOp.opName());
        }
    }

    // ==================== Custom Call Tests ====================

    @Nested
    class CustomCallOpsTests {

        @Test
        void parseCustomCall() {
            String mlir = """
                module @test {
                  func.func public @custom(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.custom_call @my_op(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            CustomCallOp customOp = (CustomCallOp) module.functions().get(0).body().get(0);

            assertEquals("my_op", customOp.callTarget());
            assertEquals(2, customOp.inputs().size());
            assertEquals("a", customOp.inputs().get(0).name());
            assertEquals("b", customOp.inputs().get(1).name());
            assertEquals("stablehlo.custom_call", customOp.opName());
        }
    }

    // ==================== Integration Tests - Complex ML Patterns ====================

    @Nested
    class ComplexMLPatternTests {

        @Test
        void parseTransformerAttentionPattern() {
            // Simplified attention: softmax(Q*K^T / sqrt(d)) * V
            String mlir = """
                module @attention {
                  func.func public @attention(%Q: tensor<4x8x16xf32>, %K: tensor<4x8x16xf32>, %V: tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>) {
                    %scale = stablehlo.constant dense<0.25> : tensor<4x8x8xf32>
                    %0 = stablehlo.dot_general %Q, %K, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]> : (tensor<4x8x16xf32>, tensor<4x8x16xf32>) -> tensor<4x8x8xf32>
                    %1 = stablehlo.multiply %0, %scale : tensor<4x8x8xf32>
                    %2 = stablehlo.exponential %1 : tensor<4x8x8xf32>
                    %3 = stablehlo.dot_general %2, %V, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]> : (tensor<4x8x8xf32>, tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
                    stablehlo.return %3 : tensor<4x8x16xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals("attention", func.name());
            assertEquals(3, func.arguments().size());
            assertEquals(6, func.body().size());  // constant, dot, mul, exp, dot, return

            // Verify the pattern
            assertInstanceOf(ConstantOp.class, func.body().get(0));
            assertInstanceOf(DotGeneralOp.class, func.body().get(1));
            assertInstanceOf(MultiplyOp.class, func.body().get(2));
            assertInstanceOf(ExpOp.class, func.body().get(3));
            assertInstanceOf(DotGeneralOp.class, func.body().get(4));
            assertInstanceOf(ReturnOp.class, func.body().get(5));
        }

        @Test
        void parseResNetBlockPattern() {
            // Simplified ResNet block: x + F(x)
            String mlir = """
                module @resnet {
                  func.func public @residual(%input: tensor<4x8xf32>, %weight: tensor<8x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.dot_general %input, %weight, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
                    %zero = stablehlo.constant dense<0.0> : tensor<4x8xf32>
                    %relu = stablehlo.maximum %0, %zero : tensor<4x8xf32>
                    %residual = stablehlo.add %input, %relu : tensor<4x8xf32>
                    stablehlo.return %residual : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals(5, func.body().size());
            assertInstanceOf(DotGeneralOp.class, func.body().get(0));
            assertInstanceOf(ConstantOp.class, func.body().get(1));
            assertInstanceOf(MaximumOp.class, func.body().get(2));
            assertInstanceOf(AddOp.class, func.body().get(3));

            // Verify residual connection
            AddOp addOp = (AddOp) func.body().get(3);
            assertEquals("input", addOp.lhs().name());  // Original input
            assertEquals("relu", addOp.rhs().name());    // ReLU output
        }

        @Test
        void parseGeluActivationPattern() {
            // GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            String mlir = """
                module @gelu {
                  func.func public @gelu(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %half = stablehlo.constant dense<0.5> : tensor<4x8xf32>
                    %one = stablehlo.constant dense<1.0> : tensor<4x8xf32>
                    %coeff = stablehlo.constant dense<0.044715> : tensor<4x8xf32>
                    %sqrt2pi = stablehlo.constant dense<0.7978845> : tensor<4x8xf32>
                    %x2 = stablehlo.multiply %x, %x : tensor<4x8xf32>
                    %x3 = stablehlo.multiply %x2, %x : tensor<4x8xf32>
                    %cx3 = stablehlo.multiply %coeff, %x3 : tensor<4x8xf32>
                    %xcx3 = stablehlo.add %x, %cx3 : tensor<4x8xf32>
                    %scaled = stablehlo.multiply %sqrt2pi, %xcx3 : tensor<4x8xf32>
                    %tanh_out = stablehlo.tanh %scaled : tensor<4x8xf32>
                    %tanh_plus_one = stablehlo.add %one, %tanh_out : tensor<4x8xf32>
                    %half_x = stablehlo.multiply %half, %x : tensor<4x8xf32>
                    %result = stablehlo.multiply %half_x, %tanh_plus_one : tensor<4x8xf32>
                    stablehlo.return %result : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals("gelu", func.name());
            assertEquals(14, func.body().size());

            // Verify we have the expected ops
            long tanhCount = func.body().stream().filter(op -> op instanceof TanhOp).count();
            long mulCount = func.body().stream().filter(op -> op instanceof MultiplyOp).count();
            long addCount = func.body().stream().filter(op -> op instanceof AddOp).count();

            assertEquals(1, tanhCount);
            assertEquals(6, mulCount);
            assertEquals(2, addCount);
        }

        @Test
        void parseBatchNormPattern() {
            // BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
            String mlir = """
                module @batchnorm {
                  func.func public @bn(%x: tensor<4x8xf32>, %mean: tensor<8xf32>, %var: tensor<8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>) -> (tensor<4x8xf32>) {
                    %eps = stablehlo.constant dense<1.0e-5> : tensor<8xf32>
                    %x_broadcast = stablehlo.broadcast_in_dim %mean, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
                    %centered = stablehlo.subtract %x, %x_broadcast : tensor<4x8xf32>
                    %var_eps = stablehlo.add %var, %eps : tensor<8xf32>
                    %std = stablehlo.sqrt %var_eps : tensor<8xf32>
                    %std_broadcast = stablehlo.broadcast_in_dim %std, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
                    %normalized = stablehlo.divide %centered, %std_broadcast : tensor<4x8xf32>
                    %gamma_broadcast = stablehlo.broadcast_in_dim %gamma, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
                    %scaled = stablehlo.multiply %normalized, %gamma_broadcast : tensor<4x8xf32>
                    %beta_broadcast = stablehlo.broadcast_in_dim %beta, dims = [1] : (tensor<8xf32>) -> tensor<4x8xf32>
                    %result = stablehlo.add %scaled, %beta_broadcast : tensor<4x8xf32>
                    stablehlo.return %result : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals("bn", func.name());
            assertEquals(5, func.arguments().size());

            // Verify broadcast operations
            long broadcastCount = func.body().stream()
                    .filter(op -> op instanceof BroadcastInDimOp)
                    .count();
            assertEquals(4, broadcastCount);

            // Verify sqrt for normalization
            long sqrtCount = func.body().stream()
                    .filter(op -> op instanceof SqrtOp)
                    .count();
            assertEquals(1, sqrtCount);
        }
    }

    // ==================== Stress Tests ====================

    @Nested
    class StressTests {

        @Test
        void parseManyOperations() {
            // Generate a function with 100 operations
            StringBuilder mlir = new StringBuilder();
            mlir.append("module @stress {\n");
            mlir.append("  func.func public @many_ops(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n");

            String prevVar = "arg0";
            for (int i = 0; i < 100; i++) {
                String op = i % 2 == 0 ? "negate" : "abs";
                mlir.append(String.format("    %%v%d = stablehlo.%s %%%s : tensor<4xf32>\n", i, op, prevVar));
                prevVar = "v" + i;
            }
            mlir.append(String.format("    stablehlo.return %%%s : tensor<4xf32>\n", prevVar));
            mlir.append("  }\n");
            mlir.append("}\n");

            Module module = StableHloParser.parse(mlir.toString());
            Function func = module.functions().get(0);

            assertEquals(101, func.body().size());  // 100 ops + return
        }

        @Test
        void parseDeepNesting() {
            // Deep chain of operations
            StringBuilder mlir = new StringBuilder();
            mlir.append("module @deep {\n");
            mlir.append("  func.func public @deep_chain(%x: tensor<4xf32>) -> (tensor<4xf32>) {\n");

            String prevVar = "x";
            for (int i = 0; i < 50; i++) {
                mlir.append(String.format("    %%%d = stablehlo.exponential %%%s : tensor<4xf32>\n", i, prevVar));
                prevVar = String.valueOf(i);
            }
            for (int i = 50; i < 100; i++) {
                mlir.append(String.format("    %%%d = stablehlo.log %%%s : tensor<4xf32>\n", i, prevVar));
                prevVar = String.valueOf(i);
            }

            mlir.append(String.format("    stablehlo.return %%%s : tensor<4xf32>\n", prevVar));
            mlir.append("  }\n");
            mlir.append("}\n");

            Module module = StableHloParser.parse(mlir.toString());
            Function func = module.functions().get(0);

            assertEquals(101, func.body().size());

            // First 50 should be exp, last 50 should be log
            assertInstanceOf(ExpOp.class, func.body().get(0));
            assertInstanceOf(ExpOp.class, func.body().get(49));
            assertInstanceOf(LogOp.class, func.body().get(50));
            assertInstanceOf(LogOp.class, func.body().get(99));
        }

        @Test
        void parseLargeTensorShapes() {
            String mlir = """
                module @large {
                  func.func public @large_tensors(%a: tensor<1024x1024x1024xf32>) -> (tensor<1024x1024x1024xf32>) {
                    %0 = stablehlo.negate %a : tensor<1024x1024x1024xf32>
                    stablehlo.return %0 : tensor<1024x1024x1024xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            TensorType inputType = (TensorType) func.arguments().get(0).type();

            assertEquals(List.of(1024, 1024, 1024), inputType.shape());
            assertEquals(3, inputType.rank());
        }

        @Test
        void parseManyArguments() {
            // Function with 20 arguments
            StringBuilder mlir = new StringBuilder();
            mlir.append("module @many_args {\n");
            mlir.append("  func.func public @f(");
            for (int i = 0; i < 20; i++) {
                if (i > 0) mlir.append(", ");
                mlir.append("%arg").append(i).append(": tensor<4xf32>");
            }
            mlir.append(") -> (tensor<4xf32>) {\n");
            mlir.append("    stablehlo.return %arg0 : tensor<4xf32>\n");
            mlir.append("  }\n");
            mlir.append("}\n");

            Module module = StableHloParser.parse(mlir.toString());
            Function func = module.functions().get(0);

            assertEquals(20, func.arguments().size());
        }
    }

    // ==================== Dynamic Shape Operations Tests ====================

    @Nested
    class DynamicShapeOpsTests {

        @Test
        void parseDynamicBroadcastInDim() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4xf32>, %shape: tensor<2xi32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.dynamic_broadcast_in_dim %x, %shape, broadcast_dimensions = [0] : (tensor<4xf32>, tensor<2xi32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(DynamicBroadcastInDimOp.class, func.body().get(0));

            DynamicBroadcastInDimOp op = (DynamicBroadcastInDimOp) func.body().get(0);
            assertEquals("x", op.operand().name());
            assertEquals("shape", op.outputDimensions().name());
            assertEquals(List.of(0L), op.broadcastDimensions());
        }

        @Test
        void parseDynamicGather() {
            String mlir = """
                module @test {
                  func.func public @f(%data: tensor<8x4xf32>, %indices: tensor<2x1xi32>, %sizes: tensor<2xi32>) -> (tensor<2x4xf32>) {
                    %0 = stablehlo.dynamic_gather %data, %indices, %sizes, offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1 : (tensor<8x4xf32>, tensor<2x1xi32>, tensor<2xi32>) -> tensor<2x4xf32>
                    stablehlo.return %0 : tensor<2x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(DynamicGatherOp.class, func.body().get(0));

            DynamicGatherOp op = (DynamicGatherOp) func.body().get(0);
            assertEquals("data", op.operand().name());
            assertEquals(List.of(1L), op.offsetDims());
            assertEquals(List.of(0L), op.collapsedSliceDims());
            assertEquals(1L, op.indexVectorDim());
        }

        @Test
        void parseDynamicIota() {
            String mlir = """
                module @test {
                  func.func public @f(%shape: tensor<2xi32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.dynamic_iota %shape, dim = 1 : tensor<2xi32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(DynamicIotaOp.class, func.body().get(0));

            DynamicIotaOp op = (DynamicIotaOp) func.body().get(0);
            assertEquals(1L, op.iotaDimension());
        }

        @Test
        void parseDynamicPad() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>, %val: tensor<f32>, %low: tensor<2xi32>, %high: tensor<2xi32>, %interior: tensor<2xi32>) -> (tensor<6x12xf32>) {
                    %0 = stablehlo.dynamic_pad %x, %val, %low, %high, %interior : (tensor<4x8xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<6x12xf32>
                    stablehlo.return %0 : tensor<6x12xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(DynamicPadOp.class, func.body().get(0));

            DynamicPadOp op = (DynamicPadOp) func.body().get(0);
            assertEquals("x", op.operand().name());
            assertEquals("val", op.paddingValue().name());
        }

        @Test
        void parseDynamicReshape() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>, %shape: tensor<1xi32>) -> (tensor<32xf32>) {
                    %0 = stablehlo.dynamic_reshape %x, %shape : (tensor<4x8xf32>, tensor<1xi32>) -> tensor<32xf32>
                    stablehlo.return %0 : tensor<32xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(DynamicReshapeOp.class, func.body().get(0));

            DynamicReshapeOp op = (DynamicReshapeOp) func.body().get(0);
            assertEquals("x", op.operand().name());
            assertEquals("shape", op.outputShape().name());
        }

        @Test
        void parseGetDimensionSize() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<i32>) {
                    %0 = stablehlo.get_dimension_size %x, dim = 1 : tensor<4x8xf32> -> tensor<i32>
                    stablehlo.return %0 : tensor<i32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(GetDimensionSizeOp.class, func.body().get(0));

            GetDimensionSizeOp op = (GetDimensionSizeOp) func.body().get(0);
            assertEquals("x", op.operand().name());
            assertEquals(1L, op.dimension());
        }
    }

    // ==================== Quantization Operations Tests ====================

    @Nested
    class QuantizationOpsTests {

        @Test
        void parseUniformQuantize() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<4x8xi8>) {
                    %0 = stablehlo.uniform_quantize %x : tensor<4x8xf32> -> tensor<4x8xi8>
                    stablehlo.return %0 : tensor<4x8xi8>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(UniformQuantizeOp.class, func.body().get(0));

            UniformQuantizeOp op = (UniformQuantizeOp) func.body().get(0);
            assertEquals("x", op.operand().name());
        }

        @Test
        void parseUniformDequantize() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xi8>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.uniform_dequantize %x : tensor<4x8xi8> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(UniformDequantizeOp.class, func.body().get(0));

            UniformDequantizeOp op = (UniformDequantizeOp) func.body().get(0);
            assertEquals("x", op.operand().name());
        }
    }

    // ==================== Additional Reduction Operations Tests ====================

    @Nested
    class AdditionalReductionOpsTests {

        @Test
        void parseReducePrecision() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.reduce_precision %x, exponent_bits = 5, mantissa_bits = 10 : tensor<4x8xf32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(ReducePrecisionOp.class, func.body().get(0));

            ReducePrecisionOp op = (ReducePrecisionOp) func.body().get(0);
            assertEquals(5, op.exponentBits());
            assertEquals(10, op.mantissaBits());
        }

        @Test
        void parseSelectAndScatter() {
            String mlir = """
                module @test {
                  func.func public @f(%operand: tensor<4x8xf32>, %source: tensor<2x4xf32>, %init: tensor<f32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.select_and_scatter %operand, %source, %init, window = [2, 2], strides = [2, 2], padding = [0, 0, 0, 0], select = ge, scatter = add : (tensor<4x8xf32>, tensor<2x4xf32>, tensor<f32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(SelectAndScatterOp.class, func.body().get(0));

            SelectAndScatterOp op = (SelectAndScatterOp) func.body().get(0);
            assertEquals("ge", op.selectFn());
            assertEquals("add", op.scatterFn());
        }
    }

    // ==================== Control Flow Operations Tests ====================

    @Nested
    class AdditionalControlFlowOpsTests {

        @Test
        void parseCase() {
            String mlir = """
                module @test {
                  func.func public @f(%idx: tensor<i32>, %x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.case %idx {
                      %a = stablehlo.negate %x : tensor<4xf32>
                      stablehlo.return %a : tensor<4xf32>
                    } {
                      %b = stablehlo.abs %x : tensor<4xf32>
                      stablehlo.return %b : tensor<4xf32>
                    } : (tensor<i32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(CaseOp.class, func.body().get(0));

            CaseOp op = (CaseOp) func.body().get(0);
            assertEquals("idx", op.index().name());
            assertEquals(2, op.branches().size());
        }

        @Test
        void parseMap() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.map %x, dims = [0, 1] {
                      %a = stablehlo.negate %x : tensor<4x8xf32>
                      stablehlo.return %a : tensor<4x8xf32>
                    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(MapOp.class, func.body().get(0));

            MapOp op = (MapOp) func.body().get(0);
            assertEquals(List.of(0L, 1L), op.dimensions());
        }
    }

    // ==================== Distributed/Collective Operations Tests ====================

    @Nested
    class DistributedOpsTests {

        @Test
        void parsePartitionId() {
            String mlir = """
                module @test {
                  func.func public @f() -> (tensor<ui32>) {
                    %0 = stablehlo.partition_id : () -> tensor<ui32>
                    stablehlo.return %0 : tensor<ui32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(PartitionIdOp.class, func.body().get(0));
        }

        @Test
        void parseReplicaId() {
            String mlir = """
                module @test {
                  func.func public @f() -> (tensor<ui32>) {
                    %0 = stablehlo.replica_id : () -> tensor<ui32>
                    stablehlo.return %0 : tensor<ui32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(ReplicaIdOp.class, func.body().get(0));
        }

        @Test
        void parseAllGather() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<8x8xf32>) {
                    %0 = stablehlo.all_gather %x, all_gather_dim = 0, replica_groups = [[0, 1]], channel_id = 1 : tensor<4x8xf32> -> tensor<8x8xf32>
                    stablehlo.return %0 : tensor<8x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(AllGatherOp.class, func.body().get(0));

            AllGatherOp op = (AllGatherOp) func.body().get(0);
            assertEquals(0L, op.allGatherDim());
            assertEquals(1L, op.channelId());
        }

        @Test
        void parseAllReduce() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.all_reduce %x, replica_groups = [[0, 1]], channel_id = 1, reducer = add : tensor<4x8xf32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(AllReduceOp.class, func.body().get(0));

            AllReduceOp op = (AllReduceOp) func.body().get(0);
            assertEquals("add", op.reducer());
        }

        @Test
        void parseAllToAll() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.all_to_all %x, split_dimension = 1, concat_dimension = 0, split_count = 2, replica_groups = [[0, 1]] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(AllToAllOp.class, func.body().get(0));

            AllToAllOp op = (AllToAllOp) func.body().get(0);
            assertEquals(1L, op.splitDimension());
            assertEquals(0L, op.concatDimension());
            assertEquals(2L, op.splitCount());
        }

        @Test
        void parseCollectiveBroadcast() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.collective_broadcast %x, replica_groups = [[0, 1]], channel_id = 1 : tensor<4x8xf32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(CollectiveBroadcastOp.class, func.body().get(0));
        }

        @Test
        void parseCollectivePermute() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.collective_permute %x, source_target_pairs = [[0, 1], [1, 0]], channel_id = 1 : tensor<4x8xf32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(CollectivePermuteOp.class, func.body().get(0));

            CollectivePermuteOp op = (CollectivePermuteOp) func.body().get(0);
            assertEquals(2, op.sourceTargetPairs().size());
        }

        @Test
        void parseReduceScatter() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x8xf32>) -> (tensor<2x8xf32>) {
                    %0 = stablehlo.reduce_scatter %x, scatter_dimension = 0, replica_groups = [[0, 1]], channel_id = 1, reducer = add : tensor<4x8xf32> -> tensor<2x8xf32>
                    stablehlo.return %0 : tensor<2x8xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(ReduceScatterOp.class, func.body().get(0));

            ReduceScatterOp op = (ReduceScatterOp) func.body().get(0);
            assertEquals(0L, op.scatterDimension());
            assertEquals("add", op.reducer());
        }
    }

    // ==================== Tuple Operations Tests ====================

    @Nested
    class TupleOpsTests {

        @Test
        void parseTuple() {
            String mlir = """
                module @test {
                  func.func public @f(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>) {
                    %t = stablehlo.tuple %a, %b : (tensor<4xf32>, tensor<8xf32>) -> tensor<4xf32>
                    stablehlo.return %t : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(TupleOp.class, func.body().get(0));

            TupleOp op = (TupleOp) func.body().get(0);
            assertEquals(2, op.inputs().size());
            assertEquals("a", op.inputs().get(0).name());
            assertEquals("b", op.inputs().get(1).name());
        }

        @Test
        void parseGetTupleElement() {
            String mlir = """
                module @test {
                  func.func public @f(%t: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.get_tuple_element %t, index = 0 : (tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(GetTupleElementOp.class, func.body().get(0));

            GetTupleElementOp op = (GetTupleElementOp) func.body().get(0);
            assertEquals("t", op.operand().name());
            assertEquals(0, op.index());
        }
    }

    // ==================== Other Operations Tests ====================

    @Nested
    class OtherOpsTests {

        @Test
        void parseOptimizationBarrier() {
            String mlir = """
                module @test {
                  func.func public @f(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.optimization_barrier %a, %b : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(OptimizationBarrierOp.class, func.body().get(0));

            OptimizationBarrierOp op = (OptimizationBarrierOp) func.body().get(0);
            assertEquals(2, op.operands().size());
        }

        @Test
        void parseComposite() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.composite %x, name = "my_custom_op", decomposition = @my_decomp, version = 1 : (tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(CompositeOp.class, func.body().get(0));

            CompositeOp op = (CompositeOp) func.body().get(0);
            assertEquals("my_custom_op", op.name());
            assertEquals("my_decomp", op.decomposition());
            assertEquals(1L, op.version());
        }
    }

    // ==================== Distributed Training Pattern Test ====================

    @Nested
    class DistributedPatternTests {

        @Test
        void parseDataParallelAllReduce() {
            // Pattern: gradient -> all_reduce -> update
            String mlir = """
                module @data_parallel {
                  func.func public @gradient_sync(%grad: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>) {
                    %0 = stablehlo.all_reduce %grad, replica_groups = [[0, 1, 2, 3]], channel_id = 1, reducer = add : tensor<1024x1024xf32> -> tensor<1024x1024xf32>
                    %num_replicas = stablehlo.constant dense<4.0> : tensor<1024x1024xf32>
                    %avg = stablehlo.divide %0, %num_replicas : tensor<1024x1024xf32>
                    stablehlo.return %avg : tensor<1024x1024xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            assertEquals("gradient_sync", func.name());
            assertInstanceOf(AllReduceOp.class, func.body().get(0));
            assertInstanceOf(ConstantOp.class, func.body().get(1));
            assertInstanceOf(DivideOp.class, func.body().get(2));
        }

        @Test
        void parseModelParallelGather() {
            // Pattern: distributed embeddings with all_gather
            String mlir = """
                module @model_parallel {
                  func.func public @gather_embeddings(%local: tensor<256x512xf32>) -> (tensor<1024x512xf32>) {
                    %0 = stablehlo.all_gather %local, all_gather_dim = 0, replica_groups = [[0, 1, 2, 3]], channel_id = 1 : tensor<256x512xf32> -> tensor<1024x512xf32>
                    stablehlo.return %0 : tensor<1024x512xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);

            AllGatherOp op = (AllGatherOp) func.body().get(0);
            assertEquals(0L, op.allGatherDim());
        }

        @Test
        void parseAfterAll() {
            String mlir = """
                module @test {
                  func.func public @f(%t1: tensor<ui32>, %t2: tensor<ui32>) -> (tensor<ui32>) {
                    %0 = stablehlo.after_all %t1, %t2 : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
                    stablehlo.return %0 : tensor<ui32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(AfterAllOp.class, func.body().get(0));

            AfterAllOp op = (AfterAllOp) func.body().get(0);
            assertEquals(2, op.inputs().size());
        }
    }

    // ==================== Control Flow Operations Tests ====================

    @Nested
    class ControlFlowOpsTests {

        @Test
        void parseIf() {
            String mlir = """
                module @test {
                  func.func public @f(%pred: tensor<i1>, %x: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.if %pred {
                      %a = stablehlo.negate %x : tensor<4xf32>
                      stablehlo.return %a : tensor<4xf32>
                    } else {
                      %b = stablehlo.abs %x : tensor<4xf32>
                      stablehlo.return %b : tensor<4xf32>
                    } : (tensor<i1>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(IfOp.class, func.body().get(0));

            IfOp op = (IfOp) func.body().get(0);
            assertEquals("pred", op.pred().name());
            assertFalse(op.trueBranch().isEmpty());
            assertFalse(op.falseBranch().isEmpty());
        }

        @Test
        void parseWhile() {
            // Note: Using numeric constant 1 for true in i1 since parser doesn't handle dense<true/false>
            String mlir = """
                module @test {
                  func.func public @f(%init: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.while %init {
                      %cond = stablehlo.constant dense<1> : tensor<i1>
                      stablehlo.return %cond : tensor<i1>
                    } do {
                      %next = stablehlo.negate %init : tensor<4xf32>
                      stablehlo.return %next : tensor<4xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(WhileOp.class, func.body().get(0));

            WhileOp op = (WhileOp) func.body().get(0);
            assertFalse(op.condBody().isEmpty());
            assertFalse(op.body().isEmpty());
        }
    }

    // ==================== Neural Network Gradient Operations Tests ====================

    @Nested
    class NeuralNetGradOpsTests {

        @Test
        void parseBatchNormTraining() {
            String mlir = """
                module @test {
                  func.func public @f(%x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>) -> (tensor<4x256xf32>) {
                    %out = stablehlo.batch_norm_training %x, %scale, %offset, epsilon=1.0e-5, feature_index=1 : (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>)
                    stablehlo.return %out : tensor<4x256xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(BatchNormTrainingOp.class, func.body().get(0));

            BatchNormTrainingOp op = (BatchNormTrainingOp) func.body().get(0);
            assertEquals(1.0e-5f, op.epsilon(), 1e-10);
            assertEquals(1L, op.featureIndex());
            assertEquals(3, op.results().size());
        }

        @Test
        void parseBatchNormGrad() {
            String mlir = """
                module @test {
                  func.func public @f(%operand: tensor<4x256xf32>, %scale: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>, %grad_output: tensor<4x256xf32>) -> (tensor<4x256xf32>) {
                    %grad = stablehlo.batch_norm_grad %operand, %scale, %mean, %variance, %grad_output, epsilon=1.0e-5, feature_index=1 : (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<4x256xf32>) -> (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>)
                    stablehlo.return %grad : tensor<4x256xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(BatchNormGradOp.class, func.body().get(0));

            BatchNormGradOp op = (BatchNormGradOp) func.body().get(0);
            assertEquals(1.0e-5f, op.epsilon(), 1e-10);
            assertEquals(1L, op.featureIndex());
            assertEquals(3, op.results().size());
        }
    }

    // ==================== Dynamic Shape Operations Tests ====================

    @Nested
    class DynamicConvOpsTests {

        @Test
        void parseDynamicConv() {
            String mlir = """
                module @test {
                  func.func public @f(%input: tensor<1x4x4x1xf32>, %kernel: tensor<3x3x1x1xf32>, %padding: tensor<4xi32>) -> (tensor<1x4x4x1xf32>) {
                    %0 = stablehlo.dynamic_conv %input, %kernel, %padding, strides=[1, 1], lhs_dilation=[1, 1], rhs_dilation=[1, 1], feature_group_count=1, batch_group_count=1 : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>, tensor<4xi32>) -> tensor<1x4x4x1xf32>
                    stablehlo.return %0 : tensor<1x4x4x1xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(DynamicConvOp.class, func.body().get(0));

            DynamicConvOp op = (DynamicConvOp) func.body().get(0);
            assertEquals(List.of(1L, 1L), op.windowStrides());
            assertEquals(1L, op.featureGroupCount());
        }
    }

    // ==================== Random Operations Tests ====================

    @Nested
    class RandomOpsTests {

        @Test
        void parseRngBitGenerator() {
            String mlir = """
                module @test {
                  func.func public @f(%state: tensor<2xui64>) -> (tensor<2xui64>) {
                    %out = stablehlo.rng_bit_generator %state, algorithm=DEFAULT : tensor<2xui64> -> (tensor<2xui64>, tensor<4x4xui32>)
                    stablehlo.return %out : tensor<2xui64>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(RngBitGeneratorOp.class, func.body().get(0));

            RngBitGeneratorOp op = (RngBitGeneratorOp) func.body().get(0);
            assertEquals(RngAlgorithm.DEFAULT, op.algorithm());
            assertEquals(2, op.results().size());
        }

        @Test
        void parseRngBitGeneratorThreeFry() {
            String mlir = """
                module @test {
                  func.func public @f(%state: tensor<2xui64>) -> (tensor<2xui64>) {
                    %out = stablehlo.rng_bit_generator %state, algorithm=THREE_FRY : tensor<2xui64> -> (tensor<2xui64>, tensor<8xui64>)
                    stablehlo.return %out : tensor<2xui64>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            RngBitGeneratorOp op = (RngBitGeneratorOp) func.body().get(0);
            assertEquals(RngAlgorithm.THREE_FRY, op.algorithm());
        }

        @Test
        void parseRngBitGeneratorPhilox() {
            String mlir = """
                module @test {
                  func.func public @f(%state: tensor<3xui64>) -> (tensor<3xui64>) {
                    %out = stablehlo.rng_bit_generator %state, algorithm=PHILOX : tensor<3xui64> -> (tensor<3xui64>, tensor<16xui32>)
                    stablehlo.return %out : tensor<3xui64>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            RngBitGeneratorOp op = (RngBitGeneratorOp) func.body().get(0);
            assertEquals(RngAlgorithm.PHILOX, op.algorithm());
        }
    }

    // ==================== Communication Operations Tests ====================

    @Nested
    class CommunicationOpsTests {

        @Test
        void parseInfeed() {
            String mlir = """
                module @test {
                  func.func public @f(%token: tensor<ui32>) -> (tensor<4xf32>) {
                    %data = stablehlo.infeed %token, infeed_config="config_string" : tensor<ui32> -> (tensor<4xf32>, tensor<ui32>)
                    stablehlo.return %data : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(InfeedOp.class, func.body().get(0));

            InfeedOp op = (InfeedOp) func.body().get(0);
            assertEquals("config_string", op.infeedConfig());
            assertEquals("token", op.token().name());
        }

        @Test
        void parseOutfeed() {
            String mlir = """
                module @test {
                  func.func public @f(%data: tensor<4xf32>, %token: tensor<ui32>) -> (tensor<ui32>) {
                    %token_out = stablehlo.outfeed %data, %token, outfeed_config="out_config" : (tensor<4xf32>, tensor<ui32>) -> tensor<ui32>
                    stablehlo.return %token_out : tensor<ui32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(OutfeedOp.class, func.body().get(0));

            OutfeedOp op = (OutfeedOp) func.body().get(0);
            assertEquals("out_config", op.outfeedConfig());
        }

        @Test
        void parseRecv() {
            String mlir = """
                module @test {
                  func.func public @f(%token: tensor<ui32>) -> (tensor<4xf32>) {
                    %data = stablehlo.recv %token, channel_id=42, channel_type=host_to_device : tensor<ui32> -> (tensor<4xf32>, tensor<ui32>)
                    stablehlo.return %data : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(RecvOp.class, func.body().get(0));

            RecvOp op = (RecvOp) func.body().get(0);
            assertEquals(42L, op.channelId());
            assertEquals("host_to_device", op.channelType());
        }

        @Test
        void parseSend() {
            String mlir = """
                module @test {
                  func.func public @f(%data: tensor<4xf32>, %token: tensor<ui32>) -> (tensor<ui32>) {
                    %token_out = stablehlo.send %data, %token, channel_id=99, channel_type=device_to_host : (tensor<4xf32>, tensor<ui32>) -> tensor<ui32>
                    stablehlo.return %token_out : tensor<ui32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            Function func = module.functions().get(0);
            assertInstanceOf(SendOp.class, func.body().get(0));

            SendOp op = (SendOp) func.body().get(0);
            assertEquals(99L, op.channelId());
            assertEquals("device_to_host", op.channelType());
        }
    }
}
