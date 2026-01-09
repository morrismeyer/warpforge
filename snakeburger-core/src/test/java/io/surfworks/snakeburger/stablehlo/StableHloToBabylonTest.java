package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloToBabylon.EmitResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

class StableHloToBabylonTest {

    private EmitResult emit(String mlir) {
        Module module = StableHloParser.parse(mlir);
        return new StableHloToBabylon().emit(module);
    }

    @Nested
    class BasicEmissionTests {

        @Test
        void emitSimplePassthrough() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);

            assertNotNull(result.babylonText());
            assertTrue(result.babylonText().contains("func @main"));
            assertTrue(result.babylonText().contains("return %arg0"));
        }

        @Test
        void emitResultContainsModuleReference() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);

            assertNotNull(result.stableHloModule());
            assertEquals("test", result.stableHloModule().name());
        }

        @Test
        void emitResultContainsOperationsList() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            assertNotNull(result.operations());
            assertEquals(2, result.operations().size()); // add + return
        }
    }

    @Nested
    class FunctionSignatureTests {

        @Test
        void emitFunctionWithSingleArg() {
            EmitResult result = emit("""
                module @test {
                  func.func public @single(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("%arg0: tensor<4xf32>"));
        }

        @Test
        void emitFunctionWithMultipleArgs() {
            EmitResult result = emit("""
                module @test {
                  func.func public @multi(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %a : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("%a: tensor<4xf32>"));
            assertTrue(result.babylonText().contains("%b: tensor<8xf32>"));
        }

        @Test
        void emitFunctionWithMultipleReturns() {
            EmitResult result = emit("""
                module @test {
                  func.func public @multi(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
                    stablehlo.return %a, %b : tensor<4xf32>, tensor<8xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("(tensor<4xf32>, tensor<8xf32>)"));
        }
    }

    @Nested
    class OperationEmissionTests {

        @Test
        void emitDotGeneral() {
            EmitResult result = emit("""
                module @test {
                  func.func public @matmul(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("dot_general"));
            assertTrue(result.babylonText().contains("%arg0"));
            assertTrue(result.babylonText().contains("%arg1"));
            assertTrue(result.babylonText().contains("contract"));
        }

        @Test
        void emitConstant() {
            EmitResult result = emit("""
                module @test {
                  func.func public @const() -> (tensor<4x4xf32>) {
                    %0 = stablehlo.constant dense<0.0> : tensor<4x4xf32>
                    stablehlo.return %0 : tensor<4x4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("constant"));
            assertTrue(result.babylonText().contains("0.0"));
        }

        @Test
        void emitAdd() {
            EmitResult result = emit("""
                module @test {
                  func.func public @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("add %a, %b"));
        }

        @Test
        void emitMultiply() {
            EmitResult result = emit("""
                module @test {
                  func.func public @mul(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.multiply %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("mul %a, %b"));
        }

        @Test
        void emitDivide() {
            EmitResult result = emit("""
                module @test {
                  func.func public @div(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.divide %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("div %a, %b"));
        }

        @Test
        void emitMaximum() {
            EmitResult result = emit("""
                module @test {
                  func.func public @max(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.maximum %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("max %a, %b"));
        }

        @Test
        void emitNegate() {
            EmitResult result = emit("""
                module @test {
                  func.func public @neg(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.negate %a : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("neg %a"));
        }

        @Test
        void emitReshape() {
            EmitResult result = emit("""
                module @test {
                  func.func public @reshape(%a: tensor<4x8xf32>) -> (tensor<32xf32>) {
                    %0 = stablehlo.reshape %a : tensor<4x8xf32> -> tensor<32xf32>
                    stablehlo.return %0 : tensor<32xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("reshape"));
            assertTrue(result.babylonText().contains("32xf32"));
        }

        @Test
        void emitTranspose() {
            EmitResult result = emit("""
                module @test {
                  func.func public @transpose(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a, dims = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("transpose"));
            assertTrue(result.babylonText().contains("dims="));
        }

        @Test
        void emitBroadcast() {
            EmitResult result = emit("""
                module @test {
                  func.func public @broadcast(%a: tensor<4xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("broadcast"));
        }

        @Test
        void emitReturn() {
            EmitResult result = emit("""
                module @test {
                  func.func public @ret(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %a : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("return %a"));
        }

        @Test
        void emitMultipleReturns() {
            EmitResult result = emit("""
                module @test {
                  func.func public @ret(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
                    stablehlo.return %a, %b : tensor<4xf32>, tensor<8xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("return %a, %b"));
        }
    }

    @Nested
    class ChainedOperationsEmissionTests {

        @Test
        void emitOperationChain() {
            EmitResult result = emit("""
                module @test {
                  func.func public @chain(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    %1 = stablehlo.multiply %0, %a : tensor<4xf32>
                    %2 = stablehlo.negate %1 : tensor<4xf32>
                    stablehlo.return %2 : tensor<4xf32>
                  }
                }
                """);

            // Check all operations are present
            assertTrue(result.babylonText().contains("add"));
            assertTrue(result.babylonText().contains("mul"));
            assertTrue(result.babylonText().contains("neg"));
            assertTrue(result.babylonText().contains("return"));

            // Check SSA references
            assertTrue(result.babylonText().contains("%0"));
            assertTrue(result.babylonText().contains("%1"));
            assertTrue(result.babylonText().contains("%2"));
        }

        @Test
        void emitMlpPattern() {
            EmitResult result = emit("""
                module @main {
                  func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    %1 = stablehlo.constant dense<0.0> : tensor<4x16xf32>
                    %2 = stablehlo.maximum %0, %1 : tensor<4x16xf32>
                    stablehlo.return %2 : tensor<4x16xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("dot_general"));
            assertTrue(result.babylonText().contains("constant"));
            assertTrue(result.babylonText().contains("max"));
            assertTrue(result.babylonText().contains("return %2"));

            // Verify we have 4 operations
            assertEquals(4, result.operations().size());
        }
    }

    @Nested
    class TypeAnnotationTests {

        @Test
        void emitFloatTensorType() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("tensor<4xf32>"));
        }

        @Test
        void emitMultiDimensionalType() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4x8x16xf32>) -> (tensor<4x8x16xf32>) {
                    stablehlo.return %arg0 : tensor<4x8x16xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("tensor<4x8x16xf32>"));
        }

        @Test
        void emitIntegerTensorType() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>) {
                    stablehlo.return %arg0 : tensor<4xi32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("tensor<4xi32>"));
        }
    }

    @Nested
    class HeaderAndCommentsTests {

        @Test
        void emitContainsModuleNameComment() {
            EmitResult result = emit("""
                module @mymodule {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("@mymodule"));
            assertTrue(result.babylonText().contains("Babylon Op Tree"));
        }

        @Test
        void emitContainsGeneratedComment() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("Generated from StableHLO"));
        }
    }

    @Nested
    class ErrorCaseTests {

        @Test
        void emitThrowsOnEmptyModule() {
            assertThrows(StableHloParseException.class, () -> {
                Module module = StableHloParser.parse("""
                    module @empty {
                    }
                    """);
                new StableHloToBabylon().emit(module);
            });
        }
    }

    @Nested
    class ShapeAnnotationTests {

        @Test
        void emitShapeInComments() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.negate %arg0 : tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """);

            // Check that shape info appears in output
            assertTrue(result.babylonText().contains("4x8xf32"));
        }

        @Test
        void emitJavaTypeAnnotations() {
            EmitResult result = emit("""
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.constant dense<1.0> : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """);

            // Check that Java type info appears in output
            assertTrue(result.babylonText().contains("Tensor") || result.babylonText().contains("Float"));
        }
    }

    @Nested
    class MultipleFunctionEmissionTests {

        @Test
        void emitFirstFunctionOnly() {
            // Current implementation only emits the first function
            EmitResult result = emit("""
                module @test {
                  func.func public @first(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                  func.func public @second(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
                    stablehlo.return %arg0 : tensor<8xf32>
                  }
                }
                """);

            assertTrue(result.babylonText().contains("@first"));
            // Note: Current impl only emits first function
        }
    }
}
