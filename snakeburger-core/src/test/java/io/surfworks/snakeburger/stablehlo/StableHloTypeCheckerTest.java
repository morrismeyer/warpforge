package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StableHloTypeCheckerTest {

    private List<String> validate(String mlir) {
        Module module = StableHloParser.parse(mlir);
        return new StableHloTypeChecker().validate(module);
    }

    private void assertValid(String mlir) {
        List<String> errors = validate(mlir);
        assertTrue(errors.isEmpty(), "Expected no errors but got: " + errors);
    }

    private void assertInvalid(String mlir, String... expectedErrorPatterns) {
        List<String> errors = validate(mlir);
        assertFalse(errors.isEmpty(), "Expected errors but got none");
        for (String pattern : expectedErrorPatterns) {
            assertTrue(errors.stream().anyMatch(e -> e.contains(pattern)),
                    "Expected error containing '" + pattern + "' but got: " + errors);
        }
    }

    @Nested
    class ValidProgramsTests {

        @Test
        void simplePassthrough() {
            assertValid("""
                module @test {
                  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %arg0 : tensor<4xf32>
                  }
                }
                """);
        }

        @Test
        void matrixMultiplication() {
            assertValid("""
                module @test {
                  func.func public @matmul(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """);
        }

        @Test
        void batchedMatrixMultiplication() {
            assertValid("""
                module @test {
                  func.func public @batched(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> (tensor<2x4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]> : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
                    stablehlo.return %0 : tensor<2x4x16xf32>
                  }
                }
                """);
        }

        @Test
        void mlpPattern() {
            assertValid("""
                module @main {
                  func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    %1 = stablehlo.constant dense<0.0> : tensor<4x16xf32>
                    %2 = stablehlo.maximum %0, %1 : tensor<4x16xf32>
                    stablehlo.return %2 : tensor<4x16xf32>
                  }
                }
                """);
        }

        @Test
        void elementwiseOperations() {
            assertValid("""
                module @test {
                  func.func public @elementwise(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> (tensor<4x4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4x4xf32>
                    %1 = stablehlo.multiply %0, %b : tensor<4x4xf32>
                    %2 = stablehlo.divide %1, %a : tensor<4x4xf32>
                    %3 = stablehlo.negate %2 : tensor<4x4xf32>
                    stablehlo.return %3 : tensor<4x4xf32>
                  }
                }
                """);
        }

        @Test
        void reshapePreservesElements() {
            assertValid("""
                module @test {
                  func.func public @reshape(%a: tensor<4x8xf32>) -> (tensor<32xf32>) {
                    %0 = stablehlo.reshape %a : tensor<4x8xf32> -> tensor<32xf32>
                    stablehlo.return %0 : tensor<32xf32>
                  }
                }
                """);
        }

        @Test
        void transposeSwapsDimensions() {
            assertValid("""
                module @test {
                  func.func public @transpose(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a, dims = [1, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """);
        }

        @Test
        void multipleReturns() {
            assertValid("""
                module @test {
                  func.func public @multi(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
                    stablehlo.return %a, %b : tensor<4xf32>, tensor<8xf32>
                  }
                }
                """);
        }
    }

    @Nested
    class ValueAvailabilityTests {

        @Test
        void undefinedValueInOperand() {
            // Note: This is caught by the parser, not the type checker
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse("""
                module @test {
                  func.func public @bad(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %undefined : tensor<4xf32>
                  }
                }
                """));
        }

        @Test
        void valueUsedBeforeDefinition() {
            // Parser catches this
            assertThrows(StableHloParseException.class, () -> StableHloParser.parse("""
                module @test {
                  func.func public @bad(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
                    %1 = stablehlo.add %0, %arg0 : tensor<4xf32>
                    %0 = stablehlo.negate %arg0 : tensor<4xf32>
                    stablehlo.return %1 : tensor<4xf32>
                  }
                }
                """));
        }
    }

    @Nested
    class BinaryOperationShapeTests {

        @Test
        void addWithMismatchedShapes() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """, "shapes must match");
        }

        @Test
        void multiplyWithMismatchedShapes() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x4xf32>, %b: tensor<4x8xf32>) -> (tensor<4x4xf32>) {
                    %0 = stablehlo.multiply %a, %b : tensor<4x4xf32>
                    stablehlo.return %0 : tensor<4x4xf32>
                  }
                }
                """, "shapes must match");
        }

        @Test
        void maximumWithMismatchedShapes() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.maximum %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """, "shapes must match");
        }

        @Test
        void addWithMismatchedResultShape() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<8xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<8xf32>
                    stablehlo.return %0 : tensor<8xf32>
                  }
                }
                """, "result shape");
        }
    }

    @Nested
    class ElementTypeTests {

        @Test
        void addWithMismatchedElementTypes() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<4xf64>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """, "element types must match");
        }

        @Test
        void dotGeneralWithMismatchedElementTypes() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>, %b: tensor<8x16xf64>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf64>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """, "element types must match");
        }
    }

    @Nested
    class DotGeneralValidationTests {

        @Test
        void contractingDimensionMismatch() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>, %b: tensor<16x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<16x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """, "contracting dimensions must have same size");
        }

        @Test
        void batchingDimensionMismatch() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<2x4x8xf32>, %b: tensor<4x8x16xf32>) -> (tensor<2x4x16xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]> : (tensor<2x4x8xf32>, tensor<4x8x16xf32>) -> tensor<2x4x16xf32>
                    stablehlo.return %0 : tensor<2x4x16xf32>
                  }
                }
                """, "batching dimensions must have same size");
        }

        @Test
        void contractingDimensionOutOfRange() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [5], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """, "out of range");
        }

        @Test
        void incorrectResultShape() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """, "result shape");
        }
    }

    @Nested
    class ReshapeValidationTests {

        @Test
        void reshapeChangesElementCount() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<64xf32>) {
                    %0 = stablehlo.reshape %a : tensor<4x8xf32> -> tensor<64xf32>
                    stablehlo.return %0 : tensor<64xf32>
                  }
                }
                """, "element count mismatch");
        }

        @Test
        void reshapeChangesElementType() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<32xf64>) {
                    %0 = stablehlo.reshape %a : tensor<4x8xf32> -> tensor<32xf64>
                    stablehlo.return %0 : tensor<32xf64>
                  }
                }
                """, "element types must match");
        }
    }

    @Nested
    class TransposeValidationTests {

        @Test
        void transposeInvalidPermutation() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a, dims = [0, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """, "duplicate");
        }

        @Test
        void transposePermutationOutOfRange() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a, dims = [2, 0] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """, "out of range");
        }

        @Test
        void transposePermutationWrongLength() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<8x4xf32>) {
                    %0 = stablehlo.transpose %a, dims = [0] : tensor<4x8xf32> -> tensor<8x4xf32>
                    stablehlo.return %0 : tensor<8x4xf32>
                  }
                }
                """, "permutation length");
        }

        @Test
        void transposeWrongResultShape() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.transpose %a, dims = [1, 0] : tensor<4x8xf32> -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """, "result shape");
        }
    }

    @Nested
    class BroadcastValidationTests {

        @Test
        void broadcastIncompatibleDimension() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>) -> (tensor<8x8xf32>) {
                    %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4xf32>) -> tensor<8x8xf32>
                    stablehlo.return %0 : tensor<8x8xf32>
                  }
                }
                """, "incompatible");
        }

        @Test
        void broadcastDimensionOutOfRange() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>) -> (tensor<4x8xf32>) {
                    %0 = stablehlo.broadcast_in_dim %a, dims = [5] : (tensor<4xf32>) -> tensor<4x8xf32>
                    stablehlo.return %0 : tensor<4x8xf32>
                  }
                }
                """, "out of range");
        }

        @Test
        void broadcastWrongDimensionCount() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4x8xf32>) -> (tensor<4x8x16xf32>) {
                    %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<4x8xf32>) -> tensor<4x8x16xf32>
                    stablehlo.return %0 : tensor<4x8x16xf32>
                  }
                }
                """, "dimensions count");
        }
    }

    @Nested
    class ReturnTypeValidationTests {

        @Test
        void returnTypeMismatch() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>) -> (tensor<8xf32>) {
                    stablehlo.return %a : tensor<4xf32>
                  }
                }
                """, "return value");
        }

        @Test
        void returnCountMismatch() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    stablehlo.return %a, %b : tensor<4xf32>, tensor<4xf32>
                  }
                }
                """, "returns");
        }

        @Test
        void missingReturn() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.negate %a : tensor<4xf32>
                  }
                }
                """, "return operation");
        }
    }

    @Nested
    class UnaryOperationTests {

        @Test
        void negateResultMismatch() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>) -> (tensor<8xf32>) {
                    %0 = stablehlo.negate %a : tensor<8xf32>
                    stablehlo.return %0 : tensor<8xf32>
                  }
                }
                """, "result shape");
        }

        @Test
        void negateElementTypeMismatch() {
            assertInvalid("""
                module @test {
                  func.func public @bad(%a: tensor<4xf32>) -> (tensor<4xf64>) {
                    %0 = stablehlo.negate %a : tensor<4xf64>
                    stablehlo.return %0 : tensor<4xf64>
                  }
                }
                """, "element type");
        }
    }

    @Nested
    class CheckMethodTests {

        @Test
        void checkThrowsOnError() {
            String mlir = """
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<8xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            StableHloTypeChecker checker = new StableHloTypeChecker();

            StableHloParseException ex = assertThrows(StableHloParseException.class,
                    () -> checker.check(module));

            assertTrue(ex.getMessage().contains("validation failed"));
        }

        @Test
        void checkPassesOnValid() {
            String mlir = """
                module @test {
                  func.func public @good(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<4xf32>
                  }
                }
                """;

            Module module = StableHloParser.parse(mlir);
            StableHloTypeChecker checker = new StableHloTypeChecker();

            assertDoesNotThrow(() -> checker.check(module));
        }
    }

    @Nested
    class MultipleErrorsTests {

        @Test
        void collectsMultipleErrors() {
            String mlir = """
                module @test {
                  func.func public @bad(%a: tensor<4xf32>, %b: tensor<8xf64>) -> (tensor<16xf32>) {
                    %0 = stablehlo.add %a, %b : tensor<4xf32>
                    stablehlo.return %0 : tensor<16xf32>
                  }
                }
                """;

            List<String> errors = validate(mlir);
            assertTrue(errors.size() >= 2, "Expected multiple errors but got: " + errors.size());
        }
    }
}
