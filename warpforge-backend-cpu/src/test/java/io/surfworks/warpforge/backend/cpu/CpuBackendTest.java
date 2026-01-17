package io.surfworks.warpforge.backend.cpu;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AbsOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CompareOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComparisonDirection;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogisticOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MaximumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MinimumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NegateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SelectOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanhOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TransposeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.testing.TensorAssert;
import io.surfworks.warpforge.core.testing.ToleranceConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuBackendTest {

    private CpuBackend backend;
    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;

    // Helper to create TensorType
    private static TensorType tensor(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), F32);
    }

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
    }

    @Test
    void backendProperties() {
        assertEquals("cpu", backend.name());
        assertNotNull(backend.capabilities());
        assertTrue(backend.capabilities().supportedDtypes().contains(ScalarType.F32));
    }

    @Nested
    @DisplayName("Binary Operations")
    class BinaryOps {

        @Test
        void add() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            AddOp op = new AddOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));
                assertEquals(1, results.size());

                try (Tensor result = results.getFirst()) {
                    float[] expected = {6, 8, 10, 12};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void subtract() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            SubtractOp op = new SubtractOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{10, 20, 30, 40}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {9, 18, 27, 36};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void multiply() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            MultiplyOp op = new MultiplyOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{2, 2, 2, 2}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {2, 4, 6, 8};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void divide() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            DivideOp op = new DivideOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{10, 20, 30, 40}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{2, 4, 5, 8}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {5, 5, 6, 5};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void maximum() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            MaximumOp op = new MaximumOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 5, 3, 8}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{2, 4, 6, 7}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {2, 5, 6, 8};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void minimum() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            MinimumOp op = new MinimumOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 5, 3, 8}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{2, 4, 6, 7}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, 4, 3, 7};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Unary Operations")
    class UnaryOps {

        @Test
        void negate() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            NegateOp op = new NegateOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, -2, 3, -4}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {-1, 2, -3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void abs() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            AbsOp op = new AbsOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{-1, 2, -3, 4}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void exp() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            ExpOp op = new ExpOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, 2, -1}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1.0f, (float) Math.E, (float) Math.pow(Math.E, 2), (float) Math.exp(-1)};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void tanh() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            TanhOp op = new TanhOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, -1, 2}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {0, (float) Math.tanh(1), (float) Math.tanh(-1), (float) Math.tanh(2)};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void logistic() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            LogisticOp op = new LogisticOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, -1, 100}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0.5f, actual[0], 1e-5f); // sigmoid(0) = 0.5
                    assertTrue(actual[1] > 0.5f); // sigmoid(1) > 0.5
                    assertTrue(actual[2] < 0.5f); // sigmoid(-1) < 0.5
                    assertTrue(actual[3] > 0.99f); // sigmoid(100) ~ 1
                }
            }
        }
    }

    @Nested
    @DisplayName("Shape Operations")
    class ShapeOps {

        @Test
        void reshape() {
            TensorType t6 = tensor(6);
            TensorType t2x3 = tensor(2, 3);
            Value r = new Value("0", t2x3);
            Value arg = new Value("arg0", t6);
            ReshapeOp op = new ReshapeOp(r, arg, t2x3);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 6)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2, 3}, result.shape());
                    assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void transpose2D() {
            TensorType t2x3 = tensor(2, 3);
            TensorType t3x2 = tensor(3, 2);
            Value r = new Value("0", t3x2);
            Value arg = new Value("arg0", t2x3);
            TransposeOp op = new TransposeOp(r, arg, List.of(1L, 0L), t3x2);

            // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{3, 2}, result.shape());
                    float[] expected = {1, 4, 2, 5, 3, 6};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Comparison and Selection")
    class ComparisonAndSelection {

        @Test
        void compareGT() {
            TensorType t4 = tensor(4);
            TensorType t4bool = new TensorType(List.of(4), StableHloAst.ScalarType.I1);
            Value r = new Value("0", t4bool);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            CompareOp op = new CompareOp(r, lhs, rhs, ComparisonDirection.GT, t4bool);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 5, 3, 4}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{2, 4, 3, 5}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {0, 1, 0, 0}; // false, true, false, false
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }

        @Test
        void select() {
            TensorType t4 = tensor(4);
            TensorType t4bool = new TensorType(List.of(4), StableHloAst.ScalarType.I1);
            Value r = new Value("0", t4);
            Value pred = new Value("pred", t4bool);
            Value onTrue = new Value("arg0", t4);
            Value onFalse = new Value("arg1", t4);
            SelectOp op = new SelectOp(r, pred, onTrue, onFalse, t4);

            try (Tensor p = Tensor.fromFloatArray(new float[]{1, 0, 1, 0}, 4);
                 Tensor a = Tensor.fromFloatArray(new float[]{10, 20, 30, 40}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{100, 200, 300, 400}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(p, a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {10, 200, 30, 400}; // true->a, false->b
                    assertArrayEquals(expected, result.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Test
    void closedBackendThrows() {
        backend.close();
        TensorType t4 = tensor(4);
        Value r = new Value("0", t4);
        Value lhs = new Value("arg0", t4);
        Value rhs = new Value("arg1", t4);
        AddOp op = new AddOp(r, lhs, rhs, t4);

        try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
             Tensor b = Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 4)) {

            assertThrows(IllegalStateException.class, () ->
                backend.execute(op, List.of(a, b)));
        }
    }
}
