package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Atan2Op;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CbrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CeilOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ClampOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConcatenateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CosOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Expm1Op;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FloorOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetDimensionSizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IsFiniteOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Log1pOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PowerOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RemainderOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReverseOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SignOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SinOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.warpforge.backend.cpu.CpuBackend;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Comprehensive unit tests for all scalar CPU kernels.
 */
class ScalarKernelTest {

    private CpuBackend backend;
    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;
    private static final StableHloAst.ScalarType I32 = StableHloAst.ScalarType.I32;
    private static final StableHloAst.ScalarType I1 = StableHloAst.ScalarType.I1;

    private static TensorType tensor(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), F32);
    }

    private static TensorType tensorI32(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), I32);
    }

    private static TensorType tensorBool(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), I1);
    }

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
    }

    @Nested
    @DisplayName("Unary Math Operations")
    class UnaryMathOps {

        @Test
        void log() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            LogOp op = new LogOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, (float) Math.E, 10, 100}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0, actual[0], 1e-5f);
                    assertEquals(1, actual[1], 1e-5f);
                    assertEquals((float) Math.log(10), actual[2], 1e-5f);
                    assertEquals((float) Math.log(100), actual[3], 1e-5f);
                }
            }
        }

        @Test
        void sqrt() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            SqrtOp op = new SqrtOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, 4, 9}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {0, 1, 2, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void rsqrt() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            RsqrtOp op = new RsqrtOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 4, 9, 16}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1.0f, 0.5f, 1.0f/3.0f, 0.25f};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void sin() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            SinOp op = new SinOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, (float) Math.PI/2, (float) Math.PI, (float) (3*Math.PI/2)}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0, actual[0], 1e-5f);
                    assertEquals(1, actual[1], 1e-5f);
                    assertEquals(0, actual[2], 1e-5f);
                    assertEquals(-1, actual[3], 1e-5f);
                }
            }
        }

        @Test
        void cos() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            CosOp op = new CosOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, (float) Math.PI/2, (float) Math.PI, (float) (2*Math.PI)}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(1, actual[0], 1e-5f);
                    assertEquals(0, actual[1], 1e-5f);
                    assertEquals(-1, actual[2], 1e-5f);
                    assertEquals(1, actual[3], 1e-5f);
                }
            }
        }

        @Test
        void tan() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            TanOp op = new TanOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, (float) Math.PI/4, (float) -Math.PI/4, 1}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0, actual[0], 1e-5f);
                    assertEquals(1, actual[1], 1e-5f);
                    assertEquals(-1, actual[2], 1e-5f);
                    assertEquals((float) Math.tan(1), actual[3], 1e-5f);
                }
            }
        }

        @Test
        void ceil() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            CeilOp op = new CeilOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1.2f, 2.7f, -1.5f, 3.0f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {2, 3, -1, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void floor() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            FloorOp op = new FloorOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1.2f, 2.7f, -1.5f, 3.0f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, 2, -2, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void sign() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            SignOp op = new SignOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{5, -3, 0, -0.5f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, -1, 0, -1};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void expm1() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            Expm1Op op = new Expm1Op(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, -1, 0.001f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0, actual[0], 1e-5f);
                    assertEquals((float) Math.expm1(1), actual[1], 1e-5f);
                    assertEquals((float) Math.expm1(-1), actual[2], 1e-5f);
                    // expm1 is more accurate for small values
                    assertEquals((float) Math.expm1(0.001), actual[3], 1e-7f);
                }
            }
        }

        @Test
        void log1p() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            Log1pOp op = new Log1pOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, (float)(Math.E - 1), 0.001f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0, actual[0], 1e-5f);
                    assertEquals((float) Math.log(2), actual[1], 1e-5f);
                    assertEquals(1, actual[2], 1e-5f);
                    // log1p is more accurate for small values
                    assertEquals((float) Math.log1p(0.001), actual[3], 1e-7f);
                }
            }
        }

        @Test
        void cbrt() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            CbrtOp op = new CbrtOp(r, arg, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, 8, 27}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {0, 1, 2, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void isFinite() {
            TensorType t4 = tensor(4);
            TensorType t4bool = tensorBool(4);
            Value r = new Value("0", t4bool);
            Value arg = new Value("arg0", t4);
            IsFiniteOp op = new IsFiniteOp(r, arg, t4bool);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1.0f, Float.POSITIVE_INFINITY, Float.NaN, 0.0f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(1, actual[0], 1e-5f); // 1.0 is finite
                    assertEquals(0, actual[1], 1e-5f); // infinity is not finite
                    assertEquals(0, actual[2], 1e-5f); // NaN is not finite
                    assertEquals(1, actual[3], 1e-5f); // 0.0 is finite
                }
            }
        }
    }

    @Nested
    @DisplayName("Binary Math Operations")
    class BinaryMathOps {

        @Test
        void power() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            PowerOp op = new PowerOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{2, 3, 4, 10}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{3, 2, 0.5f, 2}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {8, 9, 2, 100};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void remainder() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            RemainderOp op = new RemainderOp(r, lhs, rhs, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{7, 10, 15, 20}, 4);
                 Tensor b = Tensor.fromFloatArray(new float[]{3, 4, 4, 6}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, 2, 3, 2};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void atan2() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value lhs = new Value("arg0", t4);
            Value rhs = new Value("arg1", t4);
            Atan2Op op = new Atan2Op(r, lhs, rhs, t4);

            try (Tensor y = Tensor.fromFloatArray(new float[]{0, 1, 0, 1}, 4);
                 Tensor x = Tensor.fromFloatArray(new float[]{1, 0, -1, 1}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(y, x));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    assertEquals(0, actual[0], 1e-5f); // atan2(0, 1) = 0
                    assertEquals((float) Math.PI/2, actual[1], 1e-5f); // atan2(1, 0) = pi/2
                    assertEquals((float) Math.PI, actual[2], 1e-5f); // atan2(0, -1) = pi
                    assertEquals((float) Math.PI/4, actual[3], 1e-5f); // atan2(1, 1) = pi/4
                }
            }
        }
    }

    @Nested
    @DisplayName("Selection Operations")
    class SelectionOps {

        @Test
        void clamp() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value minV = new Value("min", t4);
            Value operand = new Value("operand", t4);
            Value maxV = new Value("max", t4);
            ClampOp op = new ClampOp(r, minV, operand, maxV, t4);

            try (Tensor min = Tensor.fromFloatArray(new float[]{0, 0, 0, 0}, 4);
                 Tensor input = Tensor.fromFloatArray(new float[]{-1, 0.5f, 1, 2}, 4);
                 Tensor max = Tensor.fromFloatArray(new float[]{1, 1, 1, 1}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(min, input, max));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {0, 0.5f, 1, 1}; // clamped to [0, 1]
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Data Movement Operations")
    class DataMovementOps {

        @Test
        void concatenate() {
            TensorType t2 = tensor(2);
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg0 = new Value("arg0", t2);
            Value arg1 = new Value("arg1", t2);
            ConcatenateOp op = new ConcatenateOp(r, List.of(arg0, arg1), 0L, t4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2}, 2);
                 Tensor b = Tensor.fromFloatArray(new float[]{3, 4}, 2)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{4}, result.shape());
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void concatenate2D() {
            TensorType t2x2 = tensor(2, 2);
            TensorType t2x4 = tensor(2, 4);
            Value r = new Value("0", t2x4);
            Value arg0 = new Value("arg0", t2x2);
            Value arg1 = new Value("arg1", t2x2);
            ConcatenateOp op = new ConcatenateOp(r, List.of(arg0, arg1), 1L, t2x4);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2);
                 Tensor b = Tensor.fromFloatArray(new float[]{5, 6, 7, 8}, 2, 2)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2, 4}, result.shape());
                    float[] expected = {1, 2, 5, 6, 3, 4, 7, 8};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void slice() {
            TensorType t6 = tensor(6);
            TensorType t3 = tensor(3);
            Value r = new Value("0", t3);
            Value arg = new Value("arg0", t6);
            SliceOp op = new SliceOp(r, arg, List.of(1L), List.of(4L), List.of(1L), t3);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, 2, 3, 4, 5}, 6)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{3}, result.shape());
                    float[] expected = {1, 2, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void sliceWithStride() {
            TensorType t6 = tensor(6);
            TensorType t3 = tensor(3);
            Value r = new Value("0", t3);
            Value arg = new Value("arg0", t6);
            SliceOp op = new SliceOp(r, arg, List.of(0L), List.of(6L), List.of(2L), t3);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 1, 2, 3, 4, 5}, 6)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{3}, result.shape());
                    float[] expected = {0, 2, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void reverse() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);
            ReverseOp op = new ReverseOp(r, arg, List.of(0L), t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {4, 3, 2, 1};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void reverse2D() {
            TensorType t2x3 = tensor(2, 3);
            Value r = new Value("0", t2x3);
            Value arg = new Value("arg0", t2x3);
            ReverseOp op = new ReverseOp(r, arg, List.of(1L), t2x3);

            // Reverse along axis 1: [[1,2,3],[4,5,6]] -> [[3,2,1],[6,5,4]]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {3, 2, 1, 6, 5, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void iota() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            IotaOp op = new IotaOp(r, 0L, t4);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                assertArrayEquals(new int[]{4}, result.shape());
                float[] expected = {0, 1, 2, 3};
                assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
            }
        }

        @Test
        void iota2D() {
            TensorType t2x3 = tensor(2, 3);
            Value r = new Value("0", t2x3);
            IotaOp op = new IotaOp(r, 0L, t2x3);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                assertArrayEquals(new int[]{2, 3}, result.shape());
                // Along axis 0: row indices broadcast
                float[] expected = {0, 0, 0, 1, 1, 1};
                assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
            }
        }

        @Test
        void getDimensionSize() {
            TensorType t2x3x4 = tensor(2, 3, 4);
            TensorType tScalar = new TensorType(List.of(), I32);
            Value r = new Value("0", tScalar);
            Value arg = new Value("arg0", t2x3x4);
            GetDimensionSizeOp op = new GetDimensionSizeOp(r, arg, 1L, tScalar);

            try (Tensor input = Tensor.zeros(io.surfworks.warpforge.core.tensor.ScalarType.F32, 2, 3, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertEquals(3, result.toIntArray()[0]);
                }
            }
        }
    }

    @Nested
    @DisplayName("Linear Algebra Operations")
    class LinearAlgebraOps {

        @Test
        void dot() {
            TensorType t3 = tensor(3);
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            Value lhs = new Value("arg0", t3);
            Value rhs = new Value("arg1", t3);
            DotOp op = new DotOp(r, lhs, rhs, tScalar);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3}, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{4, 5, 6}, 3)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
                    float[] actual = result.toFloatArray();
                    assertEquals(32, actual[0], 1e-5f);
                }
            }
        }

        @Test
        void dotMatVec() {
            TensorType t2x3 = tensor(2, 3);
            TensorType t3 = tensor(3);
            TensorType t2 = tensor(2);
            Value r = new Value("0", t2);
            Value lhs = new Value("arg0", t2x3);
            Value rhs = new Value("arg1", t3);
            DotOp op = new DotOp(r, lhs, rhs, t2);

            // [[1,2,3],[4,5,6]] @ [1,1,1] = [6, 15]
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 1, 1}, 3)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2}, result.shape());
                    float[] expected = {6, 15};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void dotMatMat() {
            TensorType t2x3 = tensor(2, 3);
            TensorType t3x2 = tensor(3, 2);
            TensorType t2x2 = tensor(2, 2);
            Value r = new Value("0", t2x2);
            Value lhs = new Value("arg0", t2x3);
            Value rhs = new Value("arg1", t3x2);
            DotOp op = new DotOp(r, lhs, rhs, t2x2);

            // [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
            // = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
            // = [[22, 28], [49, 64]]
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 3, 2)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2, 2}, result.shape());
                    float[] expected = {22, 28, 49, 64};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Broadcast Operations")
    class BroadcastOps {

        @Test
        void broadcastInDim() {
            TensorType t3 = tensor(3);
            TensorType t2x3 = tensor(2, 3);
            Value r = new Value("0", t2x3);
            Value arg = new Value("arg0", t3);
            BroadcastInDimOp op = new BroadcastInDimOp(r, arg, List.of(1L), t2x3);

            // Broadcast [1,2,3] to shape [2,3]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3}, 3)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2, 3}, result.shape());
                    float[] expected = {1, 2, 3, 1, 2, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Vectorization Verification")
    class VectorizationTests {

        @Test
        void addLargeArray() {
            // Test with a large array to exercise vectorized path
            int size = 1024;
            TensorType t = tensor(size);
            Value r = new Value("0", t);
            Value lhs = new Value("arg0", t);
            Value rhs = new Value("arg1", t);
            AddOp op = new AddOp(r, lhs, rhs, t);

            float[] aData = new float[size];
            float[] bData = new float[size];
            float[] expected = new float[size];
            for (int i = 0; i < size; i++) {
                aData[i] = i;
                bData[i] = size - i;
                expected[i] = size;
            }

            try (Tensor a = Tensor.fromFloatArray(aData, size);
                 Tensor b = Tensor.fromFloatArray(bData, size)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void multiplyLargeArray() {
            int size = 1024;
            TensorType t = tensor(size);
            Value r = new Value("0", t);
            Value lhs = new Value("arg0", t);
            Value rhs = new Value("arg1", t);
            MultiplyOp op = new MultiplyOp(r, lhs, rhs, t);

            float[] aData = new float[size];
            float[] bData = new float[size];
            float[] expected = new float[size];
            for (int i = 0; i < size; i++) {
                aData[i] = i;
                bData[i] = 2;
                expected[i] = i * 2;
            }

            try (Tensor a = Tensor.fromFloatArray(aData, size);
                 Tensor b = Tensor.fromFloatArray(bData, size)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void expLargeArray() {
            int size = 256;
            TensorType t = tensor(size);
            Value r = new Value("0", t);
            Value arg = new Value("arg0", t);
            ExpOp op = new ExpOp(r, arg, t);

            float[] inputData = new float[size];
            float[] expected = new float[size];
            for (int i = 0; i < size; i++) {
                inputData[i] = (i - size/2) * 0.01f;
                expected[i] = (float) Math.exp(inputData[i]);
            }

            try (Tensor input = Tensor.fromFloatArray(inputData, size)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(expected, result.toFloatArray(), 1e-4f);
                }
            }
        }

        @Test
        void sqrtLargeArray() {
            int size = 1024;
            TensorType t = tensor(size);
            Value r = new Value("0", t);
            Value arg = new Value("arg0", t);
            SqrtOp op = new SqrtOp(r, arg, t);

            float[] inputData = new float[size];
            float[] expected = new float[size];
            for (int i = 0; i < size; i++) {
                inputData[i] = i + 1; // avoid sqrt(0) precision issues
                expected[i] = (float) Math.sqrt(inputData[i]);
            }

            try (Tensor input = Tensor.fromFloatArray(inputData, size)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }
}
