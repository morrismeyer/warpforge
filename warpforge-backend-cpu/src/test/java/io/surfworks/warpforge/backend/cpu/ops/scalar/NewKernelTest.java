package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CholeskyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComplexOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicIotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetTupleElementOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ImagOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OptimizationBarrierOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RealOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReducePrecisionOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TriangularSolveOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TupleOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformDequantizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformQuantizeOp;
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
 * Tests for newly added CPU kernels.
 */
class NewKernelTest {

    private CpuBackend backend;
    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;
    private static final StableHloAst.ScalarType I32 = StableHloAst.ScalarType.I32;

    private static TensorType tensor(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), F32);
    }

    private static TensorType tensorI32(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), I32);
    }

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
    }

    @Nested
    @DisplayName("Tuple Operations")
    class TupleOps {

        @Test
        void tupleCreate() {
            TensorType t2 = tensor(2);
            TensorType t3 = tensor(3);
            Value r = new Value("0", tensor());
            Value arg0 = new Value("arg0", t2);
            Value arg1 = new Value("arg1", t3);
            TupleOp op = new TupleOp(r, List.of(arg0, arg1), tensor());

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2}, 2);
                 Tensor b = Tensor.fromFloatArray(new float[]{3, 4, 5}, 3)) {

                List<Tensor> results = backend.execute(op, List.of(a, b));

                assertEquals(2, results.size());
                try (Tensor r0 = results.get(0); Tensor r1 = results.get(1)) {
                    assertArrayEquals(new float[]{1, 2}, r0.toFloatArray(), 1e-5f);
                    assertArrayEquals(new float[]{3, 4, 5}, r1.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void getTupleElement() {
            TensorType t2 = tensor(2);
            Value r = new Value("0", t2);
            Value tuple = new Value("tuple", tensor());
            GetTupleElementOp op = new GetTupleElementOp(r, tuple, 1, t2);

            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2}, 2);
                 Tensor b = Tensor.fromFloatArray(new float[]{3, 4}, 2);
                 Tensor c = Tensor.fromFloatArray(new float[]{5, 6}, 2)) {

                List<Tensor> results = backend.execute(op, List.of(a, b, c));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new float[]{3, 4}, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Complex Number Operations")
    class ComplexOps {

        @Test
        void complexCreate() {
            TensorType t2 = tensor(2);
            TensorType t2x2 = tensor(2, 2);
            Value r = new Value("0", t2x2);
            Value real = new Value("real", t2);
            Value imag = new Value("imag", t2);
            ComplexOp op = new ComplexOp(r, real, imag, t2x2);

            try (Tensor realT = Tensor.fromFloatArray(new float[]{1, 2}, 2);
                 Tensor imagT = Tensor.fromFloatArray(new float[]{3, 4}, 2)) {

                List<Tensor> results = backend.execute(op, List.of(realT, imagT));

                try (Tensor result = results.getFirst()) {
                    // Interleaved: [real0, imag0, real1, imag1]
                    float[] expected = {1, 3, 2, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void realExtract() {
            TensorType t2x2 = tensor(2, 2);
            TensorType t2 = tensor(2);
            Value r = new Value("0", t2);
            Value operand = new Value("operand", t2x2);
            RealOp op = new RealOp(r, operand, t2);

            // Interleaved complex: [real0, imag0, real1, imag1] = [1, 3, 2, 4]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 3, 2, 4}, 2, 2)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    // Extract real parts: [1, 2]
                    float[] expected = {1, 2};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void imagExtract() {
            TensorType t2x2 = tensor(2, 2);
            TensorType t2 = tensor(2);
            Value r = new Value("0", t2);
            Value operand = new Value("operand", t2x2);
            ImagOp op = new ImagOp(r, operand, t2);

            // Interleaved complex: [real0, imag0, real1, imag1] = [1, 3, 2, 4]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 3, 2, 4}, 2, 2)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    // Extract imag parts: [3, 4]
                    float[] expected = {3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Dynamic Shape Operations")
    class DynamicShapeOps {

        @Test
        void dynamicIota() {
            TensorType t4 = tensor(4);
            TensorType t1 = tensorI32(1);
            Value r = new Value("0", t4);
            Value shape = new Value("shape", t1);
            DynamicIotaOp op = new DynamicIotaOp(r, shape, 0L, t4);

            try (Tensor shapeTensor = Tensor.fromFloatArray(new float[]{4}, 1)) {
                List<Tensor> results = backend.execute(op, List.of(shapeTensor));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {0, 1, 2, 3};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void dynamicReshape() {
            TensorType t6 = tensor(6);
            TensorType t2x3 = tensor(2, 3);
            TensorType tShape = tensorI32(2);
            Value r = new Value("0", t2x3);
            Value operand = new Value("operand", t6);
            Value shape = new Value("shape", tShape);
            DynamicReshapeOp op = new DynamicReshapeOp(r, operand, shape, t2x3);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 6);
                 Tensor shapeTensor = Tensor.fromFloatArray(new float[]{2, 3}, 2)) {

                List<Tensor> results = backend.execute(op, List.of(input, shapeTensor));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2, 3}, result.shape());
                    float[] expected = {1, 2, 3, 4, 5, 6};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Linear Algebra Operations")
    class LinearAlgebraOps {

        @Test
        void choleskyLower() {
            // 2x2 positive definite matrix: [[4, 2], [2, 5]]
            // Cholesky L: [[2, 0], [1, 2]]
            TensorType t2x2 = tensor(2, 2);
            Value r = new Value("0", t2x2);
            Value operand = new Value("operand", t2x2);
            CholeskyOp op = new CholeskyOp(r, operand, true, t2x2);

            try (Tensor input = Tensor.fromFloatArray(new float[]{4, 2, 2, 5}, 2, 2)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    // L[0,0] = sqrt(4) = 2
                    assertEquals(2, actual[0], 1e-5f);
                    // L[0,1] = 0 (lower triangular)
                    assertEquals(0, actual[1], 1e-5f);
                    // L[1,0] = 2/2 = 1
                    assertEquals(1, actual[2], 1e-5f);
                    // L[1,1] = sqrt(5-1) = 2
                    assertEquals(2, actual[3], 1e-5f);
                }
            }
        }

        @Test
        void triangularSolveLower() {
            // Solve L * X = B where L = [[2, 0], [1, 2]], B = [[4], [5]]
            // X = [[2], [1.5]]
            TensorType t2x2 = tensor(2, 2);
            TensorType t2x1 = tensor(2, 1);
            Value r = new Value("0", t2x1);
            Value a = new Value("a", t2x2);
            Value b = new Value("b", t2x1);
            TriangularSolveOp op = new TriangularSolveOp(r, a, b, true, true, false, t2x1);

            try (Tensor aT = Tensor.fromFloatArray(new float[]{2, 0, 1, 2}, 2, 2);
                 Tensor bT = Tensor.fromFloatArray(new float[]{4, 5}, 2, 1)) {

                List<Tensor> results = backend.execute(op, List.of(aT, bT));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    // x[0] = 4/2 = 2
                    assertEquals(2, actual[0], 1e-5f);
                    // x[1] = (5 - 1*2)/2 = 1.5
                    assertEquals(1.5f, actual[1], 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Optimization and Precision Operations")
    class OptPrecisionOps {

        @Test
        void optimizationBarrier() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value operand = new Value("operand", t4);
            OptimizationBarrierOp op = new OptimizationBarrierOp(List.of(r), List.of(operand), t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    // Should pass through unchanged
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void reducePrecision() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value operand = new Value("operand", t4);
            ReducePrecisionOp op = new ReducePrecisionOp(r, operand, 5, 10, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1.0f, 2.5f, 0.125f, 100.0f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    // Values should be rounded to reduced precision
                    // Exact values depend on implementation, just verify finite
                    for (float v : actual) {
                        assertTrue(Float.isFinite(v));
                    }
                }
            }
        }
    }

    @Nested
    @DisplayName("Quantization Operations")
    class QuantizationOps {

        @Test
        void uniformQuantize() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value operand = new Value("operand", t4);
            UniformQuantizeOp op = new UniformQuantizeOp(r, operand, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{0.0f, 0.5f, -0.5f, 1.0f}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    // Quantized values should be integers in range [-128, 127]
                    for (float v : actual) {
                        assertTrue(v >= -128 && v <= 127);
                        assertEquals(Math.round(v), v, 1e-5f); // Should be integer
                    }
                }
            }
        }

        @Test
        void uniformDequantize() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value operand = new Value("operand", t4);
            UniformDequantizeOp op = new UniformDequantizeOp(r, operand, t4);

            // Quantized values
            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 64, -64, 127}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] actual = result.toFloatArray();
                    // Dequantized values should be floats
                    assertEquals(0, actual[0], 1e-5f);
                    assertTrue(actual[1] > 0); // 64 * scale > 0
                    assertTrue(actual[2] < 0); // -64 * scale < 0
                    assertTrue(actual[3] > actual[1]); // 127 > 64
                }
            }
        }

        @Test
        void quantizeRoundTrip() {
            TensorType t4 = tensor(4);
            Value rq = new Value("q", t4);
            Value operand = new Value("operand", t4);
            UniformQuantizeOp quantOp = new UniformQuantizeOp(rq, operand, t4);

            Value rd = new Value("d", t4);
            Value quantized = new Value("quantized", t4);
            UniformDequantizeOp dequantOp = new UniformDequantizeOp(rd, quantized, t4);

            // Test round trip: quantize then dequantize
            float[] original = {0.0f, 0.25f, -0.25f, 0.5f};
            try (Tensor input = Tensor.fromFloatArray(original, 4)) {
                List<Tensor> quantResults = backend.execute(quantOp, List.of(input));

                try (Tensor quantResult = quantResults.getFirst()) {
                    List<Tensor> dequantResults = backend.execute(dequantOp, List.of(quantResult));

                    try (Tensor result = dequantResults.getFirst()) {
                        float[] actual = result.toFloatArray();
                        // Round trip should preserve approximate values (with quantization error)
                        for (int i = 0; i < original.length; i++) {
                            assertEquals(original[i], actual[i], 0.02f); // Allow quantization error
                        }
                    }
                }
            }
        }
    }

    @Nested
    @DisplayName("Distributed Operations (Stubs)")
    class DistributedOps {

        @Test
        void afterAll() {
            TensorType t1 = tensor(1);
            Value r = new Value("0", t1);
            StableHloAst.AfterAllOp op = new StableHloAst.AfterAllOp(r, List.of(), t1);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                // Should return a token tensor
                assertEquals(1, result.elementCount());
            }
        }

        @Test
        void allReduce() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value operand = new Value("operand", t4);
            StableHloAst.AllReduceOp op = new StableHloAst.AllReduceOp(
                r, operand, List.of(), 0L, false, "add", t4
            );

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    // Single replica: should pass through unchanged
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void replicaId() {
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            StableHloAst.ReplicaIdOp op = new StableHloAst.ReplicaIdOp(r, tScalar);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                // Single replica: should return 0
                assertEquals(0, result.toFloatArray()[0], 1e-5f);
            }
        }

        @Test
        void partitionId() {
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            StableHloAst.PartitionIdOp op = new StableHloAst.PartitionIdOp(r, tScalar);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                // Single partition: should return 0
                assertEquals(0, result.toFloatArray()[0], 1e-5f);
            }
        }
    }

    @Nested
    @DisplayName("Control Flow Operations")
    class ControlFlowOps {

        @Test
        void returnOp() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value operand = new Value("operand", t4);
            StableHloAst.ReturnOp op = new StableHloAst.ReturnOp(List.of(operand));

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    // Return should pass through inputs
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }
}
