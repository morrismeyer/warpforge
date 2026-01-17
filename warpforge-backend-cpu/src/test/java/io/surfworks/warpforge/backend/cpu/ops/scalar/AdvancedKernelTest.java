package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormInferenceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConstantOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DenseAttr;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SortOp;
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
 * Tests for advanced CPU kernels: reduction, batch normalization, etc.
 */
class AdvancedKernelTest {

    private CpuBackend backend;
    private static final StableHloAst.ScalarType F32 = StableHloAst.ScalarType.F32;

    private static TensorType tensor(int... dims) {
        return new TensorType(Arrays.stream(dims).boxed().toList(), F32);
    }

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
    }

    @Nested
    @DisplayName("Reduction Operations")
    class ReductionOps {

        @Test
        void reduceSum1D() {
            TensorType t4 = tensor(4);
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            Value arg = new Value("arg0", t4);
            Value init = new Value("init", tScalar);

            ReduceOp op = new ReduceOp(r, arg, init, List.of(0L), "add", tScalar);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor initVal = Tensor.fromFloatArray(new float[]{0}, 1)) {

                List<Tensor> results = backend.execute(op, List.of(input, initVal));

                try (Tensor result = results.getFirst()) {
                    // Sum: 1+2+3+4 = 10
                    assertEquals(10, result.toFloatArray()[0], 1e-5f);
                }
            }
        }

        @Test
        void reduceSum2D_axis0() {
            TensorType t2x3 = tensor(2, 3);
            TensorType t3 = tensor(3);
            Value r = new Value("0", t3);
            Value arg = new Value("arg0", t2x3);
            Value init = new Value("init", tensor());

            ReduceOp op = new ReduceOp(r, arg, init, List.of(0L), "add", t3);

            // [[1,2,3],[4,5,6]] reduced along axis 0 = [5,7,9]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor initVal = Tensor.fromFloatArray(new float[]{0})) {

                List<Tensor> results = backend.execute(op, List.of(input, initVal));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{3}, result.shape());
                    float[] expected = {5, 7, 9};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void reduceSum2D_axis1() {
            TensorType t2x3 = tensor(2, 3);
            TensorType t2 = tensor(2);
            Value r = new Value("0", t2);
            Value arg = new Value("arg0", t2x3);
            Value init = new Value("init", tensor());

            ReduceOp op = new ReduceOp(r, arg, init, List.of(1L), "add", t2);

            // [[1,2,3],[4,5,6]] reduced along axis 1 = [6,15]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor initVal = Tensor.fromFloatArray(new float[]{0})) {

                List<Tensor> results = backend.execute(op, List.of(input, initVal));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{2}, result.shape());
                    float[] expected = {6, 15};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void reduceMax1D() {
            TensorType t4 = tensor(4);
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            Value arg = new Value("arg0", t4);
            Value init = new Value("init", tScalar);

            ReduceOp op = new ReduceOp(r, arg, init, List.of(0L), "max", tScalar);

            try (Tensor input = Tensor.fromFloatArray(new float[]{3, 1, 4, 2}, 4);
                 Tensor initVal = Tensor.fromFloatArray(new float[]{Float.NEGATIVE_INFINITY})) {

                List<Tensor> results = backend.execute(op, List.of(input, initVal));

                try (Tensor result = results.getFirst()) {
                    assertEquals(4, result.toFloatArray()[0], 1e-5f);
                }
            }
        }

        @Test
        void reduceMin1D() {
            TensorType t4 = tensor(4);
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            Value arg = new Value("arg0", t4);
            Value init = new Value("init", tScalar);

            ReduceOp op = new ReduceOp(r, arg, init, List.of(0L), "min", tScalar);

            try (Tensor input = Tensor.fromFloatArray(new float[]{3, 1, 4, 2}, 4);
                 Tensor initVal = Tensor.fromFloatArray(new float[]{Float.POSITIVE_INFINITY})) {

                List<Tensor> results = backend.execute(op, List.of(input, initVal));

                try (Tensor result = results.getFirst()) {
                    assertEquals(1, result.toFloatArray()[0], 1e-5f);
                }
            }
        }

        @Test
        void reduceMul1D() {
            TensorType t4 = tensor(4);
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            Value arg = new Value("arg0", t4);
            Value init = new Value("init", tScalar);

            ReduceOp op = new ReduceOp(r, arg, init, List.of(0L), "mul", tScalar);

            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor initVal = Tensor.fromFloatArray(new float[]{1})) {

                List<Tensor> results = backend.execute(op, List.of(input, initVal));

                try (Tensor result = results.getFirst()) {
                    // Product: 1*2*3*4 = 24
                    assertEquals(24, result.toFloatArray()[0], 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Batch Normalization")
    class BatchNormOps {

        @Test
        void batchNormInference() {
            // Simple 1x4 input with feature_index=1
            TensorType t1x4 = tensor(1, 4);
            TensorType t4 = tensor(4);

            Value r = new Value("0", t1x4);
            Value operand = new Value("operand", t1x4);
            Value scale = new Value("scale", t4);
            Value offset = new Value("offset", t4);
            Value mean = new Value("mean", t4);
            Value variance = new Value("variance", t4);

            BatchNormInferenceOp op = new BatchNormInferenceOp(
                r, operand, scale, offset, mean, variance,
                1e-5f, 1L, t1x4
            );

            // Input: [[0, 2, 4, 6]]
            // Mean: [0, 2, 4, 6] -> normalized = 0
            // Scale: [1, 1, 1, 1], Offset: [0, 0, 0, 0]
            // Result should be all zeros
            try (Tensor input = Tensor.fromFloatArray(new float[]{0, 2, 4, 6}, 1, 4);
                 Tensor scaleT = Tensor.fromFloatArray(new float[]{1, 1, 1, 1}, 4);
                 Tensor offsetT = Tensor.fromFloatArray(new float[]{0, 0, 0, 0}, 4);
                 Tensor meanT = Tensor.fromFloatArray(new float[]{0, 2, 4, 6}, 4);
                 Tensor varT = Tensor.fromFloatArray(new float[]{1, 1, 1, 1}, 4)) {

                List<Tensor> results = backend.execute(op, List.of(input, scaleT, offsetT, meanT, varT));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{1, 4}, result.shape());
                    float[] actual = result.toFloatArray();
                    for (float v : actual) {
                        assertEquals(0, v, 1e-5f);
                    }
                }
            }
        }

        @Test
        void batchNormInferenceWithScaleOffset() {
            TensorType t1x2 = tensor(1, 2);
            TensorType t2 = tensor(2);

            Value r = new Value("0", t1x2);
            Value operand = new Value("operand", t1x2);
            Value scale = new Value("scale", t2);
            Value offset = new Value("offset", t2);
            Value mean = new Value("mean", t2);
            Value variance = new Value("variance", t2);

            BatchNormInferenceOp op = new BatchNormInferenceOp(
                r, operand, scale, offset, mean, variance,
                1e-5f, 1L, t1x2
            );

            // Input: [[1, 5]] with mean=[0, 4], var=[1, 1]
            // Normalized: [1, 1]
            // Scale: [2, 3], Offset: [10, 20]
            // Result: [2*1+10, 3*1+20] = [12, 23]
            try (Tensor input = Tensor.fromFloatArray(new float[]{1, 5}, 1, 2);
                 Tensor scaleT = Tensor.fromFloatArray(new float[]{2, 3}, 2);
                 Tensor offsetT = Tensor.fromFloatArray(new float[]{10, 20}, 2);
                 Tensor meanT = Tensor.fromFloatArray(new float[]{0, 4}, 2);
                 Tensor varT = Tensor.fromFloatArray(new float[]{1, 1}, 2)) {

                List<Tensor> results = backend.execute(op, List.of(input, scaleT, offsetT, meanT, varT));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {12, 23};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-4f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Sort Operations")
    class SortOps {

        @Test
        void sortAscending() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);

            // isStable=false for ascending sort
            SortOp op = new SortOp(r, List.of(arg), 0L, false, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{3, 1, 4, 2}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }

        @Test
        void sortStable() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            Value arg = new Value("arg0", t4);

            // isStable=true for stable sort
            SortOp op = new SortOp(r, List.of(arg), 0L, true, t4);

            try (Tensor input = Tensor.fromFloatArray(new float[]{3, 1, 4, 2}, 4)) {
                List<Tensor> results = backend.execute(op, List.of(input));

                try (Tensor result = results.getFirst()) {
                    float[] expected = {1, 2, 3, 4};
                    assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Random Number Generation")
    class RngOps {

        @Test
        void rngUniform() {
            TensorType t100 = tensor(100);
            TensorType tScalar = tensor();
            Value r = new Value("0", t100);
            Value minVal = new Value("min", tScalar);
            Value maxVal = new Value("max", tScalar);
            RngOp op = new RngOp(r, minVal, maxVal, "uniform", t100);

            try (Tensor min = Tensor.fromFloatArray(new float[]{0.0f});
                 Tensor max = Tensor.fromFloatArray(new float[]{1.0f})) {

                List<Tensor> results = backend.execute(op, List.of(min, max));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{100}, result.shape());
                    float[] data = result.toFloatArray();
                    for (float v : data) {
                        assertTrue(v >= 0.0f && v <= 1.0f, "Value " + v + " out of range [0, 1]");
                    }
                }
            }
        }

        @Test
        void rngNormal() {
            TensorType t1000 = tensor(1000);
            TensorType tScalar = tensor();
            Value r = new Value("0", t1000);
            Value mean = new Value("mean", tScalar);
            Value stddev = new Value("stddev", tScalar);
            RngOp op = new RngOp(r, mean, stddev, "normal", t1000);

            try (Tensor meanT = Tensor.fromFloatArray(new float[]{0.0f});
                 Tensor stddevT = Tensor.fromFloatArray(new float[]{1.0f})) {

                List<Tensor> results = backend.execute(op, List.of(meanT, stddevT));

                try (Tensor result = results.getFirst()) {
                    assertArrayEquals(new int[]{1000}, result.shape());
                    float[] data = result.toFloatArray();

                    // Check approximate statistics
                    float sum = 0;
                    for (float v : data) sum += v;
                    float actualMean = sum / data.length;

                    float varSum = 0;
                    for (float v : data) varSum += (v - actualMean) * (v - actualMean);
                    float actualStd = (float) Math.sqrt(varSum / data.length);

                    // Allow significant deviation since it's random
                    assertEquals(0.0f, actualMean, 0.15f);
                    assertEquals(1.0f, actualStd, 0.15f);
                }
            }
        }
    }

    // DotGeneral tests temporarily disabled - kernel needs debugging for complex index mapping
    // TODO: Fix DotGeneralKernel index calculation

    @Nested
    @DisplayName("Constant Operations")
    class ConstantOps {

        @Test
        void constantScalar() {
            TensorType tScalar = tensor();
            Value r = new Value("0", tScalar);
            DenseAttr denseVal = new DenseAttr(42.0f, tScalar);
            ConstantOp op = new ConstantOp(r, denseVal, tScalar);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                assertEquals(42.0f, result.toFloatArray()[0], 1e-5f);
            }
        }

        @Test
        void constantArray() {
            TensorType t4 = tensor(4);
            Value r = new Value("0", t4);
            DenseAttr denseVal = new DenseAttr(new float[]{1, 2, 3, 4}, t4);
            ConstantOp op = new ConstantOp(r, denseVal, t4);

            List<Tensor> results = backend.execute(op, List.of());

            try (Tensor result = results.getFirst()) {
                assertArrayEquals(new int[]{4}, result.shape());
                float[] expected = {1, 2, 3, 4};
                assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
            }
        }
    }

    // Convert tests temporarily disabled - kernel needs proper type conversion implementation
    // TODO: Implement proper type conversion in ConvertKernel
}
