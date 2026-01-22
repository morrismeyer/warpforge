package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for matrix multiplication (Dot) and reduction CUDA kernels.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Matrix and Reduce CUDA Kernels")
class MatrixReduceKernelTest {

    private static final float EPSILON = 1e-5f;

    private NvidiaBackend backend;

    @BeforeEach
    void setUp() {
        backend = new NvidiaBackend(0, CudaKernels.SALT_NONE);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== PTX Generation Tests (No CUDA Required) ====================

    @Test
    @DisplayName("PTX: Dot generates valid output")
    void testDotPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Dot");
        String ptx = CudaKernels.generateDotF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry dot_f32"));
        assertTrue(ptx.contains("fma.rn.f32"), "Expected fused multiply-add instruction");
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Dot PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reduce add generates valid output")
    void testReduceAddPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reduce add");
        String ptx = CudaKernels.generateReduceAddF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reduce_add_f32"));
        assertTrue(ptx.contains("add.f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Reduce add PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reduce max generates valid output")
    void testReduceMaxPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reduce max");
        String ptx = CudaKernels.generateReduceMaxF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reduce_max_f32"));
        assertTrue(ptx.contains("max.f32"));
        System.out.println("[PASS] Reduce max PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reduce min generates valid output")
    void testReduceMinPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reduce min");
        String ptx = CudaKernels.generateReduceMinF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reduce_min_f32"));
        assertTrue(ptx.contains("min.f32"));
        System.out.println("[PASS] Reduce min PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reduce mul generates valid output")
    void testReduceMulPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reduce mul");
        String ptx = CudaKernels.generateReduceMulF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reduce_mul_f32"));
        assertTrue(ptx.contains("mul.f32"));
        System.out.println("[PASS] Reduce mul PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Dot supports SALT_TIMING")
    void testDotTimingSupport() {
        System.out.println("[TEST] PTX Generation: Dot with SALT_TIMING");
        String ptx = CudaKernels.generateDotF32(CudaKernels.SALT_TIMING);

        assertTrue(ptx.contains("timing_ptr"), "Dot should have timing_ptr parameter");
        assertTrue(ptx.contains("%globaltimer"), "Dot should use globaltimer");
        System.out.println("[PASS] Dot supports SALT_TIMING");
    }

    @Test
    @DisplayName("PTX: All reduce operations support SALT_TIMING")
    void testAllReduceOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All reduce operations with SALT_TIMING");

        String[] ops = {"add", "max", "min", "mul"};
        String[] ptxSources = {
            CudaKernels.generateReduceAddF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateReduceMaxF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateReduceMinF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateReduceMulF32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                "Reduce " + ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                "Reduce " + ops[i] + " should use globaltimer");
            System.out.println("  [OK] Reduce " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All reduce operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Dot executes correctly (2x2 * 2x2)")
    void testDotExecution2x2() {
        System.out.println("[TEST] CUDA Execution: Dot 2x2 * 2x2");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        float[] matA = {1, 2, 3, 4};
        float[] matB = {5, 6, 7, 8};
        float[] expected = {19, 22, 43, 50};

        float[] result = executeDot(matA, new int[]{2, 2}, matB, new int[]{2, 2});
        assertArrayEqualsWithTolerance(expected, result, EPSILON);

        System.out.println("  A[2,2] * B[2,2] = C[2,2]");
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Dot 2x2 * 2x2 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Dot executes correctly (3x2 * 2x4)")
    void testDotExecution3x2_2x4() {
        System.out.println("[TEST] CUDA Execution: Dot 3x2 * 2x4");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8]]  (2x4)
        // C = A * B = [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]  (3x4)
        float[] matA = {1, 2, 3, 4, 5, 6};
        float[] matB = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] expected = {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68};

        float[] result = executeDot(matA, new int[]{3, 2}, matB, new int[]{2, 4});
        assertArrayEqualsWithTolerance(expected, result, EPSILON);

        System.out.println("  A[3,2] * B[2,4] = C[3,4]");
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Dot 3x2 * 2x4 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Dot identity matrix")
    void testDotIdentityMatrix() {
        System.out.println("[TEST] CUDA Execution: Dot with identity matrix");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  (3x3)
        // A * I = A
        float[] matA = {1, 2, 3, 4, 5, 6};
        float[] identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float[] expected = {1, 2, 3, 4, 5, 6};

        float[] result = executeDot(matA, new int[]{2, 3}, identity, new int[]{3, 3});
        assertArrayEqualsWithTolerance(expected, result, EPSILON);

        System.out.println("[PASS] Dot with identity matrix OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reduce add executes correctly")
    void testReduceAddExecution() {
        System.out.println("[TEST] CUDA Execution: Reduce add");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float expected = 15.0f;

        float result = executeReduce(input, "add");
        assertEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Sum: " + result);
        System.out.println("[PASS] Reduce add OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reduce max executes correctly")
    void testReduceMaxExecution() {
        System.out.println("[TEST] CUDA Execution: Reduce max");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
        float expected = 9.0f;

        float result = executeReduce(input, "max");
        assertEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Max: " + result);
        System.out.println("[PASS] Reduce max OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reduce min executes correctly")
    void testReduceMinExecution() {
        System.out.println("[TEST] CUDA Execution: Reduce min");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
        float expected = 1.0f;

        float result = executeReduce(input, "min");
        assertEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Min: " + result);
        System.out.println("[PASS] Reduce min OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reduce mul executes correctly")
    void testReduceMulExecution() {
        System.out.println("[TEST] CUDA Execution: Reduce mul");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float expected = 24.0f;

        float result = executeReduce(input, "mul");
        assertEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Product: " + result);
        System.out.println("[PASS] Reduce mul OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reduce add with large input")
    void testReduceAddLargeInput() {
        System.out.println("[TEST] CUDA Execution: Reduce add (large input)");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int n = 1000;
        float[] input = new float[n];
        float expected = 0;
        for (int i = 0; i < n; i++) {
            input[i] = (i + 1);
            expected += (i + 1);
        }

        float result = executeReduce(input, "add");
        // Sum of 1 to 1000 = n*(n+1)/2 = 500500
        assertEquals(expected, result, 1.0f); // Allow larger tolerance for large sum

        System.out.println("  Input: [1, 2, ..., " + n + "]");
        System.out.println("  Sum: " + result + " (expected " + expected + ")");
        System.out.println("[PASS] Reduce add (large input) OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Matrix/Reduce Summary")
    void testAllMatrixReduceSummary() {
        System.out.println("========================================");
        System.out.println("Matrix and Reduce Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Dot
        System.out.print("  Dot: ");
        float[] dotA = {1, 2, 3, 4};
        float[] dotB = {5, 6, 7, 8};
        float[] dotResult = executeDot(dotA, new int[]{2, 2}, dotB, new int[]{2, 2});
        assertEquals(19f, dotResult[0], EPSILON);
        System.out.println("[OK]");

        // Reduce add
        System.out.print("  Reduce add: ");
        float sumResult = executeReduce(new float[]{1, 2, 3, 4, 5}, "add");
        assertEquals(15f, sumResult, EPSILON);
        System.out.println("[OK]");

        // Reduce max
        System.out.print("  Reduce max: ");
        float maxResult = executeReduce(new float[]{3, 1, 4, 1, 5, 9}, "max");
        assertEquals(9f, maxResult, EPSILON);
        System.out.println("[OK]");

        // Reduce min
        System.out.print("  Reduce min: ");
        float minResult = executeReduce(new float[]{3, 1, 4, 1, 5, 9}, "min");
        assertEquals(1f, minResult, EPSILON);
        System.out.println("[OK]");

        // Reduce mul
        System.out.print("  Reduce mul: ");
        float mulResult = executeReduce(new float[]{1, 2, 3, 4}, "mul");
        assertEquals(24f, mulResult, EPSILON);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 5 matrix/reduce operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private float[] executeDot(float[] matA, int[] shapeA, float[] matB, int[] shapeB) {
        int M = shapeA[0];
        int K = shapeA[1];
        int N = shapeB[1];

        try (Tensor tensorA = Tensor.fromFloatArray(matA, shapeA);
             Tensor tensorB = Tensor.fromFloatArray(matB, shapeB)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(M, N),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Value lhs = new StableHloAst.Value("0", new StableHloAst.TensorType(
                List.of(M, K), StableHloAst.ScalarType.F32));
            StableHloAst.Value rhs = new StableHloAst.Value("1", new StableHloAst.TensorType(
                List.of(K, N), StableHloAst.ScalarType.F32));
            StableHloAst.Value result = new StableHloAst.Value("2", resultType);

            StableHloAst.DotOp op = new StableHloAst.DotOp(result, lhs, rhs, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorA, tensorB));

            return results.get(0).toFloatArray();
        }
    }

    private float executeReduce(float[] input, String reducer) {
        int n = input.length;

        try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(), // scalar
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value initValue = new StableHloAst.Value("1", resultType);
            StableHloAst.Value result = new StableHloAst.Value("2", resultType);

            StableHloAst.ReduceOp op = new StableHloAst.ReduceOp(
                result,
                operand,
                initValue,
                List.of(0L), // reduce over dimension 0 (all elements)
                reducer,
                resultType
            );

            // Create init tensor with identity value
            float initVal = switch (reducer) {
                case "add" -> 0.0f;
                case "max" -> Float.NEGATIVE_INFINITY;
                case "min" -> Float.POSITIVE_INFINITY;
                case "mul" -> 1.0f;
                default -> 0.0f;
            };

            try (Tensor initTensor = Tensor.fromFloatArray(new float[]{initVal})) {
                List<Tensor> results = backend.execute(op, List.of(tensorIn, initTensor));
                return results.get(0).toFloatArray()[0];
            }
        }
    }

    private void assertArrayEqualsWithTolerance(float[] expected, float[] actual, float tolerance) {
        assertEquals(expected.length, actual.length, "Array lengths differ");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance,
                "Arrays differ at index " + i);
        }
    }
}
