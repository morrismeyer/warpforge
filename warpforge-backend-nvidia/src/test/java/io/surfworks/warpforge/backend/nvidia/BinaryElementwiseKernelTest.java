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

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for all binary elementwise CUDA kernels.
 *
 * <p>Each test outputs its status for visibility in CI logs.
 */
@DisplayName("Binary Elementwise CUDA Kernels")
class BinaryElementwiseKernelTest {

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
    @DisplayName("PTX: Subtract generates valid output")
    void testSubtractPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Subtract");
        String ptx = CudaKernels.generateSubtractF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry subtract_f32"));
        assertTrue(ptx.contains("sub.f32"));
        System.out.println("[PASS] Subtract PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Divide generates valid output")
    void testDividePtxGeneration() {
        System.out.println("[TEST] PTX Generation: Divide");
        String ptx = CudaKernels.generateDivideF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry divide_f32"));
        assertTrue(ptx.contains("div.approx.f32"));
        System.out.println("[PASS] Divide PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Maximum generates valid output")
    void testMaximumPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Maximum");
        String ptx = CudaKernels.generateMaximumF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry maximum_f32"));
        assertTrue(ptx.contains("max.f32"));
        System.out.println("[PASS] Maximum PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Minimum generates valid output")
    void testMinimumPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Minimum");
        String ptx = CudaKernels.generateMinimumF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry minimum_f32"));
        assertTrue(ptx.contains("min.f32"));
        System.out.println("[PASS] Minimum PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Power generates valid output")
    void testPowerPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Power");
        String ptx = CudaKernels.generatePowerF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry power_f32"));
        assertTrue(ptx.contains("lg2.approx.f32"));
        assertTrue(ptx.contains("ex2.approx.f32"));
        System.out.println("[PASS] Power PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Remainder generates valid output")
    void testRemainderPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Remainder");
        String ptx = CudaKernels.generateRemainderF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry remainder_f32"));
        assertTrue(ptx.contains("div.approx.f32"));
        assertTrue(ptx.contains("cvt.rzi.f32.f32")); // truncate
        System.out.println("[PASS] Remainder PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All operations support SALT_TIMING")
    void testAllOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All operations with SALT_TIMING");

        String[] ops = {"Subtract", "Divide", "Maximum", "Minimum", "Power", "Remainder"};
        String[] ptxSources = {
            CudaKernels.generateSubtractF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateDivideF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateMaximumF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateMinimumF32(CudaKernels.SALT_TIMING),
            CudaKernels.generatePowerF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateRemainderF32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Subtract executes correctly")
    void testSubtractExecution() {
        System.out.println("[TEST] CUDA Execution: Subtract");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        float[] b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] expected = {9.0f, 18.0f, 27.0f, 36.0f, 45.0f, 54.0f, 63.0f, 72.0f};

        float[] result = executeSubtract(a, b);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Subtract execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Divide executes correctly")
    void testDivideExecution() {
        System.out.println("[TEST] CUDA Execution: Divide");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        float[] b = {2.0f, 4.0f, 5.0f, 8.0f, 10.0f, 6.0f, 7.0f, 8.0f};
        float[] expected = {5.0f, 5.0f, 6.0f, 5.0f, 5.0f, 10.0f, 10.0f, 10.0f};

        float[] result = executeDivide(a, b);
        assertArrayEquals(expected, result, 1e-4f); // div.approx has less precision

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Divide execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Maximum executes correctly")
    void testMaximumExecution() {
        System.out.println("[TEST] CUDA Execution: Maximum");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {1.0f, 5.0f, 3.0f, 8.0f, 2.0f, 9.0f, 4.0f, 7.0f};
        float[] b = {4.0f, 2.0f, 6.0f, 1.0f, 9.0f, 3.0f, 8.0f, 5.0f};
        float[] expected = {4.0f, 5.0f, 6.0f, 8.0f, 9.0f, 9.0f, 8.0f, 7.0f};

        float[] result = executeMaximum(a, b);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Maximum execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Minimum executes correctly")
    void testMinimumExecution() {
        System.out.println("[TEST] CUDA Execution: Minimum");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {1.0f, 5.0f, 3.0f, 8.0f, 2.0f, 9.0f, 4.0f, 7.0f};
        float[] b = {4.0f, 2.0f, 6.0f, 1.0f, 9.0f, 3.0f, 8.0f, 5.0f};
        float[] expected = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        float[] result = executeMinimum(a, b);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Minimum execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Power executes correctly")
    void testPowerExecution() {
        System.out.println("[TEST] CUDA Execution: Power");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {2.0f, 3.0f, 4.0f, 10.0f, 2.0f, 5.0f, 1.0f, 16.0f};
        float[] b = {3.0f, 2.0f, 0.5f, 2.0f, 10.0f, 0.0f, 100.0f, 0.25f};
        // Expected: 8, 9, 2, 100, 1024, 1, 1, 2

        float[] result = executePower(a, b);

        assertEquals(8.0f, result[0], 1e-2f, "2^3");
        assertEquals(9.0f, result[1], 1e-2f, "3^2");
        assertEquals(2.0f, result[2], 1e-2f, "4^0.5");
        assertEquals(100.0f, result[3], 1e-1f, "10^2");
        assertEquals(1024.0f, result[4], 1e-0f, "2^10");
        assertEquals(1.0f, result[5], 1e-3f, "5^0");
        assertEquals(1.0f, result[6], 1e-3f, "1^100");
        assertEquals(2.0f, result[7], 1e-2f, "16^0.25");

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Power execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Remainder executes correctly")
    void testRemainderExecution() {
        System.out.println("[TEST] CUDA Execution: Remainder");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {10.0f, 10.0f, 7.5f, 5.0f, -10.0f, 10.0f, 3.0f, 100.0f};
        float[] b = {3.0f, 4.0f, 2.0f, 3.0f, 3.0f, -3.0f, 5.0f, 7.0f};
        // Expected: 1, 2, 1.5, 2, -1, 1, 3, 2

        float[] result = executeRemainder(a, b);

        assertEquals(1.0f, result[0], 1e-3f, "10 % 3");
        assertEquals(2.0f, result[1], 1e-3f, "10 % 4");
        assertEquals(1.5f, result[2], 1e-3f, "7.5 % 2");
        assertEquals(2.0f, result[3], 1e-3f, "5 % 3");
        assertEquals(-1.0f, result[4], 1e-3f, "-10 % 3");
        assertEquals(1.0f, result[5], 1e-3f, "10 % -3");
        assertEquals(3.0f, result[6], 1e-3f, "3 % 5");
        assertEquals(2.0f, result[7], 1e-3f, "100 % 7");

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Remainder execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All operations handle large tensors (1M elements)")
    void testLargeTensorAllOperations() {
        System.out.println("[TEST] CUDA Large Tensor: All operations (1M elements)");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int n = 1_000_000;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (i + 1) * 0.001f;
            b[i] = (i + 1) * 0.0005f;
        }

        // Test Subtract
        System.out.println("  Testing Subtract...");
        float[] subResult = executeSubtract(a, b);
        assertEquals(a[0] - b[0], subResult[0], 1e-5f);
        assertEquals(a[n/2] - b[n/2], subResult[n/2], 1e-4f);
        System.out.println("  [OK] Subtract passed");

        // Test Divide
        System.out.println("  Testing Divide...");
        float[] divResult = executeDivide(a, b);
        assertEquals(a[0] / b[0], divResult[0], 1e-3f); // div.approx
        System.out.println("  [OK] Divide passed");

        // Test Maximum
        System.out.println("  Testing Maximum...");
        float[] maxResult = executeMaximum(a, b);
        assertEquals(Math.max(a[0], b[0]), maxResult[0], 1e-5f);
        assertEquals(Math.max(a[n-1], b[n-1]), maxResult[n-1], 1e-4f);
        System.out.println("  [OK] Maximum passed");

        // Test Minimum
        System.out.println("  Testing Minimum...");
        float[] minResult = executeMinimum(a, b);
        assertEquals(Math.min(a[0], b[0]), minResult[0], 1e-5f);
        assertEquals(Math.min(a[n-1], b[n-1]), minResult[n-1], 1e-4f);
        System.out.println("  [OK] Minimum passed");

        System.out.println("[PASS] All large tensor operations OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Operations handle negative numbers")
    void testNegativeNumbers() {
        System.out.println("[TEST] CUDA Edge Case: Negative numbers");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {-5.0f, 3.0f, -2.0f, 7.0f};
        float[] b = {2.0f, -4.0f, -3.0f, 1.0f};

        // Subtract: [-7, 7, 1, 6]
        float[] subResult = executeSubtract(a, b);
        assertArrayEquals(new float[]{-7.0f, 7.0f, 1.0f, 6.0f}, subResult, 1e-5f);
        System.out.println("  [OK] Subtract with negatives: " + Arrays.toString(subResult));

        // Divide: [-2.5, -0.75, 0.666..., 7]
        float[] divResult = executeDivide(a, b);
        assertEquals(-2.5f, divResult[0], 1e-3f);
        System.out.println("  [OK] Divide with negatives: " + Arrays.toString(divResult));

        // Maximum: [2, 3, -2, 7]
        float[] maxResult = executeMaximum(a, b);
        assertArrayEquals(new float[]{2.0f, 3.0f, -2.0f, 7.0f}, maxResult, 1e-5f);
        System.out.println("  [OK] Maximum with negatives: " + Arrays.toString(maxResult));

        // Minimum: [-5, -4, -3, 1]
        float[] minResult = executeMinimum(a, b);
        assertArrayEquals(new float[]{-5.0f, -4.0f, -3.0f, 1.0f}, minResult, 1e-5f);
        System.out.println("  [OK] Minimum with negatives: " + Arrays.toString(minResult));

        System.out.println("[PASS] Negative number handling OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Operations with SALT_TIMING produce correct results")
    void testTimingInstrumentation() {
        System.out.println("[TEST] CUDA Instrumentation: SALT_TIMING");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");

        try (NvidiaBackend timedBackend = new NvidiaBackend(0, CudaKernels.SALT_TIMING)) {
            assumeTrue(timedBackend.hasCudaContext(), "CUDA context not available");

            float[] a = {10.0f, 20.0f, 30.0f, 40.0f};
            float[] b = {2.0f, 5.0f, 6.0f, 8.0f};

            // Test each operation with timing
            System.out.println("  Testing Subtract with timing...");
            float[] subResult = executeBinaryOp(timedBackend, a, b, StableHloAst.SubtractOp.class);
            assertArrayEquals(new float[]{8.0f, 15.0f, 24.0f, 32.0f}, subResult, 1e-5f);
            System.out.println("  [OK] Subtract with timing");

            System.out.println("  Testing Divide with timing...");
            float[] divResult = executeBinaryOp(timedBackend, a, b, StableHloAst.DivideOp.class);
            assertEquals(5.0f, divResult[0], 1e-3f);
            System.out.println("  [OK] Divide with timing");

            System.out.println("  Testing Maximum with timing...");
            float[] maxResult = executeBinaryOp(timedBackend, a, b, StableHloAst.MaximumOp.class);
            assertArrayEquals(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, maxResult, 1e-5f);
            System.out.println("  [OK] Maximum with timing");

            System.out.println("  Testing Minimum with timing...");
            float[] minResult = executeBinaryOp(timedBackend, a, b, StableHloAst.MinimumOp.class);
            assertArrayEquals(new float[]{2.0f, 5.0f, 6.0f, 8.0f}, minResult, 1e-5f);
            System.out.println("  [OK] Minimum with timing");
        }

        System.out.println("[PASS] SALT_TIMING instrumentation OK");
    }

    // ==================== Helper Methods ====================

    private float[] executeSubtract(float[] a, float[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.SubtractOp.class);
    }

    private float[] executeDivide(float[] a, float[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.DivideOp.class);
    }

    private float[] executeMaximum(float[] a, float[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.MaximumOp.class);
    }

    private float[] executeMinimum(float[] a, float[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.MinimumOp.class);
    }

    private float[] executePower(float[] a, float[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.PowerOp.class);
    }

    private float[] executeRemainder(float[] a, float[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.RemainderOp.class);
    }

    private float[] executeBinaryOp(NvidiaBackend backend, float[] a, float[] b,
                                     Class<? extends StableHloAst.Operation> opClass) {
        int n = a.length;

        try (Tensor tensorA = Tensor.fromFloatArray(a, n);
             Tensor tensorB = Tensor.fromFloatArray(b, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Operation op = createOp(opClass, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorA, tensorB));

            return results.get(0).toFloatArray();
        }
    }

    private StableHloAst.Operation createOp(Class<? extends StableHloAst.Operation> opClass,
                                             StableHloAst.TensorType resultType) {
        StableHloAst.Value lhs = new StableHloAst.Value("0", resultType);
        StableHloAst.Value rhs = new StableHloAst.Value("1", resultType);
        StableHloAst.Value result = new StableHloAst.Value("2", resultType);

        if (opClass == StableHloAst.SubtractOp.class) {
            return new StableHloAst.SubtractOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.DivideOp.class) {
            return new StableHloAst.DivideOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.MaximumOp.class) {
            return new StableHloAst.MaximumOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.MinimumOp.class) {
            return new StableHloAst.MinimumOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.PowerOp.class) {
            return new StableHloAst.PowerOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.RemainderOp.class) {
            return new StableHloAst.RemainderOp(lhs, rhs, result, resultType);
        } else {
            throw new IllegalArgumentException("Unknown operation class: " + opClass);
        }
    }
}
