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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for unary elementwise CUDA kernels (Negate, Abs, Exp, Log, Sqrt, Tanh).
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Unary Elementwise CUDA Kernels")
class UnaryElementwiseKernelTest {

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
    @DisplayName("PTX: Negate generates valid output")
    void testNegatePtxGeneration() {
        System.out.println("[TEST] PTX Generation: Negate");
        String ptx = CudaKernels.generateNegateF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry negate_f32"));
        assertTrue(ptx.contains("neg.f32"));
        System.out.println("[PASS] Negate PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Abs generates valid output")
    void testAbsPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Abs");
        String ptx = CudaKernels.generateAbsF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry abs_f32"));
        assertTrue(ptx.contains("abs.f32"));
        System.out.println("[PASS] Abs PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Exp generates valid output")
    void testExpPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Exp");
        String ptx = CudaKernels.generateExpF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry exp_f32"));
        assertTrue(ptx.contains("ex2.approx.f32"));
        System.out.println("[PASS] Exp PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Log generates valid output")
    void testLogPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Log");
        String ptx = CudaKernels.generateLogF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry log_f32"));
        assertTrue(ptx.contains("lg2.approx.f32"));
        System.out.println("[PASS] Log PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Sqrt generates valid output")
    void testSqrtPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Sqrt");
        String ptx = CudaKernels.generateSqrtF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry sqrt_f32"));
        assertTrue(ptx.contains("sqrt.approx.f32"));
        System.out.println("[PASS] Sqrt PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Tanh generates valid output")
    void testTanhPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Tanh");
        String ptx = CudaKernels.generateTanhF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry tanh_f32"));
        assertTrue(ptx.contains("ex2.approx.f32")); // tanh uses exp
        System.out.println("[PASS] Tanh PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All unary operations support SALT_TIMING")
    void testAllUnaryOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All unary operations with SALT_TIMING");

        String[] ops = {"Negate", "Abs", "Exp", "Log", "Sqrt", "Tanh"};
        String[] ptxSources = {
            CudaKernels.generateNegateF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateAbsF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateExpF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateLogF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateSqrtF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateTanhF32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All unary operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Negate executes correctly")
    void testNegateExecution() {
        System.out.println("[TEST] CUDA Execution: Negate");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f, 5.5f, -6.5f, 100.0f};
        float[] expected = {-1.0f, 2.0f, -3.0f, 4.0f, 0.0f, -5.5f, 6.5f, -100.0f};

        float[] result = executeNegate(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "negate(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Negate execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Abs executes correctly")
    void testAbsExecution() {
        System.out.println("[TEST] CUDA Execution: Abs");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, -2.0f, -3.0f, 4.0f, 0.0f, -5.5f, 6.5f, -100.0f};
        float[] expected = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 5.5f, 6.5f, 100.0f};

        float[] result = executeAbs(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "abs(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Abs execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Exp executes correctly")
    void testExpExecution() {
        System.out.println("[TEST] CUDA Execution: Exp");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {0.0f, 1.0f, -1.0f, 2.0f};

        float[] result = executeExp(input);

        assertEquals(1.0f, result[0], 1e-4f, "exp(0)");
        assertEquals((float) Math.exp(1), result[1], 1e-3f, "exp(1)");
        assertEquals((float) Math.exp(-1), result[2], 1e-3f, "exp(-1)");
        assertEquals((float) Math.exp(2), result[3], 1e-2f, "exp(2)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Exp execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Log executes correctly")
    void testLogExecution() {
        System.out.println("[TEST] CUDA Execution: Log");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, (float) Math.E, 10.0f, 100.0f};

        float[] result = executeLog(input);

        assertEquals(0.0f, result[0], 1e-4f, "log(1)");
        assertEquals(1.0f, result[1], 1e-3f, "log(e)");
        assertEquals((float) Math.log(10), result[2], 1e-2f, "log(10)");
        assertEquals((float) Math.log(100), result[3], 1e-2f, "log(100)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Log execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Sqrt executes correctly")
    void testSqrtExecution() {
        System.out.println("[TEST] CUDA Execution: Sqrt");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 100.0f, 0.0f, 2.0f};

        float[] result = executeSqrt(input);

        assertEquals(1.0f, result[0], 1e-4f, "sqrt(1)");
        assertEquals(2.0f, result[1], 1e-4f, "sqrt(4)");
        assertEquals(3.0f, result[2], 1e-4f, "sqrt(9)");
        assertEquals(4.0f, result[3], 1e-4f, "sqrt(16)");
        assertEquals(5.0f, result[4], 1e-4f, "sqrt(25)");
        assertEquals(10.0f, result[5], 1e-3f, "sqrt(100)");
        assertEquals(0.0f, result[6], 1e-4f, "sqrt(0)");
        assertEquals((float) Math.sqrt(2), result[7], 1e-4f, "sqrt(2)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Sqrt execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Tanh executes correctly")
    void testTanhExecution() {
        System.out.println("[TEST] CUDA Execution: Tanh");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};

        float[] result = executeTanh(input);

        assertEquals(0.0f, result[0], 1e-4f, "tanh(0)");
        assertEquals((float) Math.tanh(1), result[1], 1e-3f, "tanh(1)");
        assertEquals((float) Math.tanh(-1), result[2], 1e-3f, "tanh(-1)");
        assertEquals((float) Math.tanh(2), result[3], 1e-3f, "tanh(2)");
        assertEquals((float) Math.tanh(-2), result[4], 1e-3f, "tanh(-2)");
        assertEquals((float) Math.tanh(0.5), result[5], 1e-3f, "tanh(0.5)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Tanh execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All unary operations handle large tensors (1M elements)")
    void testLargeTensorAllOperations() {
        System.out.println("[TEST] CUDA Large Tensor: All unary operations (1M elements)");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int n = 1_000_000;
        float[] input = new float[n];
        for (int i = 0; i < n; i++) {
            input[i] = (i + 1) * 0.001f;
        }

        // Test Negate
        System.out.println("  Testing Negate...");
        float[] negResult = executeNegate(input);
        assertEquals(-input[0], negResult[0], 1e-5f);
        assertEquals(-input[n/2], negResult[n/2], 1e-4f);
        System.out.println("  [OK] Negate passed");

        // Test Abs (use negative input)
        System.out.println("  Testing Abs...");
        float[] negInput = new float[n];
        for (int i = 0; i < n; i++) {
            negInput[i] = -input[i];
        }
        float[] absResult = executeAbs(negInput);
        assertEquals(input[0], absResult[0], 1e-5f);
        System.out.println("  [OK] Abs passed");

        // Test Exp (use bounded values to avoid overflow)
        System.out.println("  Testing Exp...");
        float[] expInput = new float[n];
        for (int i = 0; i < n; i++) {
            expInput[i] = (float) (i % 10) * 0.3f; // values 0 to 2.7
        }
        float[] expResult = executeExp(expInput);
        assertEquals((float) Math.exp(expInput[0]), expResult[0], 1e-3f);
        System.out.println("  [OK] Exp passed");

        // Test Log
        System.out.println("  Testing Log...");
        float[] logResult = executeLog(input); // input is all positive
        assertEquals((float) Math.log(input[0]), logResult[0], 1e-2f);
        System.out.println("  [OK] Log passed");

        // Test Sqrt
        System.out.println("  Testing Sqrt...");
        float[] sqrtResult = executeSqrt(input);
        assertEquals((float) Math.sqrt(input[0]), sqrtResult[0], 1e-4f);
        System.out.println("  [OK] Sqrt passed");

        // Test Tanh
        System.out.println("  Testing Tanh...");
        float[] tanhResult = executeTanh(input);
        assertEquals((float) Math.tanh(input[0]), tanhResult[0], 1e-3f);
        System.out.println("  [OK] Tanh passed");

        System.out.println("[PASS] All large tensor unary operations OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Unary Elementwise Summary")
    void testAllUnaryElementwiseSummary() {
        System.out.println("========================================");
        System.out.println("Unary Elementwise Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Negate
        System.out.print("  Negate: ");
        float[] negInput = {5.0f, -3.0f};
        float[] negResult = executeNegate(negInput);
        assertTrue(Math.abs(negResult[0] - (-5.0f)) < 1e-5f);
        assertTrue(Math.abs(negResult[1] - 3.0f) < 1e-5f);
        System.out.println("[OK]");

        // Abs
        System.out.print("  Abs: ");
        float[] absInput = {-5.0f, 3.0f};
        float[] absResult = executeAbs(absInput);
        assertTrue(Math.abs(absResult[0] - 5.0f) < 1e-5f);
        assertTrue(Math.abs(absResult[1] - 3.0f) < 1e-5f);
        System.out.println("[OK]");

        // Exp
        System.out.print("  Exp: ");
        float[] expInput = {0.0f, 1.0f};
        float[] expResult = executeExp(expInput);
        assertTrue(Math.abs(expResult[0] - 1.0f) < 1e-4f);
        assertTrue(Math.abs(expResult[1] - (float) Math.E) < 1e-2f);
        System.out.println("[OK]");

        // Log
        System.out.print("  Log: ");
        float[] logInput = {1.0f, (float) Math.E};
        float[] logResult = executeLog(logInput);
        assertTrue(Math.abs(logResult[0]) < 1e-4f);
        assertTrue(Math.abs(logResult[1] - 1.0f) < 1e-2f);
        System.out.println("[OK]");

        // Sqrt
        System.out.print("  Sqrt: ");
        float[] sqrtInput = {4.0f, 9.0f};
        float[] sqrtResult = executeSqrt(sqrtInput);
        assertTrue(Math.abs(sqrtResult[0] - 2.0f) < 1e-4f);
        assertTrue(Math.abs(sqrtResult[1] - 3.0f) < 1e-4f);
        System.out.println("[OK]");

        // Tanh
        System.out.print("  Tanh: ");
        float[] tanhInput = {0.0f, 1.0f};
        float[] tanhResult = executeTanh(tanhInput);
        assertTrue(Math.abs(tanhResult[0]) < 1e-4f);
        assertTrue(Math.abs(tanhResult[1] - (float) Math.tanh(1)) < 1e-3f);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 6 unary elementwise operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private float[] executeNegate(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.NegateOp.class);
    }

    private float[] executeAbs(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.AbsOp.class);
    }

    private float[] executeExp(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.ExpOp.class);
    }

    private float[] executeLog(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.LogOp.class);
    }

    private float[] executeSqrt(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.SqrtOp.class);
    }

    private float[] executeTanh(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.TanhOp.class);
    }

    private float[] executeUnaryOp(NvidiaBackend backend, float[] input,
                                    Class<? extends StableHloAst.Operation> opClass) {
        int n = input.length;

        try (Tensor tensor = Tensor.fromFloatArray(input, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Operation op = createUnaryOp(opClass, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensor));

            return results.get(0).toFloatArray();
        }
    }

    private StableHloAst.Operation createUnaryOp(Class<? extends StableHloAst.Operation> opClass,
                                                  StableHloAst.TensorType resultType) {
        StableHloAst.Value input = new StableHloAst.Value("0", resultType);
        StableHloAst.Value result = new StableHloAst.Value("1", resultType);

        if (opClass == StableHloAst.NegateOp.class) {
            return new StableHloAst.NegateOp(input, result, resultType);
        } else if (opClass == StableHloAst.AbsOp.class) {
            return new StableHloAst.AbsOp(input, result, resultType);
        } else if (opClass == StableHloAst.ExpOp.class) {
            return new StableHloAst.ExpOp(input, result, resultType);
        } else if (opClass == StableHloAst.LogOp.class) {
            return new StableHloAst.LogOp(input, result, resultType);
        } else if (opClass == StableHloAst.SqrtOp.class) {
            return new StableHloAst.SqrtOp(input, result, resultType);
        } else if (opClass == StableHloAst.TanhOp.class) {
            return new StableHloAst.TanhOp(input, result, resultType);
        } else {
            throw new IllegalArgumentException("Unknown unary operation class: " + opClass);
        }
    }
}
