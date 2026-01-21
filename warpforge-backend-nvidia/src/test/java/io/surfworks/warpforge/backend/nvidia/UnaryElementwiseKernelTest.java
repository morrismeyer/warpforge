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
 * Tests for unary elementwise CUDA kernels.
 *
 * <p>Cluster 1: Negate, Abs, Exp, Log, Sqrt, Tanh
 * <p>Cluster 2: Rsqrt, Sin, Cos, Ceil, Floor, Sign
 * <p>Cluster 3: Tan, Logistic, Expm1, Log1p, Cbrt, IsFinite
 * <p>Cluster 4: RoundNearestEven, RoundNearestAfz
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
    @DisplayName("PTX: Rsqrt generates valid output")
    void testRsqrtPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Rsqrt");
        String ptx = CudaKernels.generateRsqrtF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry rsqrt_f32"));
        assertTrue(ptx.contains("rsqrt.approx.f32"));
        System.out.println("[PASS] Rsqrt PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Sin generates valid output")
    void testSinPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Sin");
        String ptx = CudaKernels.generateSinF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry sin_f32"));
        assertTrue(ptx.contains("sin.approx.f32"));
        System.out.println("[PASS] Sin PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Cos generates valid output")
    void testCosPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Cos");
        String ptx = CudaKernels.generateCosF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry cos_f32"));
        assertTrue(ptx.contains("cos.approx.f32"));
        System.out.println("[PASS] Cos PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Ceil generates valid output")
    void testCeilPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Ceil");
        String ptx = CudaKernels.generateCeilF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry ceil_f32"));
        assertTrue(ptx.contains("cvt.rpi.f32.f32"));
        System.out.println("[PASS] Ceil PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Floor generates valid output")
    void testFloorPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Floor");
        String ptx = CudaKernels.generateFloorF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry floor_f32"));
        assertTrue(ptx.contains("cvt.rmi.f32.f32"));
        System.out.println("[PASS] Floor PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Sign generates valid output")
    void testSignPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Sign");
        String ptx = CudaKernels.generateSignF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry sign_f32"));
        assertTrue(ptx.contains("setp.gt.f32"));
        assertTrue(ptx.contains("setp.lt.f32"));
        System.out.println("[PASS] Sign PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Tan generates valid output")
    void testTanPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Tan");
        String ptx = CudaKernels.generateTanF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry tan_f32"));
        assertTrue(ptx.contains("sin.approx.f32"));
        assertTrue(ptx.contains("cos.approx.f32"));
        assertTrue(ptx.contains("div.approx.f32"));
        System.out.println("[PASS] Tan PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Logistic generates valid output")
    void testLogisticPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Logistic");
        String ptx = CudaKernels.generateLogisticF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry logistic_f32"));
        assertTrue(ptx.contains("ex2.approx.f32"));
        assertTrue(ptx.contains("rcp.approx.f32"));
        System.out.println("[PASS] Logistic PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Expm1 generates valid output")
    void testExpm1PtxGeneration() {
        System.out.println("[TEST] PTX Generation: Expm1");
        String ptx = CudaKernels.generateExpm1F32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry expm1_f32"));
        assertTrue(ptx.contains("ex2.approx.f32"));
        assertTrue(ptx.contains("sub.f32"));
        System.out.println("[PASS] Expm1 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Log1p generates valid output")
    void testLog1pPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Log1p");
        String ptx = CudaKernels.generateLog1pF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry log1p_f32"));
        assertTrue(ptx.contains("lg2.approx.f32"));
        assertTrue(ptx.contains("add.f32"));
        System.out.println("[PASS] Log1p PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Cbrt generates valid output")
    void testCbrtPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Cbrt");
        String ptx = CudaKernels.generateCbrtF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry cbrt_f32"));
        assertTrue(ptx.contains("lg2.approx.f32"));
        assertTrue(ptx.contains("ex2.approx.f32"));
        assertTrue(ptx.contains("copysign.f32"));
        System.out.println("[PASS] Cbrt PTX generation OK");
    }

    @Test
    @DisplayName("PTX: IsFinite generates valid output")
    void testIsFinitePtxGeneration() {
        System.out.println("[TEST] PTX Generation: IsFinite");
        String ptx = CudaKernels.generateIsFiniteF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry is_finite_f32"));
        assertTrue(ptx.contains("testp.finite.f32"));
        assertTrue(ptx.contains("selp.f32"));
        System.out.println("[PASS] IsFinite PTX generation OK");
    }

    @Test
    @DisplayName("PTX: RoundNearestEven generates valid output")
    void testRoundNearestEvenPtxGeneration() {
        System.out.println("[TEST] PTX Generation: RoundNearestEven");
        String ptx = CudaKernels.generateRoundNearestEvenF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry round_nearest_even_f32"));
        assertTrue(ptx.contains("cvt.rni.f32.f32"));
        System.out.println("[PASS] RoundNearestEven PTX generation OK");
    }

    @Test
    @DisplayName("PTX: RoundNearestAfz generates valid output")
    void testRoundNearestAfzPtxGeneration() {
        System.out.println("[TEST] PTX Generation: RoundNearestAfz");
        String ptx = CudaKernels.generateRoundNearestAfzF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry round_nearest_afz_f32"));
        assertTrue(ptx.contains("abs.f32"));
        assertTrue(ptx.contains("cvt.rmi.f32.f32")); // floor
        assertTrue(ptx.contains("copysign.f32"));
        System.out.println("[PASS] RoundNearestAfz PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All unary operations support SALT_TIMING")
    void testAllUnaryOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All unary operations with SALT_TIMING");

        String[] ops = {"Negate", "Abs", "Exp", "Log", "Sqrt", "Tanh", "Rsqrt", "Sin", "Cos", "Ceil", "Floor", "Sign",
                        "Tan", "Logistic", "Expm1", "Log1p", "Cbrt", "IsFinite", "RoundNearestEven", "RoundNearestAfz"};
        String[] ptxSources = {
            CudaKernels.generateNegateF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateAbsF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateExpF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateLogF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateSqrtF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateTanhF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateRsqrtF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateSinF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateCosF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateCeilF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateFloorF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateSignF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateTanF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateLogisticF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateExpm1F32(CudaKernels.SALT_TIMING),
            CudaKernels.generateLog1pF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateCbrtF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateIsFiniteF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateRoundNearestEvenF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateRoundNearestAfzF32(CudaKernels.SALT_TIMING)
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

    // ==================== Cluster 2: Rsqrt, Sin, Cos, Ceil, Floor, Sign ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Rsqrt executes correctly")
    void testRsqrtExecution() {
        System.out.println("[TEST] CUDA Execution: Rsqrt");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 100.0f, 0.25f, 2.0f};

        float[] result = executeRsqrt(input);

        assertEquals(1.0f, result[0], 1e-3f, "rsqrt(1)");
        assertEquals(0.5f, result[1], 1e-3f, "rsqrt(4)");
        assertEquals(1.0f / 3.0f, result[2], 1e-3f, "rsqrt(9)");
        assertEquals(0.25f, result[3], 1e-3f, "rsqrt(16)");
        assertEquals(0.2f, result[4], 1e-3f, "rsqrt(25)");
        assertEquals(0.1f, result[5], 1e-3f, "rsqrt(100)");
        assertEquals(2.0f, result[6], 1e-3f, "rsqrt(0.25)");
        assertEquals((float) (1.0 / Math.sqrt(2)), result[7], 1e-3f, "rsqrt(2)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Rsqrt execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Sin executes correctly")
    void testSinExecution() {
        System.out.println("[TEST] CUDA Execution: Sin");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float pi = (float) Math.PI;
        float[] input = {0.0f, pi / 6, pi / 4, pi / 3, pi / 2, pi, -pi / 2};

        float[] result = executeSin(input);

        assertEquals(0.0f, result[0], 1e-3f, "sin(0)");
        assertEquals(0.5f, result[1], 1e-3f, "sin(pi/6)");
        assertEquals((float) Math.sin(pi / 4), result[2], 1e-3f, "sin(pi/4)");
        assertEquals((float) Math.sin(pi / 3), result[3], 1e-3f, "sin(pi/3)");
        assertEquals(1.0f, result[4], 1e-3f, "sin(pi/2)");
        assertEquals(0.0f, result[5], 1e-3f, "sin(pi)");
        assertEquals(-1.0f, result[6], 1e-3f, "sin(-pi/2)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Sin execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Cos executes correctly")
    void testCosExecution() {
        System.out.println("[TEST] CUDA Execution: Cos");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float pi = (float) Math.PI;
        float[] input = {0.0f, pi / 6, pi / 4, pi / 3, pi / 2, pi, -pi};

        float[] result = executeCos(input);

        assertEquals(1.0f, result[0], 1e-3f, "cos(0)");
        assertEquals((float) Math.cos(pi / 6), result[1], 1e-3f, "cos(pi/6)");
        assertEquals((float) Math.cos(pi / 4), result[2], 1e-3f, "cos(pi/4)");
        assertEquals(0.5f, result[3], 1e-3f, "cos(pi/3)");
        assertEquals(0.0f, result[4], 1e-3f, "cos(pi/2)");
        assertEquals(-1.0f, result[5], 1e-3f, "cos(pi)");
        assertEquals(-1.0f, result[6], 1e-3f, "cos(-pi)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Cos execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Ceil executes correctly")
    void testCeilExecution() {
        System.out.println("[TEST] CUDA Execution: Ceil");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.1f, 1.9f, -1.1f, -1.9f, 0.0f, 2.0f, 0.5f, -0.5f};
        float[] expected = {2.0f, 2.0f, -1.0f, -1.0f, 0.0f, 2.0f, 1.0f, 0.0f};

        float[] result = executeCeil(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "ceil(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Ceil execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Floor executes correctly")
    void testFloorExecution() {
        System.out.println("[TEST] CUDA Execution: Floor");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.1f, 1.9f, -1.1f, -1.9f, 0.0f, 2.0f, 0.5f, -0.5f};
        float[] expected = {1.0f, 1.0f, -2.0f, -2.0f, 0.0f, 2.0f, 0.0f, -1.0f};

        float[] result = executeFloor(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "floor(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Floor execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Sign executes correctly")
    void testSignExecution() {
        System.out.println("[TEST] CUDA Execution: Sign");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {5.0f, -3.0f, 0.0f, 100.0f, -0.001f, 0.001f, Float.MAX_VALUE, -Float.MAX_VALUE};
        float[] expected = {1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f};

        float[] result = executeSign(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "sign(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Sign execution OK");
    }

    // ==================== Cluster 3: Tan, Logistic, Expm1, Log1p, Cbrt, IsFinite ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Tan executes correctly")
    void testTanExecution() {
        System.out.println("[TEST] CUDA Execution: Tan");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float pi = (float) Math.PI;
        float[] input = {0.0f, pi / 6, pi / 4, -pi / 4, pi / 3};

        float[] result = executeTan(input);

        assertEquals(0.0f, result[0], 1e-3f, "tan(0)");
        assertEquals((float) Math.tan(pi / 6), result[1], 1e-2f, "tan(pi/6)");
        assertEquals(1.0f, result[2], 1e-2f, "tan(pi/4)");
        assertEquals(-1.0f, result[3], 1e-2f, "tan(-pi/4)");
        assertEquals((float) Math.tan(pi / 3), result[4], 1e-1f, "tan(pi/3)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Tan execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Logistic (sigmoid) executes correctly")
    void testLogisticExecution() {
        System.out.println("[TEST] CUDA Execution: Logistic");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 10.0f, -10.0f};

        float[] result = executeLogistic(input);

        assertEquals(0.5f, result[0], 1e-3f, "sigmoid(0)");
        assertEquals(1.0f / (1.0f + (float) Math.exp(-1)), result[1], 1e-2f, "sigmoid(1)");
        assertEquals(1.0f / (1.0f + (float) Math.exp(1)), result[2], 1e-2f, "sigmoid(-1)");
        assertEquals(1.0f / (1.0f + (float) Math.exp(-2)), result[3], 1e-2f, "sigmoid(2)");
        assertEquals(1.0f / (1.0f + (float) Math.exp(2)), result[4], 1e-2f, "sigmoid(-2)");
        assertTrue(result[5] > 0.99f, "sigmoid(10) should be close to 1");
        assertTrue(result[6] < 0.01f, "sigmoid(-10) should be close to 0");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Logistic execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Expm1 executes correctly")
    void testExpm1Execution() {
        System.out.println("[TEST] CUDA Execution: Expm1");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {0.0f, 1.0f, -1.0f, 0.001f, -0.001f, 2.0f};

        float[] result = executeExpm1(input);

        assertEquals(0.0f, result[0], 1e-4f, "expm1(0)");
        assertEquals((float) Math.expm1(1), result[1], 1e-2f, "expm1(1)");
        assertEquals((float) Math.expm1(-1), result[2], 1e-2f, "expm1(-1)");
        assertEquals((float) Math.expm1(0.001), result[3], 1e-4f, "expm1(0.001)");
        assertEquals((float) Math.expm1(-0.001), result[4], 1e-4f, "expm1(-0.001)");
        assertEquals((float) Math.expm1(2), result[5], 1e-1f, "expm1(2)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Expm1 execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Log1p executes correctly")
    void testLog1pExecution() {
        System.out.println("[TEST] CUDA Execution: Log1p");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {0.0f, 1.0f, 0.001f, (float) (Math.E - 1), 9.0f};

        float[] result = executeLog1p(input);

        assertEquals(0.0f, result[0], 1e-4f, "log1p(0)");
        assertEquals((float) Math.log(2), result[1], 1e-2f, "log1p(1)");
        assertEquals((float) Math.log1p(0.001), result[2], 1e-4f, "log1p(0.001)");
        assertEquals(1.0f, result[3], 1e-2f, "log1p(e-1)");
        assertEquals((float) Math.log(10), result[4], 1e-2f, "log1p(9)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Log1p execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Cbrt executes correctly")
    void testCbrtExecution() {
        System.out.println("[TEST] CUDA Execution: Cbrt");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {0.0f, 1.0f, 8.0f, 27.0f, -8.0f, -27.0f, 125.0f, 1000.0f};

        float[] result = executeCbrt(input);

        assertEquals(0.0f, result[0], 1e-4f, "cbrt(0)");
        assertEquals(1.0f, result[1], 1e-3f, "cbrt(1)");
        assertEquals(2.0f, result[2], 1e-2f, "cbrt(8)");
        assertEquals(3.0f, result[3], 1e-2f, "cbrt(27)");
        assertEquals(-2.0f, result[4], 1e-2f, "cbrt(-8)");
        assertEquals(-3.0f, result[5], 1e-2f, "cbrt(-27)");
        assertEquals(5.0f, result[6], 1e-2f, "cbrt(125)");
        assertEquals(10.0f, result[7], 1e-1f, "cbrt(1000)");

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Cbrt execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: IsFinite executes correctly")
    void testIsFiniteExecution() {
        System.out.println("[TEST] CUDA Execution: IsFinite");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.0f, -1.0f, 0.0f, Float.MAX_VALUE, Float.MIN_VALUE,
                         Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        float[] expected = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};

        float[] result = executeIsFinite(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "isFinite(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] IsFinite execution OK");
    }

    // ==================== Cluster 4: RoundNearestEven, RoundNearestAfz ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: RoundNearestEven executes correctly")
    void testRoundNearestEvenExecution() {
        System.out.println("[TEST] CUDA Execution: RoundNearestEven");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Round nearest even (banker's rounding): 0.5 rounds to even, 1.5 rounds to 2, 2.5 rounds to 2
        float[] input = {1.5f, 2.5f, 3.5f, 4.5f, -1.5f, -2.5f, 1.4f, 1.6f};
        // Expected: 2, 2, 4, 4, -2, -2, 1, 2 (ties round to nearest even)
        float[] expected = {2.0f, 2.0f, 4.0f, 4.0f, -2.0f, -2.0f, 1.0f, 2.0f};

        float[] result = executeRoundNearestEven(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "roundNearestEven(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] RoundNearestEven execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: RoundNearestAfz executes correctly")
    void testRoundNearestAfzExecution() {
        System.out.println("[TEST] CUDA Execution: RoundNearestAfz");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Round nearest away from zero: 0.5 rounds away from zero
        float[] input = {1.5f, 2.5f, 3.5f, 4.5f, -1.5f, -2.5f, 1.4f, 1.6f};
        // Expected: 2, 3, 4, 5, -2, -3, 1, 2 (ties round away from zero)
        float[] expected = {2.0f, 3.0f, 4.0f, 5.0f, -2.0f, -3.0f, 1.0f, 2.0f};

        float[] result = executeRoundNearestAfz(input);

        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 1e-5f, "roundNearestAfz(" + input[i] + ")");
        }

        System.out.println("  Input:  " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] RoundNearestAfz execution OK");
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

        // Cluster 1
        System.out.println("--- Cluster 1 ---");

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

        // Cluster 2
        System.out.println("--- Cluster 2 ---");

        // Rsqrt
        System.out.print("  Rsqrt: ");
        float[] rsqrtInput = {4.0f, 16.0f};
        float[] rsqrtResult = executeRsqrt(rsqrtInput);
        assertTrue(Math.abs(rsqrtResult[0] - 0.5f) < 1e-3f);
        assertTrue(Math.abs(rsqrtResult[1] - 0.25f) < 1e-3f);
        System.out.println("[OK]");

        // Sin
        System.out.print("  Sin: ");
        float[] sinInput = {0.0f, (float) (Math.PI / 2)};
        float[] sinResult = executeSin(sinInput);
        assertTrue(Math.abs(sinResult[0]) < 1e-3f);
        assertTrue(Math.abs(sinResult[1] - 1.0f) < 1e-3f);
        System.out.println("[OK]");

        // Cos
        System.out.print("  Cos: ");
        float[] cosInput = {0.0f, (float) Math.PI};
        float[] cosResult = executeCos(cosInput);
        assertTrue(Math.abs(cosResult[0] - 1.0f) < 1e-3f);
        assertTrue(Math.abs(cosResult[1] - (-1.0f)) < 1e-3f);
        System.out.println("[OK]");

        // Ceil
        System.out.print("  Ceil: ");
        float[] ceilInput = {1.1f, -1.9f};
        float[] ceilResult = executeCeil(ceilInput);
        assertTrue(Math.abs(ceilResult[0] - 2.0f) < 1e-5f);
        assertTrue(Math.abs(ceilResult[1] - (-1.0f)) < 1e-5f);
        System.out.println("[OK]");

        // Floor
        System.out.print("  Floor: ");
        float[] floorInput = {1.9f, -1.1f};
        float[] floorResult = executeFloor(floorInput);
        assertTrue(Math.abs(floorResult[0] - 1.0f) < 1e-5f);
        assertTrue(Math.abs(floorResult[1] - (-2.0f)) < 1e-5f);
        System.out.println("[OK]");

        // Sign
        System.out.print("  Sign: ");
        float[] signInput = {5.0f, -3.0f, 0.0f};
        float[] signResult = executeSign(signInput);
        assertTrue(Math.abs(signResult[0] - 1.0f) < 1e-5f);
        assertTrue(Math.abs(signResult[1] - (-1.0f)) < 1e-5f);
        assertTrue(Math.abs(signResult[2]) < 1e-5f);
        System.out.println("[OK]");

        // Cluster 3
        System.out.println("--- Cluster 3 ---");

        // Tan
        System.out.print("  Tan: ");
        float[] tanInput = {0.0f, (float) (Math.PI / 4)};
        float[] tanResult = executeTan(tanInput);
        assertTrue(Math.abs(tanResult[0]) < 1e-3f);
        assertTrue(Math.abs(tanResult[1] - 1.0f) < 1e-2f);
        System.out.println("[OK]");

        // Logistic
        System.out.print("  Logistic: ");
        float[] logisticInput = {0.0f, 10.0f};
        float[] logisticResult = executeLogistic(logisticInput);
        assertTrue(Math.abs(logisticResult[0] - 0.5f) < 1e-3f);
        assertTrue(logisticResult[1] > 0.99f);
        System.out.println("[OK]");

        // Expm1
        System.out.print("  Expm1: ");
        float[] expm1Input = {0.0f, 1.0f};
        float[] expm1Result = executeExpm1(expm1Input);
        assertTrue(Math.abs(expm1Result[0]) < 1e-4f);
        assertTrue(Math.abs(expm1Result[1] - (float) Math.expm1(1)) < 1e-2f);
        System.out.println("[OK]");

        // Log1p
        System.out.print("  Log1p: ");
        float[] log1pInput = {0.0f, 1.0f};
        float[] log1pResult = executeLog1p(log1pInput);
        assertTrue(Math.abs(log1pResult[0]) < 1e-4f);
        assertTrue(Math.abs(log1pResult[1] - (float) Math.log(2)) < 1e-2f);
        System.out.println("[OK]");

        // Cbrt
        System.out.print("  Cbrt: ");
        float[] cbrtInput = {8.0f, -27.0f};
        float[] cbrtResult = executeCbrt(cbrtInput);
        assertTrue(Math.abs(cbrtResult[0] - 2.0f) < 1e-2f);
        assertTrue(Math.abs(cbrtResult[1] - (-3.0f)) < 1e-2f);
        System.out.println("[OK]");

        // IsFinite
        System.out.print("  IsFinite: ");
        float[] isFiniteInput = {1.0f, Float.POSITIVE_INFINITY, Float.NaN};
        float[] isFiniteResult = executeIsFinite(isFiniteInput);
        assertTrue(Math.abs(isFiniteResult[0] - 1.0f) < 1e-5f);
        assertTrue(Math.abs(isFiniteResult[1]) < 1e-5f);
        assertTrue(Math.abs(isFiniteResult[2]) < 1e-5f);
        System.out.println("[OK]");

        // Cluster 4
        System.out.println("--- Cluster 4 ---");

        // RoundNearestEven
        System.out.print("  RoundNearestEven: ");
        float[] rneInput = {1.5f, 2.5f, 1.4f};
        float[] rneResult = executeRoundNearestEven(rneInput);
        assertTrue(Math.abs(rneResult[0] - 2.0f) < 1e-5f);
        assertTrue(Math.abs(rneResult[1] - 2.0f) < 1e-5f); // ties to even
        assertTrue(Math.abs(rneResult[2] - 1.0f) < 1e-5f);
        System.out.println("[OK]");

        // RoundNearestAfz
        System.out.print("  RoundNearestAfz: ");
        float[] rafzInput = {1.5f, 2.5f, -1.5f};
        float[] rafzResult = executeRoundNearestAfz(rafzInput);
        assertTrue(Math.abs(rafzResult[0] - 2.0f) < 1e-5f);
        assertTrue(Math.abs(rafzResult[1] - 3.0f) < 1e-5f); // ties away from zero
        assertTrue(Math.abs(rafzResult[2] - (-2.0f)) < 1e-5f);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 20 unary elementwise operations PASSED");
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

    private float[] executeRsqrt(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.RsqrtOp.class);
    }

    private float[] executeSin(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.SinOp.class);
    }

    private float[] executeCos(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.CosOp.class);
    }

    private float[] executeCeil(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.CeilOp.class);
    }

    private float[] executeFloor(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.FloorOp.class);
    }

    private float[] executeSign(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.SignOp.class);
    }

    private float[] executeTan(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.TanOp.class);
    }

    private float[] executeLogistic(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.LogisticOp.class);
    }

    private float[] executeExpm1(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.Expm1Op.class);
    }

    private float[] executeLog1p(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.Log1pOp.class);
    }

    private float[] executeCbrt(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.CbrtOp.class);
    }

    private float[] executeIsFinite(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.IsFiniteOp.class);
    }

    private float[] executeRoundNearestEven(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.RoundNearestEvenOp.class);
    }

    private float[] executeRoundNearestAfz(float[] input) {
        return executeUnaryOp(backend, input, StableHloAst.RoundNearestAfzOp.class);
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

        // Cluster 1
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
        // Cluster 2
        } else if (opClass == StableHloAst.RsqrtOp.class) {
            return new StableHloAst.RsqrtOp(input, result, resultType);
        } else if (opClass == StableHloAst.SinOp.class) {
            return new StableHloAst.SinOp(input, result, resultType);
        } else if (opClass == StableHloAst.CosOp.class) {
            return new StableHloAst.CosOp(input, result, resultType);
        } else if (opClass == StableHloAst.CeilOp.class) {
            return new StableHloAst.CeilOp(input, result, resultType);
        } else if (opClass == StableHloAst.FloorOp.class) {
            return new StableHloAst.FloorOp(input, result, resultType);
        } else if (opClass == StableHloAst.SignOp.class) {
            return new StableHloAst.SignOp(input, result, resultType);
        // Cluster 3
        } else if (opClass == StableHloAst.TanOp.class) {
            return new StableHloAst.TanOp(input, result, resultType);
        } else if (opClass == StableHloAst.LogisticOp.class) {
            return new StableHloAst.LogisticOp(input, result, resultType);
        } else if (opClass == StableHloAst.Expm1Op.class) {
            return new StableHloAst.Expm1Op(input, result, resultType);
        } else if (opClass == StableHloAst.Log1pOp.class) {
            return new StableHloAst.Log1pOp(input, result, resultType);
        } else if (opClass == StableHloAst.CbrtOp.class) {
            return new StableHloAst.CbrtOp(input, result, resultType);
        } else if (opClass == StableHloAst.IsFiniteOp.class) {
            return new StableHloAst.IsFiniteOp(input, result, resultType);
        // Cluster 4
        } else if (opClass == StableHloAst.RoundNearestEvenOp.class) {
            return new StableHloAst.RoundNearestEvenOp(input, result, resultType);
        } else if (opClass == StableHloAst.RoundNearestAfzOp.class) {
            return new StableHloAst.RoundNearestAfzOp(input, result, resultType);
        } else {
            throw new IllegalArgumentException("Unknown unary operation class: " + opClass);
        }
    }
}
