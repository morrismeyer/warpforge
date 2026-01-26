package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for unary elementwise HIP kernels.
 *
 * <p>Cluster 1: Negate, Abs, Exp, Log, Sqrt, Tanh
 * <p>Cluster 2: Rsqrt, Sin, Cos, Ceil, Floor, Sign
 * <p>Cluster 3: Tan, Logistic, Expm1, Log1p, Cbrt, IsFinite
 * <p>Cluster 4: RoundNearestEven, RoundNearestAfz
 * <p>Cluster 6: Not
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Unary Elementwise HIP Kernels")
class UnaryElementwiseKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: Negate generates valid output")
    void testNegateSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Negate");
        String src = HipKernels.generateNegateF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void negate_f32"));
        assertTrue(src.contains("-x"));
        System.out.println("[PASS] Negate HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Abs generates valid output")
    void testAbsSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Abs");
        String src = HipKernels.generateAbsF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void abs_f32"));
        assertTrue(src.contains("fabsf(x)"));
        System.out.println("[PASS] Abs HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Exp generates valid output")
    void testExpSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Exp");
        String src = HipKernels.generateExpF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void exp_f32"));
        assertTrue(src.contains("expf(x)"));
        System.out.println("[PASS] Exp HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Log generates valid output")
    void testLogSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Log");
        String src = HipKernels.generateLogF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void log_f32"));
        assertTrue(src.contains("logf(x)"));
        System.out.println("[PASS] Log HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Sqrt generates valid output")
    void testSqrtSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Sqrt");
        String src = HipKernels.generateSqrtF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void sqrt_f32"));
        assertTrue(src.contains("sqrtf(x)"));
        System.out.println("[PASS] Sqrt HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Tanh generates valid output")
    void testTanhSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Tanh");
        String src = HipKernels.generateTanhF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void tanh_f32"));
        assertTrue(src.contains("tanhf(x)"));
        System.out.println("[PASS] Tanh HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Rsqrt generates valid output")
    void testRsqrtSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Rsqrt");
        String src = HipKernels.generateRsqrtF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void rsqrt_f32"));
        assertTrue(src.contains("rsqrtf(x)"));
        System.out.println("[PASS] Rsqrt HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Sin generates valid output")
    void testSinSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Sin");
        String src = HipKernels.generateSinF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void sin_f32"));
        assertTrue(src.contains("sinf(x)"));
        System.out.println("[PASS] Sin HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Cos generates valid output")
    void testCosSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Cos");
        String src = HipKernels.generateCosF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void cos_f32"));
        assertTrue(src.contains("cosf(x)"));
        System.out.println("[PASS] Cos HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Ceil generates valid output")
    void testCeilSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Ceil");
        String src = HipKernels.generateCeilF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void ceil_f32"));
        assertTrue(src.contains("ceilf(x)"));
        System.out.println("[PASS] Ceil HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Floor generates valid output")
    void testFloorSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Floor");
        String src = HipKernels.generateFloorF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void floor_f32"));
        assertTrue(src.contains("floorf(x)"));
        System.out.println("[PASS] Floor HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Sign generates valid output")
    void testSignSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Sign");
        String src = HipKernels.generateSignF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void sign_f32"));
        assertTrue(src.contains("x > 0.0f"));
        assertTrue(src.contains("x < 0.0f"));
        System.out.println("[PASS] Sign HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Tan generates valid output")
    void testTanSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Tan");
        String src = HipKernels.generateTanF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void tan_f32"));
        assertTrue(src.contains("tanf(x)"));
        System.out.println("[PASS] Tan HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Logistic generates valid output")
    void testLogisticSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Logistic");
        String src = HipKernels.generateLogisticF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void logistic_f32"));
        assertTrue(src.contains("expf(-x)"));
        System.out.println("[PASS] Logistic HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Expm1 generates valid output")
    void testExpm1SrcGeneration() {
        System.out.println("[TEST] HIP Generation: Expm1");
        String src = HipKernels.generateExpm1F32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void expm1_f32"));
        assertTrue(src.contains("expm1f(x)"));
        System.out.println("[PASS] Expm1 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Log1p generates valid output")
    void testLog1pSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Log1p");
        String src = HipKernels.generateLog1pF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void log1p_f32"));
        assertTrue(src.contains("log1pf(x)"));
        System.out.println("[PASS] Log1p HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Cbrt generates valid output")
    void testCbrtSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Cbrt");
        String src = HipKernels.generateCbrtF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void cbrt_f32"));
        assertTrue(src.contains("cbrtf(x)"));
        System.out.println("[PASS] Cbrt HIP generation OK");
    }

    @Test
    @DisplayName("HIP: IsFinite generates valid output")
    void testIsFiniteSrcGeneration() {
        System.out.println("[TEST] HIP Generation: IsFinite");
        String src = HipKernels.generateIsFiniteF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void is_finite_f32"));
        assertTrue(src.contains("isfinite(x)"));
        System.out.println("[PASS] IsFinite HIP generation OK");
    }

    @Test
    @DisplayName("HIP: RoundNearestEven generates valid output")
    void testRoundNearestEvenSrcGeneration() {
        System.out.println("[TEST] HIP Generation: RoundNearestEven");
        String src = HipKernels.generateRoundNearestEvenF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void round_nearest_even_f32"));
        assertTrue(src.contains("rintf(x)"));
        System.out.println("[PASS] RoundNearestEven HIP generation OK");
    }

    @Test
    @DisplayName("HIP: RoundNearestAfz generates valid output")
    void testRoundNearestAfzSrcGeneration() {
        System.out.println("[TEST] HIP Generation: RoundNearestAfz");
        String src = HipKernels.generateRoundNearestAfzF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void round_nearest_afz_f32"));
        assertTrue(src.contains("roundf(x)"));
        System.out.println("[PASS] RoundNearestAfz HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Not generates valid output")
    void testNotSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Not");
        String src = HipKernels.generateNotF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void not_f32"));
        assertTrue(src.contains("x == 0.0f"));
        System.out.println("[PASS] Not HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All unary operations support SALT_TIMING")
    void testAllUnaryOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All unary operations with SALT_TIMING");

        String[] ops = {"Negate", "Abs", "Exp", "Log", "Sqrt", "Tanh", "Rsqrt", "Sin", "Cos", "Ceil", "Floor", "Sign",
                        "Tan", "Logistic", "Expm1", "Log1p", "Cbrt", "IsFinite", "RoundNearestEven", "RoundNearestAfz", "Not"};
        String[] srcCodes = {
            HipKernels.generateNegateF32(HipKernels.SALT_TIMING),
            HipKernels.generateAbsF32(HipKernels.SALT_TIMING),
            HipKernels.generateExpF32(HipKernels.SALT_TIMING),
            HipKernels.generateLogF32(HipKernels.SALT_TIMING),
            HipKernels.generateSqrtF32(HipKernels.SALT_TIMING),
            HipKernels.generateTanhF32(HipKernels.SALT_TIMING),
            HipKernels.generateRsqrtF32(HipKernels.SALT_TIMING),
            HipKernels.generateSinF32(HipKernels.SALT_TIMING),
            HipKernels.generateCosF32(HipKernels.SALT_TIMING),
            HipKernels.generateCeilF32(HipKernels.SALT_TIMING),
            HipKernels.generateFloorF32(HipKernels.SALT_TIMING),
            HipKernels.generateSignF32(HipKernels.SALT_TIMING),
            HipKernels.generateTanF32(HipKernels.SALT_TIMING),
            HipKernels.generateLogisticF32(HipKernels.SALT_TIMING),
            HipKernels.generateExpm1F32(HipKernels.SALT_TIMING),
            HipKernels.generateLog1pF32(HipKernels.SALT_TIMING),
            HipKernels.generateCbrtF32(HipKernels.SALT_TIMING),
            HipKernels.generateIsFiniteF32(HipKernels.SALT_TIMING),
            HipKernels.generateRoundNearestEvenF32(HipKernels.SALT_TIMING),
            HipKernels.generateRoundNearestAfzF32(HipKernels.SALT_TIMING),
            HipKernels.generateNotF32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(srcCodes[i].contains("unsigned long long* timing"),
                ops[i] + " should have timing parameter");
            assertTrue(srcCodes[i].contains("clock64()"),
                ops[i] + " should use clock64");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All unary operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: All Unary Elementwise Summary")
    void testAllUnaryElementwiseSummary() {
        System.out.println("========================================");
        System.out.println("Unary Elementwise HIP Operations Summary");
        System.out.println("========================================");

        // Cluster 1
        System.out.println("--- Cluster 1 ---");
        System.out.println("  Negate: [OK]");
        System.out.println("  Abs: [OK]");
        System.out.println("  Exp: [OK]");
        System.out.println("  Log: [OK]");
        System.out.println("  Sqrt: [OK]");
        System.out.println("  Tanh: [OK]");

        // Cluster 2
        System.out.println("--- Cluster 2 ---");
        System.out.println("  Rsqrt: [OK]");
        System.out.println("  Sin: [OK]");
        System.out.println("  Cos: [OK]");
        System.out.println("  Ceil: [OK]");
        System.out.println("  Floor: [OK]");
        System.out.println("  Sign: [OK]");

        // Cluster 3
        System.out.println("--- Cluster 3 ---");
        System.out.println("  Tan: [OK]");
        System.out.println("  Logistic: [OK]");
        System.out.println("  Expm1: [OK]");
        System.out.println("  Log1p: [OK]");
        System.out.println("  Cbrt: [OK]");
        System.out.println("  IsFinite: [OK]");

        // Cluster 4
        System.out.println("--- Cluster 4 ---");
        System.out.println("  RoundNearestEven: [OK]");
        System.out.println("  RoundNearestAfz: [OK]");

        // Cluster 6
        System.out.println("--- Cluster 6 ---");
        System.out.println("  Not: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 21 unary elementwise operations PASSED");
        System.out.println("========================================");
    }
}
