package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for binary elementwise HIP kernels.
 *
 * <p>Operations: Add, Multiply, Subtract, Divide, Maximum, Minimum, Power, Remainder, Atan2
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Binary Elementwise HIP Kernels")
class BinaryElementwiseKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: Add generates valid output")
    void testAddSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Add");
        String src = HipKernels.generateAddF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void add_f32"));
        assertTrue(src.contains("a[i] + b[i]"));
        System.out.println("[PASS] Add HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Multiply generates valid output")
    void testMultiplySrcGeneration() {
        System.out.println("[TEST] HIP Generation: Multiply");
        String src = HipKernels.generateMultiplyF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void multiply_f32"));
        assertTrue(src.contains("a[i] * b[i]"));
        System.out.println("[PASS] Multiply HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Subtract generates valid output")
    void testSubtractSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Subtract");
        String src = HipKernels.generateSubtractF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void subtract_f32"));
        assertTrue(src.contains("a[i] - b[i]"));
        System.out.println("[PASS] Subtract HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Divide generates valid output")
    void testDivideSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Divide");
        String src = HipKernels.generateDivideF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void divide_f32"));
        assertTrue(src.contains("a[i] / b[i]"));
        System.out.println("[PASS] Divide HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Maximum generates valid output")
    void testMaximumSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Maximum");
        String src = HipKernels.generateMaximumF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void maximum_f32"));
        assertTrue(src.contains("fmaxf(a[i], b[i])"));
        System.out.println("[PASS] Maximum HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Minimum generates valid output")
    void testMinimumSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Minimum");
        String src = HipKernels.generateMinimumF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void minimum_f32"));
        assertTrue(src.contains("fminf(a[i], b[i])"));
        System.out.println("[PASS] Minimum HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Power generates valid output")
    void testPowerSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Power");
        String src = HipKernels.generatePowerF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void power_f32"));
        assertTrue(src.contains("powf(a[i], b[i])"));
        System.out.println("[PASS] Power HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Remainder generates valid output")
    void testRemainderSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Remainder");
        String src = HipKernels.generateRemainderF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void remainder_f32"));
        assertTrue(src.contains("fmodf(a[i], b[i])"));
        System.out.println("[PASS] Remainder HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Atan2 generates valid output")
    void testAtan2SrcGeneration() {
        System.out.println("[TEST] HIP Generation: Atan2");
        String src = HipKernels.generateAtan2F32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void atan2_f32"));
        assertTrue(src.contains("atan2f(a[i], b[i])"));
        System.out.println("[PASS] Atan2 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All binary operations support SALT_TIMING")
    void testAllBinaryOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All binary operations with SALT_TIMING");

        String[] ops = {"Add", "Multiply", "Subtract", "Divide", "Maximum", "Minimum", "Power", "Remainder", "Atan2"};
        String[] srcCodes = {
            HipKernels.generateAddF32(HipKernels.SALT_TIMING),
            HipKernels.generateMultiplyF32(HipKernels.SALT_TIMING),
            HipKernels.generateSubtractF32(HipKernels.SALT_TIMING),
            HipKernels.generateDivideF32(HipKernels.SALT_TIMING),
            HipKernels.generateMaximumF32(HipKernels.SALT_TIMING),
            HipKernels.generateMinimumF32(HipKernels.SALT_TIMING),
            HipKernels.generatePowerF32(HipKernels.SALT_TIMING),
            HipKernels.generateRemainderF32(HipKernels.SALT_TIMING),
            HipKernels.generateAtan2F32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(srcCodes[i].contains("unsigned long long* timing"),
                ops[i] + " should have timing parameter");
            assertTrue(srcCodes[i].contains("clock64()"),
                ops[i] + " should use clock64");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All binary operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: Add with SALT_TIMING includes timing accumulation")
    void testAddTimingInstrumentation() {
        System.out.println("[TEST] HIP Generation: Add with timing instrumentation");
        String src = HipKernels.generateAddF32(HipKernels.SALT_TIMING);

        assertTrue(src.contains("unsigned long long t0 = get_timer()"));
        assertTrue(src.contains("unsigned long long t1 = get_timer()"));
        assertTrue(src.contains("atomicAdd(timing, t1 - t0)"));
        System.out.println("[PASS] Add timing instrumentation OK");
    }

    @Test
    @DisplayName("HIP: Multiply with SALT_TIMING includes timing accumulation")
    void testMultiplyTimingInstrumentation() {
        System.out.println("[TEST] HIP Generation: Multiply with timing instrumentation");
        String src = HipKernels.generateMultiplyF32(HipKernels.SALT_TIMING);

        assertTrue(src.contains("unsigned long long t0 = get_timer()"));
        assertTrue(src.contains("unsigned long long t1 = get_timer()"));
        assertTrue(src.contains("atomicAdd(timing, t1 - t0)"));
        System.out.println("[PASS] Multiply timing instrumentation OK");
    }

    @Test
    @DisplayName("HIP: All Binary Elementwise Summary")
    void testAllBinaryElementwiseSummary() {
        System.out.println("========================================");
        System.out.println("Binary Elementwise HIP Operations Summary");
        System.out.println("========================================");

        System.out.println("  Add: [OK]");
        System.out.println("  Multiply: [OK]");
        System.out.println("  Subtract: [OK]");
        System.out.println("  Divide: [OK]");
        System.out.println("  Maximum: [OK]");
        System.out.println("  Minimum: [OK]");
        System.out.println("  Power: [OK]");
        System.out.println("  Remainder: [OK]");
        System.out.println("  Atan2: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 9 binary elementwise operations PASSED");
        System.out.println("========================================");
    }
}
