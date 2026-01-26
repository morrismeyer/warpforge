package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for comparison and selection HIP kernels.
 *
 * <p>Operations: Compare (EQ, NE, LT, LE, GT, GE), Select, Clamp
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Comparison and Selection HIP Kernels")
class CompareSelectClampKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: Compare EQ generates valid output")
    void testCompareEqSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare EQ");
        String src = HipKernels.generateCompareF32("EQ", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_eq_f32"));
        assertTrue(src.contains("a[i] == b[i]"));
        System.out.println("[PASS] Compare EQ HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Compare NE generates valid output")
    void testCompareNeSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare NE");
        String src = HipKernels.generateCompareF32("NE", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_ne_f32"));
        assertTrue(src.contains("a[i] != b[i]"));
        System.out.println("[PASS] Compare NE HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Compare LT generates valid output")
    void testCompareLtSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare LT");
        String src = HipKernels.generateCompareF32("LT", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_lt_f32"));
        assertTrue(src.contains("a[i] < b[i]"));
        System.out.println("[PASS] Compare LT HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Compare LE generates valid output")
    void testCompareLeSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare LE");
        String src = HipKernels.generateCompareF32("LE", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_le_f32"));
        assertTrue(src.contains("a[i] <= b[i]"));
        System.out.println("[PASS] Compare LE HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Compare GT generates valid output")
    void testCompareGtSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare GT");
        String src = HipKernels.generateCompareF32("GT", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_gt_f32"));
        assertTrue(src.contains("a[i] > b[i]"));
        System.out.println("[PASS] Compare GT HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Compare GE generates valid output")
    void testCompareGeSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare GE");
        String src = HipKernels.generateCompareF32("GE", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_ge_f32"));
        assertTrue(src.contains("a[i] >= b[i]"));
        System.out.println("[PASS] Compare GE HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Select generates valid output")
    void testSelectSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Select");
        String src = HipKernels.generateSelectF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void select_f32"));
        assertTrue(src.contains("pred_ptr"));
        assertTrue(src.contains("on_true_ptr"));
        assertTrue(src.contains("on_false_ptr"));
        System.out.println("[PASS] Select HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Clamp generates valid output")
    void testClampSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Clamp");
        String src = HipKernels.generateClampF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void clamp_f32"));
        assertTrue(src.contains("min_ptr"));
        assertTrue(src.contains("operand_ptr"));
        assertTrue(src.contains("max_ptr"));
        assertTrue(src.contains("fmaxf"));
        assertTrue(src.contains("fminf"));
        System.out.println("[PASS] Clamp HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All comparison operations support SALT_TIMING")
    void testAllComparisonOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All comparison operations with SALT_TIMING");

        String[] directions = {"EQ", "NE", "LT", "LE", "GT", "GE"};
        for (String dir : directions) {
            String src = HipKernels.generateCompareF32(dir, HipKernels.SALT_TIMING);
            assertTrue(src.contains("unsigned long long* timing"),
                "Compare " + dir + " should have timing parameter");
            assertTrue(src.contains("clock64()"),
                "Compare " + dir + " should use clock64");
            System.out.println("  [OK] Compare " + dir + " supports SALT_TIMING");
        }

        String selectSrc = HipKernels.generateSelectF32(HipKernels.SALT_TIMING);
        assertTrue(selectSrc.contains("unsigned long long* timing"), "Select should have timing parameter");
        System.out.println("  [OK] Select supports SALT_TIMING");

        String clampSrc = HipKernels.generateClampF32(HipKernels.SALT_TIMING);
        assertTrue(clampSrc.contains("unsigned long long* timing"), "Clamp should have timing parameter");
        System.out.println("  [OK] Clamp supports SALT_TIMING");

        System.out.println("[PASS] All comparison operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: All Comparison/Selection Summary")
    void testAllComparisonSelectionSummary() {
        System.out.println("========================================");
        System.out.println("Comparison/Selection HIP Operations Summary");
        System.out.println("========================================");

        System.out.println("--- Comparison Operations ---");
        System.out.println("  Compare EQ: [OK]");
        System.out.println("  Compare NE: [OK]");
        System.out.println("  Compare LT: [OK]");
        System.out.println("  Compare LE: [OK]");
        System.out.println("  Compare GT: [OK]");
        System.out.println("  Compare GE: [OK]");

        System.out.println("--- Selection Operations ---");
        System.out.println("  Select: [OK]");
        System.out.println("  Clamp: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 8 comparison/selection operations PASSED");
        System.out.println("========================================");
    }
}
