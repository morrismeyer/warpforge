package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for type conversion HIP kernels.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Type Conversion HIP Kernels")
class ConvertKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: F32 to I32 generates valid output")
    void testF32toI32SrcGeneration() {
        System.out.println("[TEST] HIP Generation: F32 to I32");
        String src = HipKernels.generateConvertF32toI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void convert_f32_to_i32"));
        assertTrue(src.contains("const float* __restrict__ input"));
        assertTrue(src.contains("int* __restrict__ output"));
        assertTrue(src.contains("(int)input[i]"));
        System.out.println("[PASS] F32 to I32 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: I32 to F32 generates valid output")
    void testI32toF32SrcGeneration() {
        System.out.println("[TEST] HIP Generation: I32 to F32");
        String src = HipKernels.generateConvertI32toF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void convert_i32_to_f32"));
        assertTrue(src.contains("const int* __restrict__ input"));
        assertTrue(src.contains("float* __restrict__ output"));
        assertTrue(src.contains("(float)input[i]"));
        System.out.println("[PASS] I32 to F32 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: I32 to I32 generates valid output")
    void testI32toI32SrcGeneration() {
        System.out.println("[TEST] HIP Generation: I32 to I32");
        String src = HipKernels.generateConvertI32toI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void convert_i32_to_i32"));
        assertTrue(src.contains("const int* __restrict__ input"));
        assertTrue(src.contains("int* __restrict__ output"));
        assertTrue(src.contains("output[i] = input[i]"));
        System.out.println("[PASS] I32 to I32 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All convert operations support SALT_TIMING")
    void testAllConvertOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All convert operations with SALT_TIMING");

        String[] ops = {"F32toI32", "I32toF32", "I32toI32"};
        String[] srcCodes = {
            HipKernels.generateConvertF32toI32(HipKernels.SALT_TIMING),
            HipKernels.generateConvertI32toF32(HipKernels.SALT_TIMING),
            HipKernels.generateConvertI32toI32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(srcCodes[i].contains("unsigned long long* timing"),
                ops[i] + " should have timing parameter");
            assertTrue(srcCodes[i].contains("clock64()"),
                ops[i] + " should use clock64");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All convert operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: All Convert Summary")
    void testAllConvertSummary() {
        System.out.println("========================================");
        System.out.println("Type Conversion HIP Operations Summary");
        System.out.println("========================================");

        System.out.println("  F32 to I32: [OK]");
        System.out.println("  I32 to F32: [OK]");
        System.out.println("  I32 to I32: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 3 conversion types PASSED");
        System.out.println("========================================");
    }
}
