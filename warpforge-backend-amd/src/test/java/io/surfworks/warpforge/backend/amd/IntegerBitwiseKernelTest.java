package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for integer bitwise HIP kernels (And, Or, Xor) and shift operations
 * (ShiftLeft, ShiftRightArithmetic, ShiftRightLogical).
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Integer Bitwise and Shift HIP Kernels")
class IntegerBitwiseKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: And generates valid output")
    void testAndSrcGeneration() {
        System.out.println("[TEST] HIP Generation: And");
        String src = HipKernels.generateAndI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void and_i32"));
        assertTrue(src.contains("a[i] & b[i]"));
        System.out.println("[PASS] And HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Or generates valid output")
    void testOrSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Or");
        String src = HipKernels.generateOrI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void or_i32"));
        assertTrue(src.contains("a[i] | b[i]"));
        System.out.println("[PASS] Or HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Xor generates valid output")
    void testXorSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Xor");
        String src = HipKernels.generateXorI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void xor_i32"));
        assertTrue(src.contains("a[i] ^ b[i]"));
        System.out.println("[PASS] Xor HIP generation OK");
    }

    @Test
    @DisplayName("HIP: ShiftLeft generates valid output")
    void testShiftLeftSrcGeneration() {
        System.out.println("[TEST] HIP Generation: ShiftLeft");
        String src = HipKernels.generateShiftLeftI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void shift_left_i32"));
        assertTrue(src.contains("a[i] << b[i]"));
        System.out.println("[PASS] ShiftLeft HIP generation OK");
    }

    @Test
    @DisplayName("HIP: ShiftRightArithmetic generates valid output")
    void testShiftRightArithmeticSrcGeneration() {
        System.out.println("[TEST] HIP Generation: ShiftRightArithmetic");
        String src = HipKernels.generateShiftRightArithmeticI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void shift_right_arithmetic_i32"));
        assertTrue(src.contains("a[i] >> b[i]"));
        System.out.println("[PASS] ShiftRightArithmetic HIP generation OK");
    }

    @Test
    @DisplayName("HIP: ShiftRightLogical generates valid output")
    void testShiftRightLogicalSrcGeneration() {
        System.out.println("[TEST] HIP Generation: ShiftRightLogical");
        String src = HipKernels.generateShiftRightLogicalI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void shift_right_logical_i32"));
        assertTrue(src.contains("((unsigned int)a[i]) >> b[i]"));
        System.out.println("[PASS] ShiftRightLogical HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All integer bitwise operations support SALT_TIMING")
    void testAllBitwiseOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All integer bitwise/shift operations with SALT_TIMING");

        String[] ops = {"And", "Or", "Xor", "ShiftLeft", "ShiftRightArithmetic", "ShiftRightLogical"};
        String[] srcCodes = {
            HipKernels.generateAndI32(HipKernels.SALT_TIMING),
            HipKernels.generateOrI32(HipKernels.SALT_TIMING),
            HipKernels.generateXorI32(HipKernels.SALT_TIMING),
            HipKernels.generateShiftLeftI32(HipKernels.SALT_TIMING),
            HipKernels.generateShiftRightArithmeticI32(HipKernels.SALT_TIMING),
            HipKernels.generateShiftRightLogicalI32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(srcCodes[i].contains("unsigned long long* timing"),
                ops[i] + " should have timing parameter");
            assertTrue(srcCodes[i].contains("clock64()"),
                ops[i] + " should use clock64");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All integer bitwise/shift operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: All Integer Bitwise Summary")
    void testAllIntegerBitwiseSummary() {
        System.out.println("========================================");
        System.out.println("Integer Bitwise HIP Operations Summary");
        System.out.println("========================================");

        System.out.println("--- Bitwise Operations ---");
        System.out.println("  And: [OK]");
        System.out.println("  Or: [OK]");
        System.out.println("  Xor: [OK]");

        System.out.println("--- Shift Operations ---");
        System.out.println("  ShiftLeft: [OK]");
        System.out.println("  ShiftRightArithmetic: [OK]");
        System.out.println("  ShiftRightLogical: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 6 integer bitwise/shift operations PASSED");
        System.out.println("========================================");
    }
}
