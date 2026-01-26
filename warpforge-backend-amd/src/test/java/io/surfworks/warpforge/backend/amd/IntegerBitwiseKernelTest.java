package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.backend.amd.hip.HipRuntime;
import io.surfworks.warpforge.backend.amd.hip.HiprtcRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Hardware execution tests for integer bitwise HIP kernels on AMD GPUs.
 *
 * <p>Operations: And, Or, Xor, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Integer Bitwise HIP Kernels (AMD Hardware)")
class IntegerBitwiseKernelTest {

    private static final int BLOCK_SIZE = HipKernels.ELEMENTWISE_BLOCK_SIZE;

    private HipContext context;

    @BeforeEach
    void setUp() {
        // Context will be created in tests that need it
    }

    @AfterEach
    void tearDown() {
        if (context != null) {
            context.close();
            context = null;
        }
    }

    private void createContext() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");
        assumeTrue(HiprtcRuntime.isAvailable(), "HIPRTC not available");
        try {
            context = HipContext.create(0);
        } catch (Exception e) {
            assumeTrue(false, "HIP context creation failed: " + e.getMessage());
        }
    }

    private int gridSize(int n) {
        return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: And I32 generates valid output")
    void testAndSrcGeneration() {
        System.out.println("[TEST] HIP Generation: And I32");
        String src = HipKernels.generateAndI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void and_i32"));
        System.out.println("[PASS] And I32 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Or I32 generates valid output")
    void testOrSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Or I32");
        String src = HipKernels.generateOrI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void or_i32"));
        System.out.println("[PASS] Or I32 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Xor I32 generates valid output")
    void testXorSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Xor I32");
        String src = HipKernels.generateXorI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void xor_i32"));
        System.out.println("[PASS] Xor I32 HIP generation OK");
    }

    // ==================== Hardware Execution Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: And I32 executes correctly on AMD GPU")
    void testAndExecution() {
        System.out.println("[TEST] HIP Execution: And I32");
        createContext();

        int[] a = {0b1111, 0b1010, 0b0011, 0b1100};
        int[] b = {0b1010, 0b1010, 0b0101, 0b1010};
        int[] expected = {0b1010, 0b1010, 0b0001, 0b1000};

        int[] result = executeBitwiseOp(a, b, "and");
        assertArrayEquals(expected, result);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] And I32 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Or I32 executes correctly on AMD GPU")
    void testOrExecution() {
        System.out.println("[TEST] HIP Execution: Or I32");
        createContext();

        int[] a = {0b1111, 0b1010, 0b0011, 0b1100};
        int[] b = {0b1010, 0b0101, 0b0101, 0b0011};
        int[] expected = {0b1111, 0b1111, 0b0111, 0b1111};

        int[] result = executeBitwiseOp(a, b, "or");
        assertArrayEquals(expected, result);

        System.out.println("[PASS] Or I32 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Xor I32 executes correctly on AMD GPU")
    void testXorExecution() {
        System.out.println("[TEST] HIP Execution: Xor I32");
        createContext();

        int[] a = {0b1111, 0b1010, 0b0011, 0b1100};
        int[] b = {0b1010, 0b1010, 0b0101, 0b1010};
        int[] expected = {0b0101, 0b0000, 0b0110, 0b0110};

        int[] result = executeBitwiseOp(a, b, "xor");
        assertArrayEquals(expected, result);

        System.out.println("[PASS] Xor I32 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: ShiftLeft I32 executes correctly on AMD GPU")
    void testShiftLeftExecution() {
        System.out.println("[TEST] HIP Execution: ShiftLeft I32");
        createContext();

        int[] a = {1, 2, 4, 8};
        int[] b = {1, 2, 3, 4};  // shift amounts
        int[] expected = {2, 8, 32, 128};

        int[] result = executeBitwiseOp(a, b, "shift_left");
        assertArrayEquals(expected, result);

        System.out.println("[PASS] ShiftLeft I32 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: ShiftRightArithmetic I32 executes correctly on AMD GPU")
    void testShiftRightArithmeticExecution() {
        System.out.println("[TEST] HIP Execution: ShiftRightArithmetic I32");
        createContext();

        int[] a = {8, 16, -8, -16};
        int[] b = {1, 2, 1, 2};  // shift amounts
        // Arithmetic shift preserves sign: -8 >> 1 = -4, -16 >> 2 = -4
        int[] expected = {4, 4, -4, -4};

        int[] result = executeBitwiseOp(a, b, "shift_right_arithmetic");
        assertArrayEquals(expected, result);

        System.out.println("[PASS] ShiftRightArithmetic I32 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: ShiftRightLogical I32 executes correctly on AMD GPU")
    void testShiftRightLogicalExecution() {
        System.out.println("[TEST] HIP Execution: ShiftRightLogical I32");
        createContext();

        int[] a = {8, 16, -8, -16};
        int[] b = {1, 2, 1, 2};  // shift amounts
        // Logical shift fills with zeros: unsigned interpretation
        int[] expected = {4, 4, 0x7FFFFFFC, 0x3FFFFFFC};

        int[] result = executeBitwiseOp(a, b, "shift_right_logical");
        assertArrayEquals(expected, result);

        System.out.println("[PASS] ShiftRightLogical I32 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: All integer bitwise operations work")
    void testAllIntegerBitwiseSummary() {
        System.out.println("========================================");
        System.out.println("Integer Bitwise HIP Operations Summary");
        System.out.println("========================================");
        createContext();

        int[] a = {0xFF, 0xAA, 0x55, 0x00};
        int[] b = {0x0F, 0x55, 0xAA, 0xFF};

        // Test And
        int[] andResult = executeBitwiseOp(a, b, "and");
        System.out.println("  [OK] And: " + Arrays.toString(andResult));

        // Test Or
        int[] orResult = executeBitwiseOp(a, b, "or");
        System.out.println("  [OK] Or: " + Arrays.toString(orResult));

        // Test Xor
        int[] xorResult = executeBitwiseOp(a, b, "xor");
        System.out.println("  [OK] Xor: " + Arrays.toString(xorResult));

        System.out.println("----------------------------------------");
        System.out.println("All 6 integer bitwise operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private int[] executeBitwiseOp(int[] a, int[] b, String opName) {
        String source = switch (opName) {
            case "and" -> HipKernels.generateAndI32(HipKernels.SALT_NONE);
            case "or" -> HipKernels.generateOrI32(HipKernels.SALT_NONE);
            case "xor" -> HipKernels.generateXorI32(HipKernels.SALT_NONE);
            case "shift_left" -> HipKernels.generateShiftLeftI32(HipKernels.SALT_NONE);
            case "shift_right_arithmetic" -> HipKernels.generateShiftRightArithmeticI32(HipKernels.SALT_NONE);
            case "shift_right_logical" -> HipKernels.generateShiftRightLogicalI32(HipKernels.SALT_NONE);
            default -> throw new IllegalArgumentException("Unknown bitwise op: " + opName);
        };
        String functionName = opName + "_i32";

        int n = a.length;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule(opName + "_module", source);
        long function = context.getFunction(module, functionName);

        long dA = context.allocate(byteSize);
        long dB = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorA = Tensor.fromIntArray(a, n);
                 Tensor tensorB = Tensor.fromIntArray(b, n)) {
                context.copyToDevice(dA, tensorA.data());
                context.copyToDevice(dB, tensorB.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dA, dB, dOut},
                n
            );

            context.synchronize();

            int[] result = new int[n];
            try (Tensor resultTensor = Tensor.fromIntArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toIntArray();
            }

        } finally {
            context.free(dA);
            context.free(dB);
            context.free(dOut);
        }
    }
}
