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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Hardware execution tests for integer unary HIP kernels (Popcnt, Clz) on AMD GPUs.
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Integer Unary HIP Kernels (AMD Hardware)")
class IntegerUnaryKernelTest {

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
    @DisplayName("HIP: Popcnt generates valid output")
    void testPopcntSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Popcnt");
        String src = HipKernels.generatePopcntI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void popcnt_i32"));
        assertTrue(src.contains("__popc"));
        System.out.println("[PASS] Popcnt HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Clz generates valid output")
    void testClzSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Clz");
        String src = HipKernels.generateClzI32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void clz_i32"));
        assertTrue(src.contains("__clz"));
        System.out.println("[PASS] Clz HIP generation OK");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Popcnt executes correctly on AMD GPU")
    void testPopcntExecution() {
        System.out.println("[TEST] HIP Execution: Popcnt");
        createContext();

        // Test population count (count set bits)
        int[] input = {0, 1, 2, 3, 0xFF, 0xFFFF, 0xFFFFFFFF, 0x12345678};
        int[] expected = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            expected[i] = Integer.bitCount(input[i]);
        }

        int[] result = executePopcnt(input);
        assertArrayEquals(expected, result);

        System.out.println("  Input:     " + Arrays.toString(input));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("  Expected:  " + Arrays.toString(expected));
        System.out.println("[PASS] Popcnt execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Clz executes correctly on AMD GPU")
    void testClzExecution() {
        System.out.println("[TEST] HIP Execution: Clz");
        createContext();

        // Test count leading zeros
        int[] input = {0, 1, 2, 4, 8, 0x80000000, 0x40000000, 0x7FFFFFFF};
        int[] expected = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            expected[i] = Integer.numberOfLeadingZeros(input[i]);
        }

        int[] result = executeClz(input);
        assertArrayEquals(expected, result);

        System.out.println("  Input:     " + Arrays.toString(input));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("  Expected:  " + Arrays.toString(expected));
        System.out.println("[PASS] Clz execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Popcnt edge cases")
    void testPopcntEdgeCases() {
        System.out.println("[TEST] HIP Execution: Popcnt edge cases");
        createContext();

        // All zeros
        int[] allZeros = {0, 0, 0, 0};
        int[] resultZeros = executePopcnt(allZeros);
        for (int r : resultZeros) {
            assertEquals(0, r, "Popcnt of 0 should be 0");
        }
        System.out.println("  [OK] All zeros");

        // All ones
        int[] allOnes = {-1, -1, -1, -1}; // 0xFFFFFFFF
        int[] resultOnes = executePopcnt(allOnes);
        for (int r : resultOnes) {
            assertEquals(32, r, "Popcnt of -1 should be 32");
        }
        System.out.println("  [OK] All ones");

        // Powers of 2 (single bit set)
        int[] powersOf2 = {1, 2, 4, 8, 16, 32, 64, 128};
        int[] resultPowers = executePopcnt(powersOf2);
        for (int r : resultPowers) {
            assertEquals(1, r, "Popcnt of power of 2 should be 1");
        }
        System.out.println("  [OK] Powers of 2");

        System.out.println("[PASS] Popcnt edge cases OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Clz edge cases")
    void testClzEdgeCases() {
        System.out.println("[TEST] HIP Execution: Clz edge cases");
        createContext();

        // Zero has 32 leading zeros
        int[] zeros = {0, 0, 0, 0};
        int[] resultZeros = executeClz(zeros);
        for (int r : resultZeros) {
            assertEquals(32, r, "Clz of 0 should be 32");
        }
        System.out.println("  [OK] Zero case");

        // Highest bit set (negative in signed) has 0 leading zeros
        int[] highBit = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
        int[] resultHigh = executeClz(highBit);
        for (int r : resultHigh) {
            assertEquals(0, r, "Clz of 0x80000000 should be 0");
        }
        System.out.println("  [OK] Highest bit set");

        // One has 31 leading zeros
        int[] ones = {1, 1, 1, 1};
        int[] resultOnes = executeClz(ones);
        for (int r : resultOnes) {
            assertEquals(31, r, "Clz of 1 should be 31");
        }
        System.out.println("  [OK] One case");

        System.out.println("[PASS] Clz edge cases OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: All Integer Unary Summary")
    void testAllIntegerUnarySummary() {
        System.out.println("========================================");
        System.out.println("Integer Unary HIP Operations Summary");
        System.out.println("========================================");
        createContext();

        // Popcnt
        System.out.print("  Popcnt: ");
        int[] popcntInput = {0xFF, 0x0F};
        int[] popcntResult = executePopcnt(popcntInput);
        assertEquals(8, popcntResult[0]);
        assertEquals(4, popcntResult[1]);
        System.out.println("[OK]");

        // Clz
        System.out.print("  Clz: ");
        int[] clzInput = {0x80000000, 0x00000001};
        int[] clzResult = executeClz(clzInput);
        assertEquals(0, clzResult[0]);
        assertEquals(31, clzResult[1]);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 2 integer unary operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private int[] executePopcnt(int[] input) {
        return executeIntegerUnaryOp(input, "popcnt");
    }

    private int[] executeClz(int[] input) {
        return executeIntegerUnaryOp(input, "clz");
    }

    private int[] executeIntegerUnaryOp(int[] input, String opName) {
        String source = switch (opName) {
            case "popcnt" -> HipKernels.generatePopcntI32(HipKernels.SALT_NONE);
            case "clz" -> HipKernels.generateClzI32(HipKernels.SALT_NONE);
            default -> throw new IllegalArgumentException("Unknown op: " + opName);
        };
        String functionName = opName + "_i32";

        int n = input.length;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule(opName + "_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorIn = Tensor.fromIntArray(input, n)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dIn, dOut},
                n
            );

            context.synchronize();

            int[] result = new int[n];
            try (Tensor resultTensor = Tensor.fromIntArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toIntArray();
            }

        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }
}
