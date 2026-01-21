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
 * Tests for integer unary CUDA kernels (Popcnt, Clz).
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Integer Unary CUDA Kernels")
class IntegerUnaryKernelTest {

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
    @DisplayName("PTX: Popcnt generates valid output")
    void testPopcntPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Popcnt");
        String ptx = CudaKernels.generatePopcntI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry popcnt_i32"));
        assertTrue(ptx.contains("popc.b32"));
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] Popcnt PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Clz generates valid output")
    void testClzPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Clz");
        String ptx = CudaKernels.generateClzI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry clz_i32"));
        assertTrue(ptx.contains("clz.b32"));
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] Clz PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All integer unary operations support SALT_TIMING")
    void testAllUnaryOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All integer unary operations with SALT_TIMING");

        String[] ops = {"Popcnt", "Clz"};
        String[] ptxSources = {
            CudaKernels.generatePopcntI32(CudaKernels.SALT_TIMING),
            CudaKernels.generateClzI32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All integer unary operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Popcnt executes correctly")
    void testPopcntExecution() {
        System.out.println("[TEST] CUDA Execution: Popcnt");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

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
    @Tag("nvidia")
    @DisplayName("CUDA: Clz executes correctly")
    void testClzExecution() {
        System.out.println("[TEST] CUDA Execution: Clz");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

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
    @Tag("nvidia")
    @DisplayName("CUDA: Popcnt edge cases")
    void testPopcntEdgeCases() {
        System.out.println("[TEST] CUDA Execution: Popcnt edge cases");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

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
    @Tag("nvidia")
    @DisplayName("CUDA: Clz edge cases")
    void testClzEdgeCases() {
        System.out.println("[TEST] CUDA Execution: Clz edge cases");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

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
    @Tag("nvidia")
    @DisplayName("CUDA: All Integer Unary Summary")
    void testAllIntegerUnarySummary() {
        System.out.println("========================================");
        System.out.println("Integer Unary Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

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
        return executeUnaryOp(backend, input, StableHloAst.PopcntOp.class);
    }

    private int[] executeClz(int[] input) {
        return executeUnaryOp(backend, input, StableHloAst.ClzOp.class);
    }

    private int[] executeUnaryOp(NvidiaBackend backend, int[] input,
                                  Class<? extends StableHloAst.Operation> opClass) {
        int n = input.length;

        try (Tensor tensorIn = Tensor.fromIntArray(input, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.I32
            );

            StableHloAst.Operation op = createOp(opClass, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toIntArray();
        }
    }

    private StableHloAst.Operation createOp(Class<? extends StableHloAst.Operation> opClass,
                                             StableHloAst.TensorType resultType) {
        StableHloAst.Value operand = new StableHloAst.Value("0", resultType);
        StableHloAst.Value result = new StableHloAst.Value("1", resultType);

        if (opClass == StableHloAst.PopcntOp.class) {
            return new StableHloAst.PopcntOp(result, operand, resultType);
        } else if (opClass == StableHloAst.ClzOp.class) {
            return new StableHloAst.ClzOp(result, operand, resultType);
        } else {
            throw new IllegalArgumentException("Unknown operation class: " + opClass);
        }
    }
}
