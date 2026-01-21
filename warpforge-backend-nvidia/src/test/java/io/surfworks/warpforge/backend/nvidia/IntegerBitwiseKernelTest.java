package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for integer bitwise CUDA kernels (And, Or, Xor).
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Integer Bitwise CUDA Kernels")
class IntegerBitwiseKernelTest {

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
    @DisplayName("PTX: And generates valid output")
    void testAndPtxGeneration() {
        System.out.println("[TEST] PTX Generation: And");
        String ptx = CudaKernels.generateAndI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry and_i32"));
        assertTrue(ptx.contains("and.b32"));
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] And PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Or generates valid output")
    void testOrPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Or");
        String ptx = CudaKernels.generateOrI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry or_i32"));
        assertTrue(ptx.contains("or.b32"));
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] Or PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Xor generates valid output")
    void testXorPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Xor");
        String ptx = CudaKernels.generateXorI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry xor_i32"));
        assertTrue(ptx.contains("xor.b32"));
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] Xor PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All integer bitwise operations support SALT_TIMING")
    void testAllBitwiseOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All integer bitwise operations with SALT_TIMING");

        String[] ops = {"And", "Or", "Xor"};
        String[] ptxSources = {
            CudaKernels.generateAndI32(CudaKernels.SALT_TIMING),
            CudaKernels.generateOrI32(CudaKernels.SALT_TIMING),
            CudaKernels.generateXorI32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All integer bitwise operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: And executes correctly")
    void testAndExecution() {
        System.out.println("[TEST] CUDA Execution: And");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Test bitwise AND
        int[] a = {0b1111, 0b1010, 0b1100, 0b0000, 0xFF00, 0xFFFF, 0x1234, 0xABCD};
        int[] b = {0b0101, 0b1010, 0b0011, 0b1111, 0x00FF, 0x0F0F, 0x5678, 0x1234};
        int[] expected = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            expected[i] = a[i] & b[i];
        }

        int[] result = executeAnd(a, b);
        assertArrayEquals(expected, result);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] And execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Or executes correctly")
    void testOrExecution() {
        System.out.println("[TEST] CUDA Execution: Or");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Test bitwise OR
        int[] a = {0b1111, 0b1010, 0b1100, 0b0000, 0xFF00, 0xF0F0, 0x1234, 0x0000};
        int[] b = {0b0101, 0b0101, 0b0011, 0b1111, 0x00FF, 0x0F0F, 0x5678, 0xFFFF};
        int[] expected = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            expected[i] = a[i] | b[i];
        }

        int[] result = executeOr(a, b);
        assertArrayEquals(expected, result);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Or execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Xor executes correctly")
    void testXorExecution() {
        System.out.println("[TEST] CUDA Execution: Xor");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Test bitwise XOR
        int[] a = {0b1111, 0b1010, 0b1100, 0b0000, 0xFF00, 0xFFFF, 0x1234, 0xAAAA};
        int[] b = {0b0101, 0b1010, 0b0011, 0b1111, 0x00FF, 0xFFFF, 0x5678, 0x5555};
        int[] expected = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            expected[i] = a[i] ^ b[i];
        }

        int[] result = executeXor(a, b);
        assertArrayEquals(expected, result);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Xor execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All integer bitwise operations handle large tensors (1M elements)")
    void testLargeTensorAllOperations() {
        System.out.println("[TEST] CUDA Large Tensor: All integer bitwise operations (1M elements)");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int n = 1_000_000;
        int[] a = new int[n];
        int[] b = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        // Test And
        System.out.println("  Testing And...");
        int[] andResult = executeAnd(a, b);
        for (int i = 0; i < 10; i++) {
            int expected = a[i] & b[i];
            if (andResult[i] != expected) {
                throw new AssertionError("And mismatch at " + i + ": expected " + expected + ", got " + andResult[i]);
            }
        }
        System.out.println("  [OK] And passed");

        // Test Or
        System.out.println("  Testing Or...");
        int[] orResult = executeOr(a, b);
        for (int i = 0; i < 10; i++) {
            int expected = a[i] | b[i];
            if (orResult[i] != expected) {
                throw new AssertionError("Or mismatch at " + i + ": expected " + expected + ", got " + orResult[i]);
            }
        }
        System.out.println("  [OK] Or passed");

        // Test Xor
        System.out.println("  Testing Xor...");
        int[] xorResult = executeXor(a, b);
        for (int i = 0; i < 10; i++) {
            int expected = a[i] ^ b[i];
            if (xorResult[i] != expected) {
                throw new AssertionError("Xor mismatch at " + i + ": expected " + expected + ", got " + xorResult[i]);
            }
        }
        System.out.println("  [OK] Xor passed");

        System.out.println("[PASS] All large tensor integer bitwise operations OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Integer bitwise with SALT_TIMING produces correct results")
    void testTimingInstrumentation() {
        System.out.println("[TEST] CUDA Instrumentation: SALT_TIMING for integer bitwise");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");

        try (NvidiaBackend timedBackend = new NvidiaBackend(0, CudaKernels.SALT_TIMING)) {
            assumeTrue(timedBackend.hasCudaContext(), "CUDA context not available");

            int[] a = {0xFF, 0xF0, 0x0F, 0x00};
            int[] b = {0x0F, 0x0F, 0xF0, 0xFF};

            // Test And with timing
            System.out.println("  Testing And with timing...");
            int[] andResult = executeBinaryOp(timedBackend, a, b, StableHloAst.AndOp.class);
            for (int i = 0; i < a.length; i++) {
                int expected = a[i] & b[i];
                if (andResult[i] != expected) {
                    throw new AssertionError("And mismatch at " + i);
                }
            }
            System.out.println("  [OK] And with timing");

            // Test Or with timing
            System.out.println("  Testing Or with timing...");
            int[] orResult = executeBinaryOp(timedBackend, a, b, StableHloAst.OrOp.class);
            for (int i = 0; i < a.length; i++) {
                int expected = a[i] | b[i];
                if (orResult[i] != expected) {
                    throw new AssertionError("Or mismatch at " + i);
                }
            }
            System.out.println("  [OK] Or with timing");

            // Test Xor with timing
            System.out.println("  Testing Xor with timing...");
            int[] xorResult = executeBinaryOp(timedBackend, a, b, StableHloAst.XorOp.class);
            for (int i = 0; i < a.length; i++) {
                int expected = a[i] ^ b[i];
                if (xorResult[i] != expected) {
                    throw new AssertionError("Xor mismatch at " + i);
                }
            }
            System.out.println("  [OK] Xor with timing");
        }

        System.out.println("[PASS] SALT_TIMING instrumentation OK for integer bitwise");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Integer Bitwise Summary")
    void testAllIntegerBitwiseSummary() {
        System.out.println("========================================");
        System.out.println("Integer Bitwise Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // And
        System.out.print("  And: ");
        int[] andA = {0xFF, 0xF0};
        int[] andB = {0x0F, 0x0F};
        int[] andResult = executeAnd(andA, andB);
        assertTrue(andResult[0] == 0x0F);
        assertTrue(andResult[1] == 0x00);
        System.out.println("[OK]");

        // Or
        System.out.print("  Or: ");
        int[] orA = {0xF0, 0x00};
        int[] orB = {0x0F, 0xFF};
        int[] orResult = executeOr(orA, orB);
        assertTrue(orResult[0] == 0xFF);
        assertTrue(orResult[1] == 0xFF);
        System.out.println("[OK]");

        // Xor
        System.out.print("  Xor: ");
        int[] xorA = {0xFF, 0xAA};
        int[] xorB = {0xFF, 0x55};
        int[] xorResult = executeXor(xorA, xorB);
        assertTrue(xorResult[0] == 0x00);
        assertTrue(xorResult[1] == 0xFF);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 3 integer bitwise operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private int[] executeAnd(int[] a, int[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.AndOp.class);
    }

    private int[] executeOr(int[] a, int[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.OrOp.class);
    }

    private int[] executeXor(int[] a, int[] b) {
        return executeBinaryOp(backend, a, b, StableHloAst.XorOp.class);
    }

    private int[] executeBinaryOp(NvidiaBackend backend, int[] a, int[] b,
                                   Class<? extends StableHloAst.Operation> opClass) {
        int n = a.length;

        try (Tensor tensorA = Tensor.fromIntArray(a, n);
             Tensor tensorB = Tensor.fromIntArray(b, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.I32
            );

            StableHloAst.Operation op = createOp(opClass, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorA, tensorB));

            return results.get(0).toIntArray();
        }
    }

    private StableHloAst.Operation createOp(Class<? extends StableHloAst.Operation> opClass,
                                             StableHloAst.TensorType resultType) {
        StableHloAst.Value lhs = new StableHloAst.Value("0", resultType);
        StableHloAst.Value rhs = new StableHloAst.Value("1", resultType);
        StableHloAst.Value result = new StableHloAst.Value("2", resultType);

        if (opClass == StableHloAst.AndOp.class) {
            return new StableHloAst.AndOp(result, lhs, rhs, resultType);
        } else if (opClass == StableHloAst.OrOp.class) {
            return new StableHloAst.OrOp(result, lhs, rhs, resultType);
        } else if (opClass == StableHloAst.XorOp.class) {
            return new StableHloAst.XorOp(result, lhs, rhs, resultType);
        } else {
            throw new IllegalArgumentException("Unknown operation class: " + opClass);
        }
    }
}
