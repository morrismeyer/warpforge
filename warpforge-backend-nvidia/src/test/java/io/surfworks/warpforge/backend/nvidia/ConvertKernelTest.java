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

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for type conversion CUDA kernels.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Type Conversion CUDA Kernels")
class ConvertKernelTest {

    private static final float EPSILON = 1e-6f;

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
    @DisplayName("PTX: F32 to I32 generates valid output")
    void testF32toI32PtxGeneration() {
        System.out.println("[TEST] PTX Generation: F32 to I32");
        String ptx = CudaKernels.generateConvertF32toI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry convert_f32_to_i32"));
        assertTrue(ptx.contains("cvt.rzi.s32.f32"), "Should use round-toward-zero conversion");
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] F32 to I32 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: I32 to F32 generates valid output")
    void testI32toF32PtxGeneration() {
        System.out.println("[TEST] PTX Generation: I32 to F32");
        String ptx = CudaKernels.generateConvertI32toF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry convert_i32_to_f32"));
        assertTrue(ptx.contains("cvt.rn.f32.s32"), "Should use round-to-nearest conversion");
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] I32 to F32 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: I32 to I32 generates valid output")
    void testI32toI32PtxGeneration() {
        System.out.println("[TEST] PTX Generation: I32 to I32");
        String ptx = CudaKernels.generateConvertI32toI32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry convert_i32_to_i32"));
        assertTrue(ptx.contains("ld.global.s32"));
        assertTrue(ptx.contains("st.global.s32"));
        System.out.println("[PASS] I32 to I32 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All convert operations support SALT_TIMING")
    void testAllConvertOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All convert operations with SALT_TIMING");

        String[] ops = {"F32toI32", "I32toF32", "I32toI32"};
        String[] ptxSources = {
            CudaKernels.generateConvertF32toI32(CudaKernels.SALT_TIMING),
            CudaKernels.generateConvertI32toF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateConvertI32toI32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All convert operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: F32 to I32 conversion (truncation)")
    void testF32toI32Conversion() {
        System.out.println("[TEST] CUDA Execution: F32 to I32 conversion");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Test truncation behavior
        float[] input = {1.9f, 2.1f, -1.9f, -2.1f, 0.0f, 3.5f, -3.5f, 100.99f};
        int[] expected = {1, 2, -1, -2, 0, 3, -3, 100};

        int[] result = executeF32toI32(input);
        assertArrayEquals(expected, result);

        System.out.println("  Input:    " + java.util.Arrays.toString(input));
        System.out.println("  Result:   " + java.util.Arrays.toString(result));
        System.out.println("  Expected: " + java.util.Arrays.toString(expected));
        System.out.println("[PASS] F32 to I32 conversion OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: I32 to F32 conversion")
    void testI32toF32Conversion() {
        System.out.println("[TEST] CUDA Execution: I32 to F32 conversion");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int[] input = {1, 2, -1, -2, 0, 100, -100, 12345};
        float[] expected = {1.0f, 2.0f, -1.0f, -2.0f, 0.0f, 100.0f, -100.0f, 12345.0f};

        float[] result = executeI32toF32(input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input:    " + java.util.Arrays.toString(input));
        System.out.println("  Result:   " + java.util.Arrays.toString(result));
        System.out.println("[PASS] I32 to F32 conversion OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: F32 to F32 identity conversion")
    void testF32toF32Conversion() {
        System.out.println("[TEST] CUDA Execution: F32 to F32 identity conversion");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] input = {1.5f, 2.5f, -3.5f, 0.0f};
        float[] expected = {1.5f, 2.5f, -3.5f, 0.0f};

        float[] result = executeF32toF32(input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input preserved: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] F32 to F32 identity conversion OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: I32 to I32 identity conversion")
    void testI32toI32Conversion() {
        System.out.println("[TEST] CUDA Execution: I32 to I32 identity conversion");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int[] input = {1, 2, -3, 0, 12345, -67890};
        int[] expected = {1, 2, -3, 0, 12345, -67890};

        int[] result = executeI32toI32(input);
        assertArrayEquals(expected, result);

        System.out.println("  Input preserved: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] I32 to I32 identity conversion OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Convert Summary")
    void testAllConvertSummary() {
        System.out.println("========================================");
        System.out.println("Type Conversion Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // F32 to I32
        System.out.print("  F32 to I32: ");
        int[] f32toi32 = executeF32toI32(new float[]{1.9f, -1.9f});
        assertEquals(1, f32toi32[0]);
        assertEquals(-1, f32toi32[1]);
        System.out.println("[OK]");

        // I32 to F32
        System.out.print("  I32 to F32: ");
        float[] i32tof32 = executeI32toF32(new int[]{42, -42});
        assertEquals(42.0f, i32tof32[0], EPSILON);
        assertEquals(-42.0f, i32tof32[1], EPSILON);
        System.out.println("[OK]");

        // F32 to F32
        System.out.print("  F32 to F32: ");
        float[] f32tof32 = executeF32toF32(new float[]{3.14f});
        assertEquals(3.14f, f32tof32[0], EPSILON);
        System.out.println("[OK]");

        // I32 to I32
        System.out.print("  I32 to I32: ");
        int[] i32toi32 = executeI32toI32(new int[]{12345});
        assertEquals(12345, i32toi32[0]);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 4 conversion types PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private int[] executeF32toI32(float[] input) {
        int n = input.length;
        try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.I32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.ConvertOp op = new StableHloAst.ConvertOp(result, operand, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toIntArray();
        }
    }

    private float[] executeI32toF32(int[] input) {
        int n = input.length;
        try (Tensor tensorIn = Tensor.fromIntArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.I32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.ConvertOp op = new StableHloAst.ConvertOp(result, operand, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeF32toF32(float[] input) {
        int n = input.length;
        try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.ConvertOp op = new StableHloAst.ConvertOp(result, operand, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private int[] executeI32toI32(int[] input) {
        int n = input.length;
        try (Tensor tensorIn = Tensor.fromIntArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.I32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.I32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.ConvertOp op = new StableHloAst.ConvertOp(result, operand, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toIntArray();
        }
    }
}
