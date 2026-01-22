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
 * Tests for Iota, Pad, and Reverse CUDA kernels.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Iota, Pad, and Reverse CUDA Kernels")
class IotaPadReverseKernelTest {

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
    @DisplayName("PTX: Iota1D generates valid output")
    void testIota1DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Iota1D");
        String ptx = CudaKernels.generateIota1DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry iota_1d_f32"));
        assertTrue(ptx.contains("cvt.rn.f32.s32"));  // int to float conversion
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Iota1D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Iota2D dim0 generates valid output")
    void testIota2DDim0PtxGeneration() {
        System.out.println("[TEST] PTX Generation: Iota2D dim0");
        String ptx = CudaKernels.generateIota2DDim0F32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry iota_2d_dim0_f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Iota2D dim0 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Iota2D dim1 generates valid output")
    void testIota2DDim1PtxGeneration() {
        System.out.println("[TEST] PTX Generation: Iota2D dim1");
        String ptx = CudaKernels.generateIota2DDim1F32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry iota_2d_dim1_f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Iota2D dim1 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Pad1D generates valid output")
    void testPad1DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Pad1D");
        String ptx = CudaKernels.generatePad1DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry pad_1d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Pad1D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Pad2D generates valid output")
    void testPad2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Pad2D");
        String ptx = CudaKernels.generatePad2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry pad_2d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Pad2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reverse1D generates valid output")
    void testReverse1DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reverse1D");
        String ptx = CudaKernels.generateReverse1DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reverse_1d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Reverse1D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reverse2D dim0 generates valid output")
    void testReverse2DDim0PtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reverse2D dim0");
        String ptx = CudaKernels.generateReverse2DDim0F32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reverse_2d_dim0_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Reverse2D dim0 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Reverse2D dim1 generates valid output")
    void testReverse2DDim1PtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reverse2D dim1");
        String ptx = CudaKernels.generateReverse2DDim1F32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reverse_2d_dim1_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Reverse2D dim1 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All iota/pad/reverse operations support SALT_TIMING")
    void testAllOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All operations with SALT_TIMING");

        String[] ops = {"Iota1D", "Iota2D_dim0", "Iota2D_dim1", "Pad1D", "Pad2D",
                        "Reverse1D", "Reverse2D_dim0", "Reverse2D_dim1"};
        String[] ptxSources = {
            CudaKernels.generateIota1DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateIota2DDim0F32(CudaKernels.SALT_TIMING),
            CudaKernels.generateIota2DDim1F32(CudaKernels.SALT_TIMING),
            CudaKernels.generatePad1DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generatePad2DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateReverse1DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateReverse2DDim0F32(CudaKernels.SALT_TIMING),
            CudaKernels.generateReverse2DDim1F32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Iota 1D generates sequence")
    void testIota1D() {
        System.out.println("[TEST] CUDA Execution: Iota 1D");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Generate [0, 1, 2, 3, 4]
        float[] expected = {0, 1, 2, 3, 4};

        float[] result = executeIota1D(5);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Iota 1D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Iota 2D dim0 generates row indices")
    void testIota2DDim0() {
        System.out.println("[TEST] CUDA Execution: Iota 2D dim0");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Generate 3x4 matrix with row indices:
        // [[0, 0, 0, 0],
        //  [1, 1, 1, 1],
        //  [2, 2, 2, 2]]
        float[] expected = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

        float[] result = executeIota2D(3, 4, 0);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Result (3x4, dim0): " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Iota 2D dim0 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Iota 2D dim1 generates column indices")
    void testIota2DDim1() {
        System.out.println("[TEST] CUDA Execution: Iota 2D dim1");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Generate 3x4 matrix with column indices:
        // [[0, 1, 2, 3],
        //  [0, 1, 2, 3],
        //  [0, 1, 2, 3]]
        float[] expected = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

        float[] result = executeIota2D(3, 4, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Result (3x4, dim1): " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Iota 2D dim1 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Pad 1D with zeros")
    void testPad1D() {
        System.out.println("[TEST] CUDA Execution: Pad 1D");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Pad [1, 2, 3] with 2 zeros on left and 1 zero on right
        // -> [0, 0, 1, 2, 3, 0]
        float[] input = {1, 2, 3};
        float[] expected = {0, 0, 1, 2, 3, 0};

        float[] result = executePad1D(input, 0f, 2, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Pad 1D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Pad 1D with custom value")
    void testPad1DCustomValue() {
        System.out.println("[TEST] CUDA Execution: Pad 1D with custom value");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Pad [1, 2, 3] with -1 padding, 1 on left and 2 on right
        // -> [-1, 1, 2, 3, -1, -1]
        float[] input = {1, 2, 3};
        float[] expected = {-1, 1, 2, 3, -1, -1};

        float[] result = executePad1D(input, -1f, 1, 2);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Pad 1D with custom value OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Pad 2D")
    void testPad2D() {
        System.out.println("[TEST] CUDA Execution: Pad 2D");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[1, 2], [3, 4]]
        // Pad with 1 row top, 0 rows bottom, 1 col left, 1 col right
        // Output: [[0, 1, 2, 0],
        //          [0, 3, 4, 0]]
        // (with 1 row top pad: [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]])
        float[] input = {1, 2, 3, 4};
        int[] inputShape = {2, 2};
        float[] expected = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0};

        float[] result = executePad2D(input, inputShape, 0f, 1, 0, 1, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input (2x2): " + java.util.Arrays.toString(input));
        System.out.println("  Result (3x4): " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Pad 2D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reverse 1D")
    void testReverse1D() {
        System.out.println("[TEST] CUDA Execution: Reverse 1D");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Reverse [1, 2, 3, 4, 5] -> [5, 4, 3, 2, 1]
        float[] input = {1, 2, 3, 4, 5};
        float[] expected = {5, 4, 3, 2, 1};

        float[] result = executeReverse1D(input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Reverse 1D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reverse 2D dim0")
    void testReverse2DDim0() {
        System.out.println("[TEST] CUDA Execution: Reverse 2D dim0");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[1, 2], [3, 4], [5, 6]]
        // Reverse dim0 -> [[5, 6], [3, 4], [1, 2]]
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] shape = {3, 2};
        float[] expected = {5, 6, 3, 4, 1, 2};

        float[] result = executeReverse2D(input, shape, List.of(0L));
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input (3x2): " + java.util.Arrays.toString(input));
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Reverse 2D dim0 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reverse 2D dim1")
    void testReverse2DDim1() {
        System.out.println("[TEST] CUDA Execution: Reverse 2D dim1");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[1, 2, 3], [4, 5, 6]]
        // Reverse dim1 -> [[3, 2, 1], [6, 5, 4]]
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] shape = {2, 3};
        float[] expected = {3, 2, 1, 6, 5, 4};

        float[] result = executeReverse2D(input, shape, List.of(1L));
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input (2x3): " + java.util.Arrays.toString(input));
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Reverse 2D dim1 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Iota/Pad/Reverse Summary")
    void testAllIotaPadReverseSummary() {
        System.out.println("========================================");
        System.out.println("Iota, Pad, and Reverse Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Iota 1D
        System.out.print("  Iota 1D: ");
        float[] iota1d = executeIota1D(5);
        assertEquals(0f, iota1d[0], EPSILON);
        assertEquals(4f, iota1d[4], EPSILON);
        System.out.println("[OK]");

        // Iota 2D
        System.out.print("  Iota 2D: ");
        float[] iota2d = executeIota2D(2, 3, 0);
        assertEquals(0f, iota2d[0], EPSILON);
        assertEquals(1f, iota2d[3], EPSILON);
        System.out.println("[OK]");

        // Pad 1D
        System.out.print("  Pad 1D: ");
        float[] pad1d = executePad1D(new float[]{1, 2}, 0f, 1, 1);
        assertEquals(0f, pad1d[0], EPSILON);
        assertEquals(1f, pad1d[1], EPSILON);
        assertEquals(0f, pad1d[3], EPSILON);
        System.out.println("[OK]");

        // Reverse 1D
        System.out.print("  Reverse 1D: ");
        float[] rev1d = executeReverse1D(new float[]{1, 2, 3});
        assertEquals(3f, rev1d[0], EPSILON);
        assertEquals(1f, rev1d[2], EPSILON);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 3 operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private float[] executeIota1D(int n) {
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(
            List.of(n), StableHloAst.ScalarType.F32);
        StableHloAst.Value result = new StableHloAst.Value("0", resultType);

        StableHloAst.IotaOp op = new StableHloAst.IotaOp(result, 0, resultType);
        List<Tensor> results = backend.execute(op, List.of());

        return results.get(0).toFloatArray();
    }

    private float[] executeIota2D(int rows, int cols, int dim) {
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(
            List.of(rows, cols), StableHloAst.ScalarType.F32);
        StableHloAst.Value result = new StableHloAst.Value("0", resultType);

        StableHloAst.IotaOp op = new StableHloAst.IotaOp(result, dim, resultType);
        List<Tensor> results = backend.execute(op, List.of());

        return results.get(0).toFloatArray();
    }

    private float[] executePad1D(float[] input, float padValue, int lowPad, int highPad) {
        int n = input.length;
        int nOut = lowPad + n + highPad;

        try (Tensor tensorIn = Tensor.fromFloatArray(input, n);
             Tensor padValueTensor = Tensor.fromFloatArray(new float[]{padValue})) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType padValueType = new StableHloAst.TensorType(
                List.of(), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(nOut), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value padVal = new StableHloAst.Value("1", padValueType);
            StableHloAst.Value result = new StableHloAst.Value("2", resultType);

            StableHloAst.PadOp op = new StableHloAst.PadOp(
                result, operand, padVal,
                List.of((long) lowPad),
                List.of((long) highPad),
                List.of(0L),  // no interior padding
                resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn, padValueTensor));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executePad2D(float[] input, int[] inputShape, float padValue,
                                  int lowPad0, int highPad0, int lowPad1, int highPad1) {
        int inRows = inputShape[0];
        int inCols = inputShape[1];
        int outRows = lowPad0 + inRows + highPad0;
        int outCols = lowPad1 + inCols + highPad1;

        try (Tensor tensorIn = Tensor.fromFloatArray(input, inRows, inCols);
             Tensor padValueTensor = Tensor.fromFloatArray(new float[]{padValue})) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(inRows, inCols), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType padValueType = new StableHloAst.TensorType(
                List.of(), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(outRows, outCols), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value padVal = new StableHloAst.Value("1", padValueType);
            StableHloAst.Value result = new StableHloAst.Value("2", resultType);

            StableHloAst.PadOp op = new StableHloAst.PadOp(
                result, operand, padVal,
                List.of((long) lowPad0, (long) lowPad1),
                List.of((long) highPad0, (long) highPad1),
                List.of(0L, 0L),  // no interior padding
                resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn, padValueTensor));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeReverse1D(float[] input) {
        int n = input.length;

        try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", inputType);

            StableHloAst.ReverseOp op = new StableHloAst.ReverseOp(
                result, operand, List.of(0L), inputType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeReverse2D(float[] input, int[] shape, List<Long> dimensions) {
        int rows = shape[0];
        int cols = shape[1];

        try (Tensor tensorIn = Tensor.fromFloatArray(input, rows, cols)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(rows, cols), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", inputType);

            StableHloAst.ReverseOp op = new StableHloAst.ReverseOp(
                result, operand, dimensions, inputType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }
}
