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
 * Tests for Concatenate and Slice CUDA kernels.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Concatenate and Slice CUDA Kernels")
class ConcatenateSliceKernelTest {

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
    @DisplayName("PTX: Concatenate2 generates valid output")
    void testConcatenate2PtxGeneration() {
        System.out.println("[TEST] PTX Generation: Concatenate2");
        String ptx = CudaKernels.generateConcatenate2F32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry concatenate_2_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Concatenate2 PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Slice1D generates valid output")
    void testSlice1DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Slice1D");
        String ptx = CudaKernels.generateSlice1DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry slice_1d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Slice1D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Slice2D generates valid output")
    void testSlice2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Slice2D");
        String ptx = CudaKernels.generateSlice2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry slice_2d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Slice2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All concatenate/slice operations support SALT_TIMING")
    void testAllOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All operations with SALT_TIMING");

        String[] ops = {"Concatenate2", "Slice1D", "Slice2D"};
        String[] ptxSources = {
            CudaKernels.generateConcatenate2F32(CudaKernels.SALT_TIMING),
            CudaKernels.generateSlice1DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateSlice2DF32(CudaKernels.SALT_TIMING)
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
    @DisplayName("CUDA: Concatenate two 1D tensors")
    void testConcatenate1D() {
        System.out.println("[TEST] CUDA Execution: Concatenate two 1D tensors");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Concatenate [1, 2, 3] and [4, 5] -> [1, 2, 3, 4, 5]
        float[] inputA = {1, 2, 3};
        float[] inputB = {4, 5};
        float[] expected = {1, 2, 3, 4, 5};

        float[] result = executeConcatenate(inputA, inputB);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input A: " + java.util.Arrays.toString(inputA));
        System.out.println("  Input B: " + java.util.Arrays.toString(inputB));
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Concatenate 1D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Concatenate equal-sized tensors")
    void testConcatenateEqualSized() {
        System.out.println("[TEST] CUDA Execution: Concatenate equal-sized tensors");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] inputA = {1, 2, 3, 4};
        float[] inputB = {5, 6, 7, 8};
        float[] expected = {1, 2, 3, 4, 5, 6, 7, 8};

        float[] result = executeConcatenate(inputA, inputB);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Concatenate equal-sized OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Slice 1D basic")
    void testSlice1DBasic() {
        System.out.println("[TEST] CUDA Execution: Slice 1D basic");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Slice [0, 1, 2, 3, 4, 5] from index 1 to 4 -> [1, 2, 3]
        float[] input = {0, 1, 2, 3, 4, 5};
        int start = 1;
        int limit = 4;
        int stride = 1;
        float[] expected = {1, 2, 3};

        float[] result = executeSlice1D(input, start, limit, stride);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Slice [1:4]: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Slice 1D basic OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Slice 1D with stride")
    void testSlice1DWithStride() {
        System.out.println("[TEST] CUDA Execution: Slice 1D with stride");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Slice [0, 1, 2, 3, 4, 5] from 0 to 6 with stride 2 -> [0, 2, 4]
        float[] input = {0, 1, 2, 3, 4, 5};
        int start = 0;
        int limit = 6;
        int stride = 2;
        float[] expected = {0, 2, 4};

        float[] result = executeSlice1D(input, start, limit, stride);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + java.util.Arrays.toString(input));
        System.out.println("  Slice [::2]: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Slice 1D with stride OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Slice 2D basic")
    void testSlice2DBasic() {
        System.out.println("[TEST] CUDA Execution: Slice 2D basic");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[0, 1, 2, 3],
        //         [4, 5, 6, 7],
        //         [8, 9, 10, 11]]  (3x4)
        // Slice [0:2, 1:3] -> [[1, 2], [5, 6]]  (2x2)
        float[] input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        int[] inputShape = {3, 4};
        int[] starts = {0, 1};
        int[] limits = {2, 3};
        int[] strides = {1, 1};
        float[] expected = {1, 2, 5, 6};

        float[] result = executeSlice2D(input, inputShape, starts, limits, strides);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: 3x4 matrix");
        System.out.println("  Slice [0:2, 1:3]: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Slice 2D basic OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Slice 2D with stride")
    void testSlice2DWithStride() {
        System.out.println("[TEST] CUDA Execution: Slice 2D with stride");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[0, 1, 2, 3],
        //         [4, 5, 6, 7],
        //         [8, 9, 10, 11],
        //         [12, 13, 14, 15]]  (4x4)
        // Slice [0:4:2, 0:4:2] -> [[0, 2], [8, 10]]  (2x2)
        float[] input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        int[] inputShape = {4, 4};
        int[] starts = {0, 0};
        int[] limits = {4, 4};
        int[] strides = {2, 2};
        float[] expected = {0, 2, 8, 10};

        float[] result = executeSlice2D(input, inputShape, starts, limits, strides);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: 4x4 matrix");
        System.out.println("  Slice [::2, ::2]: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Slice 2D with stride OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Concatenate/Slice Summary")
    void testAllConcatenateSliceSummary() {
        System.out.println("========================================");
        System.out.println("Concatenate and Slice Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Concatenate
        System.out.print("  Concatenate: ");
        float[] concatResult = executeConcatenate(new float[]{1, 2}, new float[]{3, 4});
        assertEquals(1f, concatResult[0], EPSILON);
        assertEquals(4f, concatResult[3], EPSILON);
        System.out.println("[OK]");

        // Slice 1D
        System.out.print("  Slice 1D: ");
        float[] slice1dResult = executeSlice1D(new float[]{0, 1, 2, 3, 4}, 1, 4, 1);
        assertEquals(1f, slice1dResult[0], EPSILON);
        assertEquals(3f, slice1dResult[2], EPSILON);
        System.out.println("[OK]");

        // Slice 2D
        System.out.print("  Slice 2D: ");
        float[] slice2dResult = executeSlice2D(
            new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9},
            new int[]{3, 3},
            new int[]{0, 0}, new int[]{2, 2}, new int[]{1, 1}
        );
        assertEquals(1f, slice2dResult[0], EPSILON);
        assertEquals(5f, slice2dResult[3], EPSILON);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 2 operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private float[] executeConcatenate(float[] inputA, float[] inputB) {
        int nA = inputA.length;
        int nB = inputB.length;
        int nTotal = nA + nB;

        try (Tensor tensorA = Tensor.fromFloatArray(inputA, nA);
             Tensor tensorB = Tensor.fromFloatArray(inputB, nB)) {

            StableHloAst.TensorType typeA = new StableHloAst.TensorType(
                List.of(nA), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType typeB = new StableHloAst.TensorType(
                List.of(nB), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(nTotal), StableHloAst.ScalarType.F32);

            StableHloAst.Value valA = new StableHloAst.Value("0", typeA);
            StableHloAst.Value valB = new StableHloAst.Value("1", typeB);
            StableHloAst.Value result = new StableHloAst.Value("2", resultType);

            StableHloAst.ConcatenateOp op = new StableHloAst.ConcatenateOp(
                result, List.of(valA, valB), 0, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorA, tensorB));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeSlice1D(float[] input, int start, int limit, int stride) {
        int n = input.length;
        int nOut = (limit - start + stride - 1) / stride;

        try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(n), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(nOut), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.SliceOp op = new StableHloAst.SliceOp(
                result, operand,
                List.of((long) start),
                List.of((long) limit),
                List.of((long) stride),
                resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeSlice2D(float[] input, int[] inputShape,
                                    int[] starts, int[] limits, int[] strides) {
        int outRows = (limits[0] - starts[0] + strides[0] - 1) / strides[0];
        int outCols = (limits[1] - starts[1] + strides[1] - 1) / strides[1];

        try (Tensor tensorIn = Tensor.fromFloatArray(input, inputShape)) {

            StableHloAst.TensorType inputType = new StableHloAst.TensorType(
                List.of(inputShape[0], inputShape[1]), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(outRows, outCols), StableHloAst.ScalarType.F32);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.SliceOp op = new StableHloAst.SliceOp(
                result, operand,
                List.of((long) starts[0], (long) starts[1]),
                List.of((long) limits[0], (long) limits[1]),
                List.of((long) strides[0], (long) strides[1]),
                resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }
}
