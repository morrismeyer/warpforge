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
 * Tests for shape manipulation CUDA kernels (Reshape, Transpose, BroadcastInDim).
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Shape Manipulation CUDA Kernels")
class ShapeManipulationKernelTest {

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
    @DisplayName("PTX: Reshape generates valid output")
    void testReshapePtxGeneration() {
        System.out.println("[TEST] PTX Generation: Reshape");
        String ptx = CudaKernels.generateReshapeF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry reshape_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Reshape PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Transpose 2D generates valid output")
    void testTranspose2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Transpose 2D");
        String ptx = CudaKernels.generateTranspose2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry transpose_2d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Transpose 2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Broadcast scalar generates valid output")
    void testBroadcastScalarPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Broadcast scalar");
        String ptx = CudaKernels.generateBroadcastScalarF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry broadcast_scalar_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Broadcast scalar PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Broadcast 1D to 2D row generates valid output")
    void testBroadcast1Dto2DRowPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Broadcast 1D to 2D row");
        String ptx = CudaKernels.generateBroadcast1Dto2DRowF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry broadcast_1d_to_2d_row_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Broadcast 1D to 2D row PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Broadcast 1D to 2D col generates valid output")
    void testBroadcast1Dto2DColPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Broadcast 1D to 2D col");
        String ptx = CudaKernels.generateBroadcast1Dto2DColF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry broadcast_1d_to_2d_col_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Broadcast 1D to 2D col PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All shape operations support SALT_TIMING")
    void testAllShapeOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All shape operations with SALT_TIMING");

        String[] ops = {"Reshape", "Transpose2D", "BroadcastScalar", "Broadcast1Dto2DRow", "Broadcast1Dto2DCol"};
        String[] ptxSources = {
            CudaKernels.generateReshapeF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateTranspose2DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateBroadcastScalarF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateBroadcast1Dto2DRowF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateBroadcast1Dto2DColF32(CudaKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(ptxSources[i].contains("timing_ptr"),
                ops[i] + " should have timing_ptr parameter");
            assertTrue(ptxSources[i].contains("%globaltimer"),
                ops[i] + " should use globaltimer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All shape operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reshape 1D to 2D")
    void testReshape1Dto2D() {
        System.out.println("[TEST] CUDA Execution: Reshape 1D to 2D");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Reshape [6] -> [2, 3]
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] inputShape = {6};
        int[] outputShape = {2, 3};

        float[] result = executeReshape(input, inputShape, outputShape);
        assertArrayEquals(input, result, EPSILON);

        System.out.println("  Input shape: [6]");
        System.out.println("  Output shape: [2, 3]");
        System.out.println("  Data preserved: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Reshape 1D to 2D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Reshape 2D to 1D")
    void testReshape2Dto1D() {
        System.out.println("[TEST] CUDA Execution: Reshape 2D to 1D");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Reshape [2, 3] -> [6]
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] inputShape = {2, 3};
        int[] outputShape = {6};

        float[] result = executeReshape(input, inputShape, outputShape);
        assertArrayEquals(input, result, EPSILON);

        System.out.println("  Input shape: [2, 3]");
        System.out.println("  Output shape: [6]");
        System.out.println("[PASS] Reshape 2D to 1D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Transpose 2x3 matrix")
    void testTranspose2x3() {
        System.out.println("[TEST] CUDA Execution: Transpose 2x3 matrix");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[1, 2, 3], [4, 5, 6]] (2x3)
        // Output: [[1, 4], [2, 5], [3, 6]] (3x2)
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] inputShape = {2, 3};
        float[] expected = {1, 4, 2, 5, 3, 6};

        float[] result = executeTranspose(input, inputShape);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: [[1, 2, 3], [4, 5, 6]]");
        System.out.println("  Output: [[1, 4], [2, 5], [3, 6]]");
        System.out.println("[PASS] Transpose 2x3 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Transpose 3x2 matrix")
    void testTranspose3x2() {
        System.out.println("[TEST] CUDA Execution: Transpose 3x2 matrix");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [[1, 2], [3, 4], [5, 6]] (3x2)
        // Output: [[1, 3, 5], [2, 4, 6]] (2x3)
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] inputShape = {3, 2};
        float[] expected = {1, 3, 5, 2, 4, 6};

        float[] result = executeTranspose(input, inputShape);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: [[1, 2], [3, 4], [5, 6]]");
        System.out.println("  Output: [[1, 3, 5], [2, 4, 6]]");
        System.out.println("[PASS] Transpose 3x2 OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Broadcast scalar to 2x3")
    void testBroadcastScalar() {
        System.out.println("[TEST] CUDA Execution: Broadcast scalar to 2x3");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Broadcast 5.0 to [2, 3]
        float[] input = {5.0f};
        int[] outputShape = {2, 3};
        float[] expected = {5, 5, 5, 5, 5, 5};

        float[] result = executeBroadcastScalar(input, outputShape);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: 5.0");
        System.out.println("  Output shape: [2, 3]");
        System.out.println("  Result: " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Broadcast scalar OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Broadcast 1D to 2D along rows")
    void testBroadcast1Dto2DRow() {
        System.out.println("[TEST] CUDA Execution: Broadcast 1D to 2D along rows");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [1, 2, 3] (3 elements)
        // Output: [[1, 2, 3], [1, 2, 3]] (2x3) - broadcast along rows (dim 1)
        float[] input = {1, 2, 3};
        int[] inputShape = {3};
        int[] outputShape = {2, 3};
        List<Long> broadcastDims = List.of(1L);  // input maps to dim 1 (cols)
        float[] expected = {1, 2, 3, 1, 2, 3};

        float[] result = executeBroadcast1Dto2D(input, inputShape, outputShape, broadcastDims);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: [1, 2, 3]");
        System.out.println("  Output: [[1, 2, 3], [1, 2, 3]]");
        System.out.println("[PASS] Broadcast 1D to 2D along rows OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Broadcast 1D to 2D along columns")
    void testBroadcast1Dto2DCol() {
        System.out.println("[TEST] CUDA Execution: Broadcast 1D to 2D along columns");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [1, 2] (2 elements)
        // Output: [[1, 1, 1], [2, 2, 2]] (2x3) - broadcast along columns (dim 0)
        float[] input = {1, 2};
        int[] inputShape = {2};
        int[] outputShape = {2, 3};
        List<Long> broadcastDims = List.of(0L);  // input maps to dim 0 (rows)
        float[] expected = {1, 1, 1, 2, 2, 2};

        float[] result = executeBroadcast1Dto2D(input, inputShape, outputShape, broadcastDims);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: [1, 2]");
        System.out.println("  Output: [[1, 1, 1], [2, 2, 2]]");
        System.out.println("[PASS] Broadcast 1D to 2D along columns OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All Shape Manipulation Summary")
    void testAllShapeManipulationSummary() {
        System.out.println("========================================");
        System.out.println("Shape Manipulation Operations Summary");
        System.out.println("========================================");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Reshape
        System.out.print("  Reshape: ");
        float[] reshapeResult = executeReshape(new float[]{1, 2, 3, 4}, new int[]{4}, new int[]{2, 2});
        assertEquals(1f, reshapeResult[0], EPSILON);
        System.out.println("[OK]");

        // Transpose
        System.out.print("  Transpose: ");
        float[] transposeResult = executeTranspose(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        assertEquals(1f, transposeResult[0], EPSILON);
        assertEquals(3f, transposeResult[1], EPSILON);
        System.out.println("[OK]");

        // Broadcast scalar
        System.out.print("  Broadcast scalar: ");
        float[] scalarResult = executeBroadcastScalar(new float[]{7.0f}, new int[]{2, 2});
        assertEquals(7f, scalarResult[0], EPSILON);
        assertEquals(7f, scalarResult[3], EPSILON);
        System.out.println("[OK]");

        // Broadcast 1D to 2D
        System.out.print("  Broadcast 1D to 2D: ");
        float[] broadcastResult = executeBroadcast1Dto2D(
            new float[]{1, 2}, new int[]{2}, new int[]{2, 2}, List.of(1L));
        assertEquals(1f, broadcastResult[0], EPSILON);
        assertEquals(2f, broadcastResult[1], EPSILON);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 3 shape manipulation operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private float[] executeReshape(float[] input, int[] inputShape, int[] outputShape) {
        try (Tensor tensorIn = Tensor.fromFloatArray(input, inputShape)) {

            StableHloAst.TensorType inputType = createTensorType(inputShape);
            StableHloAst.TensorType resultType = createTensorType(outputShape);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.ReshapeOp op = new StableHloAst.ReshapeOp(result, operand, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeTranspose(float[] input, int[] inputShape) {
        try (Tensor tensorIn = Tensor.fromFloatArray(input, inputShape)) {

            StableHloAst.TensorType inputType = createTensorType(inputShape);
            int[] outputShape = {inputShape[1], inputShape[0]};
            StableHloAst.TensorType resultType = createTensorType(outputShape);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            // Standard 2D transpose permutation [1, 0]
            List<Long> permutation = List.of(1L, 0L);

            StableHloAst.TransposeOp op = new StableHloAst.TransposeOp(
                result, operand, permutation, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeBroadcastScalar(float[] input, int[] outputShape) {
        try (Tensor tensorIn = Tensor.fromFloatArray(input, 1)) {

            StableHloAst.TensorType inputType = createTensorType(new int[]{1});
            StableHloAst.TensorType resultType = createTensorType(outputShape);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            // Empty broadcast dimensions for scalar
            List<Long> broadcastDims = List.of();

            StableHloAst.BroadcastInDimOp op = new StableHloAst.BroadcastInDimOp(
                result, operand, broadcastDims, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private float[] executeBroadcast1Dto2D(float[] input, int[] inputShape,
                                            int[] outputShape, List<Long> broadcastDims) {
        try (Tensor tensorIn = Tensor.fromFloatArray(input, inputShape)) {

            StableHloAst.TensorType inputType = createTensorType(inputShape);
            StableHloAst.TensorType resultType = createTensorType(outputShape);

            StableHloAst.Value operand = new StableHloAst.Value("0", inputType);
            StableHloAst.Value result = new StableHloAst.Value("1", resultType);

            StableHloAst.BroadcastInDimOp op = new StableHloAst.BroadcastInDimOp(
                result, operand, broadcastDims, resultType);
            List<Tensor> results = backend.execute(op, List.of(tensorIn));

            return results.get(0).toFloatArray();
        }
    }

    private StableHloAst.TensorType createTensorType(int[] shape) {
        List<Integer> shapeList = new java.util.ArrayList<>();
        for (int dim : shape) {
            shapeList.add(dim);
        }
        return new StableHloAst.TensorType(shapeList, StableHloAst.ScalarType.F32);
    }
}
