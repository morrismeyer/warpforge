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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for complex CUDA kernels: Gather, Scatter, DotGeneral, ReduceWindow,
 * Convolution, and BatchNorm.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Complex Operations CUDA Kernels")
class ComplexOpsKernelTest {

    private static final float EPSILON = 1e-4f;

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
    @DisplayName("PTX: Gather1D generates valid output")
    void testGather1DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Gather1D");
        String ptx = CudaKernels.generateGather1DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry gather_1d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Gather1D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Gather2D generates valid output")
    void testGather2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Gather2D");
        String ptx = CudaKernels.generateGather2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry gather_2d_f32"));
        assertTrue(ptx.contains("ld.global.f32"));
        assertTrue(ptx.contains("st.global.f32"));
        System.out.println("[PASS] Gather2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: ScatterAdd generates valid output")
    void testScatterAddPtxGeneration() {
        System.out.println("[TEST] PTX Generation: ScatterAdd");
        String ptx = CudaKernels.generateScatterAddF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry scatter_add_f32"));
        assertTrue(ptx.contains("atom.global.add.f32"));  // atomic add
        System.out.println("[PASS] ScatterAdd PTX generation OK");
    }

    @Test
    @DisplayName("PTX: BatchMatMul generates valid output")
    void testBatchMatMulPtxGeneration() {
        System.out.println("[TEST] PTX Generation: BatchMatMul");
        String ptx = CudaKernels.generateBatchMatMulF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry batch_matmul_f32"));
        assertTrue(ptx.contains("fma.rn.f32"));  // fused multiply-add
        System.out.println("[PASS] BatchMatMul PTX generation OK");
    }

    @Test
    @DisplayName("PTX: MaxPool2D generates valid output")
    void testMaxPool2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: MaxPool2D");
        String ptx = CudaKernels.generateMaxPool2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry maxpool_2d_f32"));
        assertTrue(ptx.contains("max.f32"));
        System.out.println("[PASS] MaxPool2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: AvgPool2D generates valid output")
    void testAvgPool2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: AvgPool2D");
        String ptx = CudaKernels.generateAvgPool2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry avgpool_2d_f32"));
        assertTrue(ptx.contains("div.approx.f32"));  // division for average
        System.out.println("[PASS] AvgPool2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Conv2D generates valid output")
    void testConv2DPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Conv2D");
        String ptx = CudaKernels.generateConv2DF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry conv2d_f32"));
        assertTrue(ptx.contains("fma.rn.f32"));  // fused multiply-add for convolution
        System.out.println("[PASS] Conv2D PTX generation OK");
    }

    @Test
    @DisplayName("PTX: BatchNormInference generates valid output")
    void testBatchNormInferencePtxGeneration() {
        System.out.println("[TEST] PTX Generation: BatchNormInference");
        String ptx = CudaKernels.generateBatchNormInferenceF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry batchnorm_inference_f32"));
        assertTrue(ptx.contains("sqrt.approx.f32"));  // sqrt for normalization
        System.out.println("[PASS] BatchNormInference PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All complex operations support SALT_TIMING")
    void testAllOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All operations with SALT_TIMING");

        String[] ops = {"Gather1D", "Gather2D", "ScatterAdd", "BatchMatMul",
                        "MaxPool2D", "AvgPool2D", "Conv2D", "BatchNormInference"};
        String[] ptxSources = {
            CudaKernels.generateGather1DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateGather2DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateScatterAddF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateBatchMatMulF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateMaxPool2DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateAvgPool2DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateConv2DF32(CudaKernels.SALT_TIMING),
            CudaKernels.generateBatchNormInferenceF32(CudaKernels.SALT_TIMING)
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

    // ==================== CUDA Execution Tests (Require NVIDIA GPU) ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Gather1D basic test")
    void testCudaGather1D() {
        System.out.println("[TEST] CUDA: Gather1D");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: [10, 20, 30, 40, 50]
        // Indices: [1, 3, 0]
        // Expected: [20, 40, 10]
        float[] result = executeGather1D(
            new float[]{10, 20, 30, 40, 50},
            new int[]{1, 3, 0}
        );

        assertEquals(20f, result[0], EPSILON, "indices[0]=1 should give operand[1]=20");
        assertEquals(40f, result[1], EPSILON, "indices[1]=3 should give operand[3]=40");
        assertEquals(10f, result[2], EPSILON, "indices[2]=0 should give operand[0]=10");
        System.out.println("[PASS] CUDA Gather1D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Gather2D embedding lookup")
    void testCudaGather2DEmbedding() {
        System.out.println("[TEST] CUDA: Gather2D embedding lookup");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Embedding table: 3 rows x 4 cols
        // Row 0: [1, 2, 3, 4]
        // Row 1: [5, 6, 7, 8]
        // Row 2: [9, 10, 11, 12]
        float[] embeddingData = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        };
        int[] shape = {3, 4};
        int[] indices = {2, 0};  // Get rows 2 and 0

        float[] result = executeGather2D(embeddingData, shape, indices);

        // Row 0 of result should be row 2 of operand: [9, 10, 11, 12]
        assertEquals(9f, result[0], EPSILON);
        assertEquals(10f, result[1], EPSILON);
        assertEquals(11f, result[2], EPSILON);
        assertEquals(12f, result[3], EPSILON);
        // Row 1 of result should be row 0 of operand: [1, 2, 3, 4]
        assertEquals(1f, result[4], EPSILON);
        assertEquals(2f, result[5], EPSILON);
        assertEquals(3f, result[6], EPSILON);
        assertEquals(4f, result[7], EPSILON);
        System.out.println("[PASS] CUDA Gather2D embedding lookup OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Scatter-add basic test")
    void testCudaScatterAdd() {
        System.out.println("[TEST] CUDA: Scatter-add");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Operand: [0, 0, 0, 0, 0]
        // Indices: [1, 3, 1]  (scatter at positions 1, 3, 1)
        // Updates: [10, 20, 5]
        // Expected: [0, 15, 0, 20, 0] (position 1 gets 10+5=15)
        float[] result = executeScatterAdd(
            new float[]{0, 0, 0, 0, 0},
            new int[]{1, 3, 1},
            new float[]{10, 20, 5}
        );

        assertEquals(0f, result[0], EPSILON);
        assertEquals(15f, result[1], EPSILON, "Position 1 should have 10+5=15");
        assertEquals(0f, result[2], EPSILON);
        assertEquals(20f, result[3], EPSILON, "Position 3 should have 20");
        assertEquals(0f, result[4], EPSILON);
        System.out.println("[PASS] CUDA Scatter-add OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: DotGeneral 2D matmul")
    void testCudaDotGeneral2D() {
        System.out.println("[TEST] CUDA: DotGeneral 2D matmul");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A * B = [[19, 22], [43, 50]]
        float[] result = executeDotGeneral2D(
            new float[]{1, 2, 3, 4}, new int[]{2, 2},
            new float[]{5, 6, 7, 8}, new int[]{2, 2}
        );

        assertEquals(19f, result[0], EPSILON, "C[0,0] = 1*5 + 2*7 = 19");
        assertEquals(22f, result[1], EPSILON, "C[0,1] = 1*6 + 2*8 = 22");
        assertEquals(43f, result[2], EPSILON, "C[1,0] = 3*5 + 4*7 = 43");
        assertEquals(50f, result[3], EPSILON, "C[1,1] = 3*6 + 4*8 = 50");
        System.out.println("[PASS] CUDA DotGeneral 2D matmul OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: MaxPool2D basic test")
    void testCudaMaxPool2D() {
        System.out.println("[TEST] CUDA: MaxPool2D");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: 4x4
        // [1,  2,  3,  4 ]
        // [5,  6,  7,  8 ]
        // [9,  10, 11, 12]
        // [13, 14, 15, 16]
        // Window: 2x2, Stride: 2x2
        // Output: 2x2 = [[6, 8], [14, 16]]
        float[] inputData = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };

        float[] result = executeMaxPool2D(inputData, 4, 4, 2, 2, 2, 2);

        assertEquals(6f, result[0], EPSILON, "Max of top-left 2x2 window");
        assertEquals(8f, result[1], EPSILON, "Max of top-right 2x2 window");
        assertEquals(14f, result[2], EPSILON, "Max of bottom-left 2x2 window");
        assertEquals(16f, result[3], EPSILON, "Max of bottom-right 2x2 window");
        System.out.println("[PASS] CUDA MaxPool2D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: AvgPool2D basic test")
    void testCudaAvgPool2D() {
        System.out.println("[TEST] CUDA: AvgPool2D");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Same input as MaxPool2D test
        float[] inputData = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        };

        float[] result = executeAvgPool2D(inputData, 4, 4, 2, 2, 2, 2);

        // Avg of top-left: (1+2+5+6)/4 = 3.5
        assertEquals(3.5f, result[0], EPSILON, "Avg of top-left 2x2 window");
        // Avg of top-right: (3+4+7+8)/4 = 5.5
        assertEquals(5.5f, result[1], EPSILON, "Avg of top-right 2x2 window");
        // Avg of bottom-left: (9+10+13+14)/4 = 11.5
        assertEquals(11.5f, result[2], EPSILON, "Avg of bottom-left 2x2 window");
        // Avg of bottom-right: (11+12+15+16)/4 = 13.5
        assertEquals(13.5f, result[3], EPSILON, "Avg of bottom-right 2x2 window");
        System.out.println("[PASS] CUDA AvgPool2D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Conv2D basic test")
    void testCudaConv2D() {
        System.out.println("[TEST] CUDA: Conv2D");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Input: 4x4 all ones
        // Kernel: 2x2 = [[1, 1], [1, 1]]
        // Stride: 1x1, No padding
        // Output: 3x3, all values = 4 (sum of 2x2 kernel on 2x2 input window)
        float[] input = new float[16];
        for (int i = 0; i < 16; i++) {
            input[i] = 1f;
        }
        float[] kernel = {1, 1, 1, 1};

        float[] result = executeConv2D(input, 4, 4, kernel, 2, 2, 1, 1, 0, 0);

        // All output values should be 4 (1+1+1+1)
        for (int i = 0; i < 9; i++) {
            assertEquals(4f, result[i], EPSILON, "Output[" + i + "] should be 4");
        }
        System.out.println("[PASS] CUDA Conv2D OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: BatchNormInference basic test")
    void testCudaBatchNormInference() {
        System.out.println("[TEST] CUDA: BatchNormInference");
        assumeTrue(CudaRuntime.isAvailable(), "NVIDIA GPU required");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Simple test: input with known mean/var
        // Input: [2, 2] shape (N=2, C=2)
        // Values: [[0, 2], [4, 6]]
        // Mean: [2, 4] (mean of each channel)
        // Var: [4, 4]
        // Scale: [1, 1]
        // Offset: [0, 0]
        // Epsilon: 0
        // Expected output: (input - mean) / sqrt(var) = (input - mean) / 2
        // [[-1, -1], [1, 1]]
        float[] result = executeBatchNormInference(
            new float[]{0, 2, 4, 6}, new int[]{2, 2},
            new float[]{1, 1},    // scale
            new float[]{0, 0},    // offset
            new float[]{2, 4},    // mean
            new float[]{4, 4},    // variance
            0f                     // epsilon
        );

        // (0-2)/2 = -1, (2-4)/2 = -1, (4-2)/2 = 1, (6-4)/2 = 1
        assertEquals(-1f, result[0], EPSILON);
        assertEquals(-1f, result[1], EPSILON);
        assertEquals(1f, result[2], EPSILON);
        assertEquals(1f, result[3], EPSILON);
        System.out.println("[PASS] CUDA BatchNormInference OK");
    }

    // ==================== Helper Methods ====================

    private float[] executeGather1D(float[] operand, int[] indices) {
        int nIndices = indices.length;

        try (Tensor operandTensor = Tensor.fromFloatArray(operand, operand.length);
             Tensor indicesTensor = Tensor.fromIntArray(indices, nIndices)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(nIndices), StableHloAst.ScalarType.F32);

            StableHloAst.GatherOp op = new StableHloAst.GatherOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("operand", null),
                new StableHloAst.Value("indices", null),
                List.of(),     // offsetDims
                List.of(0L),   // collapsedSliceDims
                List.of(0L),   // startIndexMap
                0L,            // indexVectorDim
                List.of(1L),   // sliceSizes
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(operandTensor, indicesTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeGather2D(float[] operand, int[] shape, int[] indices) {
        int nIndices = indices.length;
        int embeddingDim = shape[1];

        try (Tensor operandTensor = Tensor.fromFloatArray(operand, shape[0], shape[1]);
             Tensor indicesTensor = Tensor.fromIntArray(indices, nIndices)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(nIndices, embeddingDim), StableHloAst.ScalarType.F32);

            StableHloAst.GatherOp op = new StableHloAst.GatherOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("operand", null),
                new StableHloAst.Value("indices", null),
                List.of(1L),   // offsetDims
                List.of(0L),   // collapsedSliceDims
                List.of(0L),   // startIndexMap
                0L,            // indexVectorDim
                List.of(1L, (long) embeddingDim),   // sliceSizes
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(operandTensor, indicesTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeScatterAdd(float[] operand, int[] indices, float[] updates) {
        int nUpdates = updates.length;

        try (Tensor operandTensor = Tensor.fromFloatArray(operand, operand.length);
             Tensor indicesTensor = Tensor.fromIntArray(indices, nUpdates);
             Tensor updatesTensor = Tensor.fromFloatArray(updates, nUpdates)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(operand.length), StableHloAst.ScalarType.F32);

            StableHloAst.ScatterOp op = new StableHloAst.ScatterOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("operand", null),
                new StableHloAst.Value("indices", null),
                new StableHloAst.Value("updates", null),
                List.of(0L),   // updateWindowDims
                List.of(0L),   // insertedWindowDims
                List.of(0L),   // scatterDimsToOperandDims
                0L,            // indexVectorDim
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(operandTensor, indicesTensor, updatesTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeDotGeneral2D(float[] lhs, int[] lhsShape, float[] rhs, int[] rhsShape) {
        int M = lhsShape[0];
        int K = lhsShape[1];
        int N = rhsShape[1];

        try (Tensor lhsTensor = Tensor.fromFloatArray(lhs, lhsShape[0], lhsShape[1]);
             Tensor rhsTensor = Tensor.fromFloatArray(rhs, rhsShape[0], rhsShape[1])) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(M, N), StableHloAst.ScalarType.F32);

            StableHloAst.DotGeneralOp op = new StableHloAst.DotGeneralOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("lhs", null),
                new StableHloAst.Value("rhs", null),
                null,  // dimensionNumbers
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(lhsTensor, rhsTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeMaxPool2D(float[] input, int inH, int inW,
                                      int windowH, int windowW, int strideH, int strideW) {
        int outH = (inH - windowH) / strideH + 1;
        int outW = (inW - windowW) / strideW + 1;

        try (Tensor inputTensor = Tensor.fromFloatArray(input, inH, inW)) {
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(outH, outW), StableHloAst.ScalarType.F32);

            StableHloAst.ReduceWindowOp op = new StableHloAst.ReduceWindowOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("operand", null),
                new StableHloAst.Value("init", null),
                List.of((long) windowH, (long) windowW),
                List.of((long) strideH, (long) strideW),
                List.of(1L, 1L),
                List.of(1L, 1L),
                List.of(0L, 0L),
                List.of(0L, 0L),
                "maximum",
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(inputTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeAvgPool2D(float[] input, int inH, int inW,
                                      int windowH, int windowW, int strideH, int strideW) {
        int outH = (inH - windowH) / strideH + 1;
        int outW = (inW - windowW) / strideW + 1;

        try (Tensor inputTensor = Tensor.fromFloatArray(input, inH, inW)) {
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(outH, outW), StableHloAst.ScalarType.F32);

            StableHloAst.ReduceWindowOp op = new StableHloAst.ReduceWindowOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("operand", null),
                new StableHloAst.Value("init", null),
                List.of((long) windowH, (long) windowW),
                List.of((long) strideH, (long) strideW),
                List.of(1L, 1L),
                List.of(1L, 1L),
                List.of(0L, 0L),
                List.of(0L, 0L),
                "add",  // treated as avg
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(inputTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeConv2D(float[] input, int inH, int inW,
                                   float[] kernel, int kH, int kW,
                                   int strideH, int strideW, int padH, int padW) {
        int outH = (inH + 2 * padH - kH) / strideH + 1;
        int outW = (inW + 2 * padW - kW) / strideW + 1;

        try (Tensor inputTensor = Tensor.fromFloatArray(input, inH, inW);
             Tensor kernelTensor = Tensor.fromFloatArray(kernel, kH, kW)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(outH, outW), StableHloAst.ScalarType.F32);

            StableHloAst.ConvolutionOp op = new StableHloAst.ConvolutionOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("lhs", null),
                new StableHloAst.Value("rhs", null),
                List.of((long) strideH, (long) strideW),  // windowStrides
                List.of((long) padH, (long) padW),         // paddingLow
                List.of((long) padH, (long) padW),         // paddingHigh
                List.of(1L, 1L),                           // lhsDilation
                List.of(1L, 1L),                           // rhsDilation
                1L,                                        // featureGroupCount
                1L,                                        // batchGroupCount
                0L,                                        // inputBatchDimension
                1L,                                        // inputFeatureDimension
                List.of(0L, 1L),                          // inputSpatialDimensions
                0L,                                        // kernelInputFeatureDimension
                1L,                                        // kernelOutputFeatureDimension
                List.of(0L, 1L),                          // kernelSpatialDimensions
                0L,                                        // outputBatchDimension
                1L,                                        // outputFeatureDimension
                List.of(0L, 1L),                          // outputSpatialDimensions
                resultType
            );

            List<Tensor> results = backend.execute(op, List.of(inputTensor, kernelTensor));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeBatchNormInference(float[] operand, int[] shape,
                                               float[] scale, float[] offset,
                                               float[] mean, float[] variance,
                                               float epsilon) {
        int numFeatures = shape[1];

        try (Tensor operandTensor = Tensor.fromFloatArray(operand, shape[0], shape[1]);
             Tensor scaleTensor = Tensor.fromFloatArray(scale, numFeatures);
             Tensor offsetTensor = Tensor.fromFloatArray(offset, numFeatures);
             Tensor meanTensor = Tensor.fromFloatArray(mean, numFeatures);
             Tensor varianceTensor = Tensor.fromFloatArray(variance, numFeatures)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(shape[0], shape[1]), StableHloAst.ScalarType.F32);

            StableHloAst.BatchNormInferenceOp op = new StableHloAst.BatchNormInferenceOp(
                new StableHloAst.Value("result", resultType),
                new StableHloAst.Value("operand", null),
                new StableHloAst.Value("scale", null),
                new StableHloAst.Value("offset", null),
                new StableHloAst.Value("mean", null),
                new StableHloAst.Value("variance", null),
                epsilon,
                1L,  // feature index (channel dimension)
                resultType
            );

            List<Tensor> results = backend.execute(op,
                List.of(operandTensor, scaleTensor, offsetTensor, meanTensor, varianceTensor));
            return results.get(0).toFloatArray();
        }
    }
}
