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
 * Hardware execution tests for complex HIP kernels on AMD GPUs.
 *
 * <p>Tests Gather, Scatter, BatchMatMul, MaxPool2D, AvgPool2D, Conv2D, and BatchNormInference.
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Complex Operations HIP Kernels (AMD Hardware)")
class ComplexOpsKernelTest {

    private static final float EPSILON = 1e-4f;
    private static final int BLOCK_SIZE = HipKernels.ELEMENTWISE_BLOCK_SIZE;
    private static final int BLOCK_SIZE_2D = 16;

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
    @DisplayName("HIP: Gather1D generates valid output")
    void testGather1DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Gather1D");
        String src = HipKernels.generateGather1DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void gather_1d_f32"));
        assertTrue(src.contains("operand[indices[i]]"));
        System.out.println("[PASS] Gather1D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Gather2D generates valid output")
    void testGather2DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Gather2D");
        String src = HipKernels.generateGather2DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void gather_2d_f32"));
        assertTrue(src.contains("embedding_dim"));
        System.out.println("[PASS] Gather2D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: ScatterAdd generates valid output")
    void testScatterAddSrcGeneration() {
        System.out.println("[TEST] HIP Generation: ScatterAdd");
        String src = HipKernels.generateScatterAddF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void scatter_add_f32"));
        assertTrue(src.contains("atomicAdd"));
        System.out.println("[PASS] ScatterAdd HIP generation OK");
    }

    @Test
    @DisplayName("HIP: BatchMatMul generates valid output")
    void testBatchMatMulSrcGeneration() {
        System.out.println("[TEST] HIP Generation: BatchMatMul");
        String src = HipKernels.generateBatchMatMulF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void batch_matmul_f32"));
        assertTrue(src.contains("fmaf"));
        System.out.println("[PASS] BatchMatMul HIP generation OK");
    }

    @Test
    @DisplayName("HIP: MaxPool2D generates valid output")
    void testMaxPool2DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: MaxPool2D");
        String src = HipKernels.generateMaxPool2DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void maxpool_2d_f32"));
        assertTrue(src.contains("fmaxf"));
        System.out.println("[PASS] MaxPool2D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: AvgPool2D generates valid output")
    void testAvgPool2DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: AvgPool2D");
        String src = HipKernels.generateAvgPool2DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void avgpool_2d_f32"));
        assertTrue(src.contains("/ (float)count"));
        System.out.println("[PASS] AvgPool2D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Conv2D generates valid output")
    void testConv2DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Conv2D");
        String src = HipKernels.generateConv2DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void conv2d_f32"));
        assertTrue(src.contains("fmaf"));
        System.out.println("[PASS] Conv2D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: BatchNormInference generates valid output")
    void testBatchNormInferenceSrcGeneration() {
        System.out.println("[TEST] HIP Generation: BatchNormInference");
        String src = HipKernels.generateBatchNormInferenceF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void batchnorm_inference_f32"));
        assertTrue(src.contains("sqrtf"));
        System.out.println("[PASS] BatchNormInference HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All complex operations support SALT_TIMING")
    void testAllOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All operations with SALT_TIMING");

        String[] ops = {"Gather1D", "Gather2D", "ScatterAdd", "BatchMatMul",
                        "MaxPool2D", "AvgPool2D", "Conv2D", "BatchNormInference"};
        String[] hipSources = {
            HipKernels.generateGather1DF32(HipKernels.SALT_TIMING),
            HipKernels.generateGather2DF32(HipKernels.SALT_TIMING),
            HipKernels.generateScatterAddF32(HipKernels.SALT_TIMING),
            HipKernels.generateBatchMatMulF32(HipKernels.SALT_TIMING),
            HipKernels.generateMaxPool2DF32(HipKernels.SALT_TIMING),
            HipKernels.generateAvgPool2DF32(HipKernels.SALT_TIMING),
            HipKernels.generateConv2DF32(HipKernels.SALT_TIMING),
            HipKernels.generateBatchNormInferenceF32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(hipSources[i].contains("timing"),
                ops[i] + " should have timing parameter");
            assertTrue(hipSources[i].contains("get_timer"),
                ops[i] + " should use get_timer");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All operations support SALT_TIMING");
    }

    // ==================== HIP Execution Tests (Require AMD GPU) ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Gather1D basic test")
    void testHipGather1D() {
        System.out.println("[TEST] HIP Execution: Gather1D");
        createContext();

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
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP Gather1D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Gather2D embedding lookup")
    void testHipGather2DEmbedding() {
        System.out.println("[TEST] HIP Execution: Gather2D embedding lookup");
        createContext();

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
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP Gather2D embedding lookup OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: ScatterAdd basic test")
    void testHipScatterAdd() {
        System.out.println("[TEST] HIP Execution: ScatterAdd");
        createContext();

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
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP ScatterAdd OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: BatchMatMul 2D matmul")
    void testHipBatchMatMul2D() {
        System.out.println("[TEST] HIP Execution: BatchMatMul 2D matmul");
        createContext();

        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A * B = [[19, 22], [43, 50]]
        float[] result = executeBatchMatMul(
            new float[]{1, 2, 3, 4}, new int[]{2, 2},
            new float[]{5, 6, 7, 8}, new int[]{2, 2}
        );

        assertEquals(19f, result[0], EPSILON, "C[0,0] = 1*5 + 2*7 = 19");
        assertEquals(22f, result[1], EPSILON, "C[0,1] = 1*6 + 2*8 = 22");
        assertEquals(43f, result[2], EPSILON, "C[1,0] = 3*5 + 4*7 = 43");
        assertEquals(50f, result[3], EPSILON, "C[1,1] = 3*6 + 4*8 = 50");
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP BatchMatMul 2D matmul OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: MaxPool2D basic test")
    void testHipMaxPool2D() {
        System.out.println("[TEST] HIP Execution: MaxPool2D");
        createContext();

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
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP MaxPool2D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: AvgPool2D basic test")
    void testHipAvgPool2D() {
        System.out.println("[TEST] HIP Execution: AvgPool2D");
        createContext();

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
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP AvgPool2D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Conv2D basic test")
    void testHipConv2D() {
        System.out.println("[TEST] HIP Execution: Conv2D");
        createContext();

        // Input: 4x4 all ones
        // Kernel: 2x2 = [[1, 1], [1, 1]]
        // Stride: 1x1, No padding
        // Output: 3x3, all values = 4 (sum of 2x2 kernel on 2x2 input window)
        float[] input = new float[16];
        for (int i = 0; i < 16; i++) {
            input[i] = 1f;
        }
        float[] kernel = {1, 1, 1, 1};

        float[] result = executeConv2D(input, 4, 4, kernel, 2, 2, 1, 1);

        // All output values should be 4 (1+1+1+1)
        for (int i = 0; i < 9; i++) {
            assertEquals(4f, result[i], EPSILON, "Output[" + i + "] should be 4");
        }
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP Conv2D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: BatchNormInference basic test")
    void testHipBatchNormInference() {
        System.out.println("[TEST] HIP Execution: BatchNormInference");
        createContext();

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
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] HIP BatchNormInference OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: All Complex Ops Summary")
    void testAllComplexOpsSummary() {
        System.out.println("========================================");
        System.out.println("Complex Operations HIP Summary");
        System.out.println("========================================");
        createContext();

        // Gather1D
        System.out.print("  Gather1D: ");
        float[] gather1d = executeGather1D(new float[]{10, 20, 30}, new int[]{2, 0});
        assertEquals(30f, gather1d[0], EPSILON);
        assertEquals(10f, gather1d[1], EPSILON);
        System.out.println("[OK]");

        // Gather2D
        System.out.print("  Gather2D: ");
        float[] gather2d = executeGather2D(new float[]{1, 2, 3, 4}, new int[]{2, 2}, new int[]{1});
        assertEquals(3f, gather2d[0], EPSILON);
        assertEquals(4f, gather2d[1], EPSILON);
        System.out.println("[OK]");

        // ScatterAdd
        System.out.print("  ScatterAdd: ");
        float[] scatter = executeScatterAdd(new float[]{0, 0, 0}, new int[]{1, 1}, new float[]{5, 3});
        assertEquals(8f, scatter[1], EPSILON);
        System.out.println("[OK]");

        // BatchMatMul
        System.out.print("  BatchMatMul: ");
        float[] matmul = executeBatchMatMul(
            new float[]{1, 0, 0, 1}, new int[]{2, 2},  // Identity
            new float[]{1, 2, 3, 4}, new int[]{2, 2}
        );
        assertEquals(1f, matmul[0], EPSILON);
        assertEquals(2f, matmul[1], EPSILON);
        System.out.println("[OK]");

        // MaxPool2D
        System.out.print("  MaxPool2D: ");
        float[] maxpool = executeMaxPool2D(new float[]{1, 2, 3, 4}, 2, 2, 2, 2, 1, 1);
        assertEquals(4f, maxpool[0], EPSILON);
        System.out.println("[OK]");

        // AvgPool2D
        System.out.print("  AvgPool2D: ");
        float[] avgpool = executeAvgPool2D(new float[]{1, 2, 3, 4}, 2, 2, 2, 2, 1, 1);
        assertEquals(2.5f, avgpool[0], EPSILON);
        System.out.println("[OK]");

        // Conv2D
        System.out.print("  Conv2D: ");
        float[] conv = executeConv2D(new float[]{1, 1, 1, 1}, 2, 2, new float[]{1, 1, 1, 1}, 2, 2, 1, 1);
        assertEquals(4f, conv[0], EPSILON);
        System.out.println("[OK]");

        // BatchNormInference
        System.out.print("  BatchNormInference: ");
        float[] bn = executeBatchNormInference(
            new float[]{1, 2}, new int[]{1, 2},
            new float[]{1, 1}, new float[]{0, 0},
            new float[]{1, 2}, new float[]{1, 1}, 0f
        );
        assertEquals(0f, bn[0], EPSILON);
        assertEquals(0f, bn[1], EPSILON);
        System.out.println("[OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 8 complex operations PASSED");
        System.out.println("========================================");
    }

    // ==================== Helper Methods ====================

    private float[] executeGather1D(float[] operand, int[] indices) {
        String source = HipKernels.generateGather1DF32(HipKernels.SALT_NONE);
        String functionName = "gather_1d_f32";

        int nIndices = indices.length;
        long operandByteSize = operand.length * 4L;
        long indicesByteSize = nIndices * 4L;
        long outputByteSize = nIndices * 4L;

        long module = context.compileAndLoadModule("gather_1d_module", source);
        long function = context.getFunction(module, functionName);

        long dOperand = context.allocate(operandByteSize);
        long dIndices = context.allocate(indicesByteSize);
        long dOutput = context.allocate(outputByteSize);

        try {
            try (Tensor operandTensor = Tensor.fromFloatArray(operand, operand.length);
                 Tensor indicesTensor = Tensor.fromIntArray(indices, nIndices)) {
                context.copyToDevice(dOperand, operandTensor.data());
                context.copyToDevice(dIndices, indicesTensor.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(nIndices)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dOperand, dIndices, dOutput},
                nIndices
            );

            context.synchronize();

            float[] result = new float[nIndices];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, nIndices)) {
                context.copyToHost(resultTensor.data(), dOutput, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dOperand);
            context.free(dIndices);
            context.free(dOutput);
        }
    }

    private float[] executeGather2D(float[] operand, int[] shape, int[] indices) {
        String source = HipKernels.generateGather2DF32(HipKernels.SALT_NONE);
        String functionName = "gather_2d_f32";

        int nIndices = indices.length;
        int embeddingDim = shape[1];
        int outputSize = nIndices * embeddingDim;

        long operandByteSize = operand.length * 4L;
        long indicesByteSize = nIndices * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("gather_2d_module", source);
        long function = context.getFunction(module, functionName);

        long dOperand = context.allocate(operandByteSize);
        long dIndices = context.allocate(indicesByteSize);
        long dOutput = context.allocate(outputByteSize);

        try {
            try (Tensor operandTensor = Tensor.fromFloatArray(operand, operand.length);
                 Tensor indicesTensor = Tensor.fromIntArray(indices, nIndices)) {
                context.copyToDevice(dOperand, operandTensor.data());
                context.copyToDevice(dIndices, indicesTensor.data());
            }

            int gridX = (embeddingDim + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (nIndices + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dOperand, dIndices, dOutput},
                embeddingDim, nIndices
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOutput, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dOperand);
            context.free(dIndices);
            context.free(dOutput);
        }
    }

    private float[] executeScatterAdd(float[] operand, int[] indices, float[] updates) {
        String source = HipKernels.generateScatterAddF32(HipKernels.SALT_NONE);
        String functionName = "scatter_add_f32";

        int nUpdates = updates.length;

        long operandByteSize = operand.length * 4L;
        long indicesByteSize = nUpdates * 4L;
        long updatesByteSize = nUpdates * 4L;

        long module = context.compileAndLoadModule("scatter_add_module", source);
        long function = context.getFunction(module, functionName);

        long dOutput = context.allocate(operandByteSize);
        long dIndices = context.allocate(indicesByteSize);
        long dUpdates = context.allocate(updatesByteSize);

        try {
            // Copy initial operand to output (scatter-add modifies in place)
            try (Tensor operandTensor = Tensor.fromFloatArray(operand, operand.length);
                 Tensor indicesTensor = Tensor.fromIntArray(indices, nUpdates);
                 Tensor updatesTensor = Tensor.fromFloatArray(updates, nUpdates)) {
                context.copyToDevice(dOutput, operandTensor.data());
                context.copyToDevice(dIndices, indicesTensor.data());
                context.copyToDevice(dUpdates, updatesTensor.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(nUpdates)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dOutput, dIndices, dUpdates},
                nUpdates
            );

            context.synchronize();

            float[] result = new float[operand.length];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, operand.length)) {
                context.copyToHost(resultTensor.data(), dOutput, operandByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dOutput);
            context.free(dIndices);
            context.free(dUpdates);
        }
    }

    private float[] executeBatchMatMul(float[] lhs, int[] lhsShape, float[] rhs, int[] rhsShape) {
        String source = HipKernels.generateBatchMatMulF32(HipKernels.SALT_NONE);
        String functionName = "batch_matmul_f32";

        int M = lhsShape[0];
        int K = lhsShape[1];
        int N = rhsShape[1];
        int batchSize = 1;  // Single batch for simplicity

        int outputSize = M * N;
        long lhsByteSize = lhs.length * 4L;
        long rhsByteSize = rhs.length * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("batch_matmul_module", source);
        long function = context.getFunction(module, functionName);

        long dLhs = context.allocate(lhsByteSize);
        long dRhs = context.allocate(rhsByteSize);
        long dOutput = context.allocate(outputByteSize);

        try {
            try (Tensor lhsTensor = Tensor.fromFloatArray(lhs, lhs.length);
                 Tensor rhsTensor = Tensor.fromFloatArray(rhs, rhs.length)) {
                context.copyToDevice(dLhs, lhsTensor.data());
                context.copyToDevice(dRhs, rhsTensor.data());
            }

            int gridX = (N + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (M + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY, batchSize}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                0,
                new long[]{dLhs, dRhs, dOutput},
                M, N, K, batchSize
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOutput, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dLhs);
            context.free(dRhs);
            context.free(dOutput);
        }
    }

    private float[] executeMaxPool2D(float[] input, int inH, int inW,
                                      int windowH, int windowW, int strideH, int strideW) {
        String source = HipKernels.generateMaxPool2DF32(HipKernels.SALT_NONE);
        String functionName = "maxpool_2d_f32";

        int outH = (inH - windowH) / strideH + 1;
        int outW = (inW - windowW) / strideW + 1;
        int outputSize = outH * outW;

        long inputByteSize = input.length * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("maxpool_2d_module", source);
        long function = context.getFunction(module, functionName);

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);

        try {
            try (Tensor inputTensor = Tensor.fromFloatArray(input, input.length)) {
                context.copyToDevice(dInput, inputTensor.data());
            }

            int gridX = (outW + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outH + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dInput, dOutput},
                inH, inW, windowH, windowW, strideH, strideW, outH, outW
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOutput, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dInput);
            context.free(dOutput);
        }
    }

    private float[] executeAvgPool2D(float[] input, int inH, int inW,
                                      int windowH, int windowW, int strideH, int strideW) {
        String source = HipKernels.generateAvgPool2DF32(HipKernels.SALT_NONE);
        String functionName = "avgpool_2d_f32";

        int outH = (inH - windowH) / strideH + 1;
        int outW = (inW - windowW) / strideW + 1;
        int outputSize = outH * outW;

        long inputByteSize = input.length * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("avgpool_2d_module", source);
        long function = context.getFunction(module, functionName);

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);

        try {
            try (Tensor inputTensor = Tensor.fromFloatArray(input, input.length)) {
                context.copyToDevice(dInput, inputTensor.data());
            }

            int gridX = (outW + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outH + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dInput, dOutput},
                inH, inW, windowH, windowW, strideH, strideW, outH, outW
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOutput, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dInput);
            context.free(dOutput);
        }
    }

    private float[] executeConv2D(float[] input, int inH, int inW,
                                   float[] kernel, int kH, int kW,
                                   int strideH, int strideW) {
        String source = HipKernels.generateConv2DF32(HipKernels.SALT_NONE);
        String functionName = "conv2d_f32";

        int outH = (inH - kH) / strideH + 1;
        int outW = (inW - kW) / strideW + 1;
        int outputSize = outH * outW;

        long inputByteSize = input.length * 4L;
        long kernelByteSize = kernel.length * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("conv2d_module", source);
        long function = context.getFunction(module, functionName);

        long dInput = context.allocate(inputByteSize);
        long dKernel = context.allocate(kernelByteSize);
        long dOutput = context.allocate(outputByteSize);

        try {
            try (Tensor inputTensor = Tensor.fromFloatArray(input, input.length);
                 Tensor kernelTensor = Tensor.fromFloatArray(kernel, kernel.length)) {
                context.copyToDevice(dInput, inputTensor.data());
                context.copyToDevice(dKernel, kernelTensor.data());
            }

            int gridX = (outW + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outH + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dInput, dKernel, dOutput},
                inH, inW, kH, kW, strideH, strideW, outH, outW
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOutput, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dInput);
            context.free(dKernel);
            context.free(dOutput);
        }
    }

    private float[] executeBatchNormInference(float[] operand, int[] shape,
                                               float[] scale, float[] offset,
                                               float[] mean, float[] variance,
                                               float epsilon) {
        String source = HipKernels.generateBatchNormInferenceF32(HipKernels.SALT_NONE);
        String functionName = "batchnorm_inference_f32";

        int n = operand.length;
        int numFeatures = shape[1];

        long operandByteSize = n * 4L;
        long paramByteSize = numFeatures * 4L;

        long module = context.compileAndLoadModule("batchnorm_inference_module", source);
        long function = context.getFunction(module, functionName);

        long dInput = context.allocate(operandByteSize);
        long dScale = context.allocate(paramByteSize);
        long dOffset = context.allocate(paramByteSize);
        long dMean = context.allocate(paramByteSize);
        long dVariance = context.allocate(paramByteSize);
        long dOutput = context.allocate(operandByteSize);

        try {
            try (Tensor inputTensor = Tensor.fromFloatArray(operand, n);
                 Tensor scaleTensor = Tensor.fromFloatArray(scale, numFeatures);
                 Tensor offsetTensor = Tensor.fromFloatArray(offset, numFeatures);
                 Tensor meanTensor = Tensor.fromFloatArray(mean, numFeatures);
                 Tensor varianceTensor = Tensor.fromFloatArray(variance, numFeatures)) {
                context.copyToDevice(dInput, inputTensor.data());
                context.copyToDevice(dScale, scaleTensor.data());
                context.copyToDevice(dOffset, offsetTensor.data());
                context.copyToDevice(dMean, meanTensor.data());
                context.copyToDevice(dVariance, varianceTensor.data());
            }

            context.launchKernelWithFloatAndIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dInput, dScale, dOffset, dMean, dVariance, dOutput},
                epsilon,
                n, numFeatures
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOutput, operandByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dInput);
            context.free(dScale);
            context.free(dOffset);
            context.free(dMean);
            context.free(dVariance);
            context.free(dOutput);
        }
    }
}
