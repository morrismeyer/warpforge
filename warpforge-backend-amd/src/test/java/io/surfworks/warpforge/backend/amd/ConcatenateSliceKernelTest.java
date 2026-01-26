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
 * Hardware execution tests for Concatenate and Slice HIP kernels on AMD GPUs.
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Concatenate and Slice HIP Kernels (AMD Hardware)")
class ConcatenateSliceKernelTest {

    private static final float EPSILON = 1e-6f;
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
    @DisplayName("HIP: Concatenate2 generates valid output")
    void testConcatenate2SrcGeneration() {
        System.out.println("[TEST] HIP Generation: Concatenate2");
        String src = HipKernels.generateConcatenate2F32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void concatenate_2_f32"));
        System.out.println("[PASS] Concatenate2 HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Slice1D generates valid output")
    void testSlice1DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Slice1D");
        String src = HipKernels.generateSlice1DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void slice_1d_f32"));
        System.out.println("[PASS] Slice1D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Slice2D generates valid output")
    void testSlice2DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Slice2D");
        String src = HipKernels.generateSlice2DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void slice_2d_f32"));
        System.out.println("[PASS] Slice2D HIP generation OK");
    }

    // ==================== Hardware Execution Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Concatenate two 1D tensors")
    void testConcatenate1D() {
        System.out.println("[TEST] HIP Execution: Concatenate two 1D tensors");
        createContext();

        // Concatenate [1, 2, 3] and [4, 5] -> [1, 2, 3, 4, 5]
        float[] inputA = {1, 2, 3};
        float[] inputB = {4, 5};
        float[] expected = {1, 2, 3, 4, 5};

        float[] result = executeConcatenate(inputA, inputB);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input A: " + Arrays.toString(inputA));
        System.out.println("  Input B: " + Arrays.toString(inputB));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Concatenate 1D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Concatenate equal-sized tensors")
    void testConcatenateEqualSized() {
        System.out.println("[TEST] HIP Execution: Concatenate equal-sized tensors");
        createContext();

        float[] inputA = {1, 2, 3, 4};
        float[] inputB = {5, 6, 7, 8};
        float[] expected = {1, 2, 3, 4, 5, 6, 7, 8};

        float[] result = executeConcatenate(inputA, inputB);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Concatenate equal-sized OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Slice 1D basic")
    void testSlice1DBasic() {
        System.out.println("[TEST] HIP Execution: Slice 1D basic");
        createContext();

        // Slice [0, 1, 2, 3, 4, 5] from index 1 to 4 -> [1, 2, 3]
        float[] input = {0, 1, 2, 3, 4, 5};
        int start = 1;
        int limit = 4;
        int stride = 1;
        float[] expected = {1, 2, 3};

        float[] result = executeSlice1D(input, start, limit, stride);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + Arrays.toString(input));
        System.out.println("  Slice [1:4]: " + Arrays.toString(result));
        System.out.println("[PASS] Slice 1D basic OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Slice 1D with stride")
    void testSlice1DWithStride() {
        System.out.println("[TEST] HIP Execution: Slice 1D with stride");
        createContext();

        // Slice [0, 1, 2, 3, 4, 5] from 0 to 6 with stride 2 -> [0, 2, 4]
        float[] input = {0, 1, 2, 3, 4, 5};
        int start = 0;
        int limit = 6;
        int stride = 2;
        float[] expected = {0, 2, 4};

        float[] result = executeSlice1D(input, start, limit, stride);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + Arrays.toString(input));
        System.out.println("  Slice [::2]: " + Arrays.toString(result));
        System.out.println("[PASS] Slice 1D with stride OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Slice 2D basic")
    void testSlice2DBasic() {
        System.out.println("[TEST] HIP Execution: Slice 2D basic");
        createContext();

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
        System.out.println("  Slice [0:2, 1:3]: " + Arrays.toString(result));
        System.out.println("[PASS] Slice 2D basic OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Slice 2D with stride")
    void testSlice2DWithStride() {
        System.out.println("[TEST] HIP Execution: Slice 2D with stride");
        createContext();

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
        System.out.println("  Slice [::2, ::2]: " + Arrays.toString(result));
        System.out.println("[PASS] Slice 2D with stride OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: All Concatenate/Slice Summary")
    void testAllConcatenateSliceSummary() {
        System.out.println("========================================");
        System.out.println("Concatenate and Slice HIP Operations Summary");
        System.out.println("========================================");
        createContext();

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
        String source = HipKernels.generateConcatenate2F32(HipKernels.SALT_NONE);
        String functionName = "concatenate_2_f32";

        int sizeA = inputA.length;
        int sizeB = inputB.length;
        int total = sizeA + sizeB;

        long byteSizeA = sizeA * 4L;
        long byteSizeB = sizeB * 4L;
        long byteSizeOut = total * 4L;

        long module = context.compileAndLoadModule("concatenate_module", source);
        long function = context.getFunction(module, functionName);

        long dA = context.allocate(byteSizeA);
        long dB = context.allocate(byteSizeB);
        long dOut = context.allocate(byteSizeOut);

        try {
            try (Tensor tensorA = Tensor.fromFloatArray(inputA, sizeA);
                 Tensor tensorB = Tensor.fromFloatArray(inputB, sizeB)) {
                context.copyToDevice(dA, tensorA.data());
                context.copyToDevice(dB, tensorB.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(total)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dA, dB, dOut},
                sizeA, sizeB
            );

            context.synchronize();

            float[] result = new float[total];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, total)) {
                context.copyToHost(resultTensor.data(), dOut, byteSizeOut);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dA);
            context.free(dB);
            context.free(dOut);
        }
    }

    private float[] executeSlice1D(float[] input, int start, int limit, int stride) {
        String source = HipKernels.generateSlice1DF32(HipKernels.SALT_NONE);
        String functionName = "slice_1d_f32";

        int inputSize = input.length;
        int outputSize = (limit - start + stride - 1) / stride;
        long inputByteSize = inputSize * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("slice_1d_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(inputByteSize);
        long dOut = context.allocate(outputByteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(input, inputSize)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(outputSize)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dIn, dOut},
                start, stride, outputSize
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOut, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }

    private float[] executeSlice2D(float[] input, int[] inputShape,
                                    int[] starts, int[] limits, int[] strides) {
        String source = HipKernels.generateSlice2DF32(HipKernels.SALT_NONE);
        String functionName = "slice_2d_f32";

        int inRows = inputShape[0];
        int inCols = inputShape[1];
        int outRows = (limits[0] - starts[0] + strides[0] - 1) / strides[0];
        int outCols = (limits[1] - starts[1] + strides[1] - 1) / strides[1];

        int inputSize = inRows * inCols;
        int outputSize = outRows * outCols;
        long inputByteSize = inputSize * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("slice_2d_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(inputByteSize);
        long dOut = context.allocate(outputByteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(input, inputSize)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            int gridX = (outCols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outRows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dIn, dOut},
                inCols, starts[0], starts[1], strides[0], strides[1], outRows, outCols
            );

            context.synchronize();

            float[] result = new float[outputSize];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, outputSize)) {
                context.copyToHost(resultTensor.data(), dOut, outputByteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }
}
