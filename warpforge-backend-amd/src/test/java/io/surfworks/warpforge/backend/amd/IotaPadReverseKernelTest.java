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
 * Hardware execution tests for Iota, Pad, and Reverse HIP kernels on AMD GPUs.
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Iota, Pad, and Reverse HIP Kernels (AMD Hardware)")
class IotaPadReverseKernelTest {

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
    @DisplayName("HIP: Iota1D generates valid output")
    void testIota1DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Iota1D");
        String src = HipKernels.generateIota1DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void iota_1d_f32"));
        System.out.println("[PASS] Iota1D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Pad1D generates valid output")
    void testPad1DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Pad1D");
        String src = HipKernels.generatePad1DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void pad_1d_f32"));
        System.out.println("[PASS] Pad1D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Reverse1D generates valid output")
    void testReverse1DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Reverse1D");
        String src = HipKernels.generateReverse1DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void reverse_1d_f32"));
        System.out.println("[PASS] Reverse1D HIP generation OK");
    }

    // ==================== Hardware Execution Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Iota 1D generates sequence")
    void testIota1D() {
        System.out.println("[TEST] HIP Execution: Iota 1D");
        createContext();

        // Generate [0, 1, 2, 3, 4]
        float[] expected = {0, 1, 2, 3, 4};

        float[] result = executeIota1D(5);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Iota 1D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Iota 2D dim0 generates row indices")
    void testIota2DDim0() {
        System.out.println("[TEST] HIP Execution: Iota 2D dim0");
        createContext();

        // Generate 3x4 matrix with row indices:
        // [[0, 0, 0, 0],
        //  [1, 1, 1, 1],
        //  [2, 2, 2, 2]]
        float[] expected = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

        float[] result = executeIota2D(3, 4, 0);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Result (3x4, dim0): " + Arrays.toString(result));
        System.out.println("[PASS] Iota 2D dim0 OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Iota 2D dim1 generates column indices")
    void testIota2DDim1() {
        System.out.println("[TEST] HIP Execution: Iota 2D dim1");
        createContext();

        // Generate 3x4 matrix with column indices:
        // [[0, 1, 2, 3],
        //  [0, 1, 2, 3],
        //  [0, 1, 2, 3]]
        float[] expected = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

        float[] result = executeIota2D(3, 4, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Result (3x4, dim1): " + Arrays.toString(result));
        System.out.println("[PASS] Iota 2D dim1 OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Pad 1D with zeros")
    void testPad1D() {
        System.out.println("[TEST] HIP Execution: Pad 1D");
        createContext();

        // Pad [1, 2, 3] with 2 zeros on left and 1 zero on right
        // -> [0, 0, 1, 2, 3, 0]
        float[] input = {1, 2, 3};
        float[] expected = {0, 0, 1, 2, 3, 0};

        float[] result = executePad1D(input, 0f, 2, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Pad 1D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Pad 1D with custom value")
    void testPad1DCustomValue() {
        System.out.println("[TEST] HIP Execution: Pad 1D with custom value");
        createContext();

        // Pad [1, 2, 3] with -1 padding, 1 on left and 2 on right
        // -> [-1, 1, 2, 3, -1, -1]
        float[] input = {1, 2, 3};
        float[] expected = {-1, 1, 2, 3, -1, -1};

        float[] result = executePad1D(input, -1f, 1, 2);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Pad 1D with custom value OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Pad 2D")
    void testPad2D() {
        System.out.println("[TEST] HIP Execution: Pad 2D");
        createContext();

        // Input: [[1, 2], [3, 4]]
        // Pad with 1 row top, 0 rows bottom, 1 col left, 1 col right
        // Output: [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]]
        float[] input = {1, 2, 3, 4};
        int[] inputShape = {2, 2};
        float[] expected = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0};

        float[] result = executePad2D(input, inputShape, 0f, 1, 0, 1, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input (2x2): " + Arrays.toString(input));
        System.out.println("  Result (3x4): " + Arrays.toString(result));
        System.out.println("[PASS] Pad 2D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Reverse 1D")
    void testReverse1D() {
        System.out.println("[TEST] HIP Execution: Reverse 1D");
        createContext();

        // Reverse [1, 2, 3, 4, 5] -> [5, 4, 3, 2, 1]
        float[] input = {1, 2, 3, 4, 5};
        float[] expected = {5, 4, 3, 2, 1};

        float[] result = executeReverse1D(input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input: " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Reverse 1D OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Reverse 2D dim0")
    void testReverse2DDim0() {
        System.out.println("[TEST] HIP Execution: Reverse 2D dim0");
        createContext();

        // Input: [[1, 2], [3, 4], [5, 6]]
        // Reverse dim0 -> [[5, 6], [3, 4], [1, 2]]
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] shape = {3, 2};
        float[] expected = {5, 6, 3, 4, 1, 2};

        float[] result = executeReverse2D(input, shape, 0);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input (3x2): " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Reverse 2D dim0 OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Reverse 2D dim1")
    void testReverse2DDim1() {
        System.out.println("[TEST] HIP Execution: Reverse 2D dim1");
        createContext();

        // Input: [[1, 2, 3], [4, 5, 6]]
        // Reverse dim1 -> [[3, 2, 1], [6, 5, 4]]
        float[] input = {1, 2, 3, 4, 5, 6};
        int[] shape = {2, 3};
        float[] expected = {3, 2, 1, 6, 5, 4};

        float[] result = executeReverse2D(input, shape, 1);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input (2x3): " + Arrays.toString(input));
        System.out.println("  Result: " + Arrays.toString(result));
        System.out.println("[PASS] Reverse 2D dim1 OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: All Iota/Pad/Reverse Summary")
    void testAllIotaPadReverseSummary() {
        System.out.println("========================================");
        System.out.println("Iota, Pad, and Reverse HIP Operations Summary");
        System.out.println("========================================");
        createContext();

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
        String source = HipKernels.generateIota1DF32(HipKernels.SALT_NONE);
        String functionName = "iota_1d_f32";

        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("iota_1d_module", source);
        long function = context.getFunction(module, functionName);

        long dOut = context.allocate(byteSize);

        try {
            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dOut},
                n
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dOut);
        }
    }

    private float[] executeIota2D(int rows, int cols, int dim) {
        String source = dim == 0 ?
            HipKernels.generateIota2DDim0F32(HipKernels.SALT_NONE) :
            HipKernels.generateIota2DDim1F32(HipKernels.SALT_NONE);
        String functionName = dim == 0 ? "iota_2d_dim0_f32" : "iota_2d_dim1_f32";

        int n = rows * cols;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("iota_2d_dim" + dim + "_module", source);
        long function = context.getFunction(module, functionName);

        long dOut = context.allocate(byteSize);

        try {
            int gridX = (cols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (rows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dOut},
                rows, cols
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dOut);
        }
    }

    private float[] executePad1D(float[] input, float padValue, int lowPad, int highPad) {
        String source = HipKernels.generatePad1DF32(HipKernels.SALT_NONE);
        String functionName = "pad_1d_f32";

        int inputSize = input.length;
        int outputSize = lowPad + inputSize + highPad;
        long inputByteSize = inputSize * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("pad_1d_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(inputByteSize);
        long dOut = context.allocate(outputByteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(input, inputSize)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            context.launchKernelWithFloatAndIntParams(
                function,
                new int[]{gridSize(outputSize)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dIn, dOut},
                padValue,
                inputSize, lowPad, outputSize
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

    private float[] executePad2D(float[] input, int[] inputShape, float padValue,
                                  int lowPad0, int highPad0, int lowPad1, int highPad1) {
        String source = HipKernels.generatePad2DF32(HipKernels.SALT_NONE);
        String functionName = "pad_2d_f32";

        int inRows = inputShape[0];
        int inCols = inputShape[1];
        int outRows = lowPad0 + inRows + highPad0;
        int outCols = lowPad1 + inCols + highPad1;

        int inputSize = inRows * inCols;
        int outputSize = outRows * outCols;
        long inputByteSize = inputSize * 4L;
        long outputByteSize = outputSize * 4L;

        long module = context.compileAndLoadModule("pad_2d_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(inputByteSize);
        long dOut = context.allocate(outputByteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(input, inputSize)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            int gridX = (outCols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outRows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithFloatAndIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dIn, dOut},
                padValue,
                inRows, inCols, lowPad0, lowPad1, outRows, outCols
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

    private float[] executeReverse1D(float[] input) {
        String source = HipKernels.generateReverse1DF32(HipKernels.SALT_NONE);
        String functionName = "reverse_1d_f32";

        int n = input.length;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("reverse_1d_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {
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

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }

    private float[] executeReverse2D(float[] input, int[] shape, int dim) {
        String source = dim == 0 ?
            HipKernels.generateReverse2DDim0F32(HipKernels.SALT_NONE) :
            HipKernels.generateReverse2DDim1F32(HipKernels.SALT_NONE);
        String functionName = dim == 0 ? "reverse_2d_dim0_f32" : "reverse_2d_dim1_f32";

        int rows = shape[0];
        int cols = shape[1];
        int n = rows * cols;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("reverse_2d_dim" + dim + "_module", source);
        long function = context.getFunction(module, functionName);

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            int gridX = (cols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (rows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            context.launchKernelWithIntParams(
                function,
                new int[]{gridX, gridY}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D},
                0,
                new long[]{dIn, dOut},
                rows, cols
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }
}
