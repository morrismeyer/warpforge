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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Hardware execution tests for comparison and selection HIP kernels on AMD GPUs.
 *
 * <p>Operations: Compare (EQ, NE, LT, LE, GT, GE), Select, Clamp
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Comparison and Selection HIP Kernels (AMD Hardware)")
class CompareSelectClampKernelTest {

    private static final float EPSILON = 1e-5f;
    private static final int BLOCK_SIZE = HipKernels.ELEMENTWISE_BLOCK_SIZE;

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
    @DisplayName("HIP: Compare EQ generates valid output")
    void testCompareEqSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Compare EQ");
        String src = HipKernels.generateCompareF32("EQ", HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void compare_eq_f32"));
        System.out.println("[PASS] Compare EQ HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Select generates valid output")
    void testSelectSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Select");
        String src = HipKernels.generateSelectF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void select_f32"));
        System.out.println("[PASS] Select HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Clamp generates valid output")
    void testClampSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Clamp");
        String src = HipKernels.generateClampF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void clamp_f32"));
        System.out.println("[PASS] Clamp HIP generation OK");
    }

    // ==================== Hardware Execution Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare EQ executes correctly on AMD GPU")
    void testCompareEqExecution() {
        System.out.println("[TEST] HIP Execution: Compare EQ");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 7.0f, 8.0f};
        float[] b = {1.0f, 3.0f, 3.0f, 5.0f, 4.0f, 5.0f, 8.0f, 8.0f};
        float[] expected = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};

        float[] result = executeCompare(a, b, "EQ");
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Compare EQ execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare NE executes correctly on AMD GPU")
    void testCompareNeExecution() {
        System.out.println("[TEST] HIP Execution: Compare NE");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {1.0f, 3.0f, 3.0f, 4.0f};
        float[] expected = {0.0f, 1.0f, 0.0f, 0.0f};

        float[] result = executeCompare(a, b, "NE");
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Compare NE execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare LT executes correctly on AMD GPU")
    void testCompareLtExecution() {
        System.out.println("[TEST] HIP Execution: Compare LT");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 7.0f, 8.0f};
        float[] b = {2.0f, 2.0f, 2.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f};
        // a < b: 1<2=T, 2<2=F, 3<2=F, 4<5=T, 5<4=F, 5<6=T, 7<8=T, 8<7=F
        float[] expected = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};

        float[] result = executeCompare(a, b, "LT");
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Compare LT execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare LE executes correctly on AMD GPU")
    void testCompareLeExecution() {
        System.out.println("[TEST] HIP Execution: Compare LE");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {2.0f, 2.0f, 2.0f, 5.0f};
        // a <= b: 1<=2=T, 2<=2=T, 3<=2=F, 4<=5=T
        float[] expected = {1.0f, 1.0f, 0.0f, 1.0f};

        float[] result = executeCompare(a, b, "LE");
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Compare LE execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare GT executes correctly on AMD GPU")
    void testCompareGtExecution() {
        System.out.println("[TEST] HIP Execution: Compare GT");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 7.0f, 8.0f};
        float[] b = {0.0f, 2.0f, 4.0f, 3.0f, 5.0f, 4.0f, 6.0f, 9.0f};
        // a > b: 1>0=T, 2>2=F, 3>4=F, 4>3=T, 5>5=F, 5>4=T, 7>6=T, 8>9=F
        float[] expected = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};

        float[] result = executeCompare(a, b, "GT");
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Compare GT execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare GE executes correctly on AMD GPU")
    void testCompareGeExecution() {
        System.out.println("[TEST] HIP Execution: Compare GE");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {0.0f, 2.0f, 4.0f, 3.0f};
        // a >= b: 1>=0=T, 2>=2=T, 3>=4=F, 4>=3=T
        float[] expected = {1.0f, 1.0f, 0.0f, 1.0f};

        float[] result = executeCompare(a, b, "GE");
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Compare GE execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Select executes correctly on AMD GPU")
    void testSelectExecution() {
        System.out.println("[TEST] HIP Execution: Select");
        createContext();

        float[] pred = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        float[] onTrue = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        float[] onFalse = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] expected = {10.0f, 2.0f, 30.0f, 4.0f, 50.0f, 6.0f, 70.0f, 8.0f};

        float[] result = executeSelect(pred, onTrue, onFalse);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Pred:      " + Arrays.toString(pred));
        System.out.println("  On True:   " + Arrays.toString(onTrue));
        System.out.println("  On False:  " + Arrays.toString(onFalse));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Select execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Clamp executes correctly on AMD GPU")
    void testClampExecution() {
        System.out.println("[TEST] HIP Execution: Clamp");
        createContext();

        float[] min = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f};
        float[] operand = {-1.0f, 0.5f, 1.0f, 2.0f, -2.0f, 0.0f, 0.5f, 2.0f};
        float[] max = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        // clamp: max(min, min(operand, max))
        float[] expected = {0.0f, 0.5f, 1.0f, 1.0f, -1.0f, 0.0f, 0.5f, 1.0f};

        float[] result = executeClamp(min, operand, max);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Min:       " + Arrays.toString(min));
        System.out.println("  Operand:   " + Arrays.toString(operand));
        System.out.println("  Max:       " + Arrays.toString(max));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Clamp execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Compare and Select combined (like ReLU)")
    void testCompareSelectReLU() {
        System.out.println("[TEST] HIP Execution: Compare + Select (ReLU pattern)");
        createContext();

        // ReLU(x) = max(0, x) = select(x > 0, x, 0)
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f};
        float[] zeros = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // First compare: input > 0
        float[] pred = executeCompare(input, zeros, "GT");
        // Then select: pred ? input : 0
        float[] result = executeSelect(pred, input, zeros);

        float[] expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.5f, 3.0f};
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input:     " + Arrays.toString(input));
        System.out.println("  Pred (>0): " + Arrays.toString(pred));
        System.out.println("  ReLU:      " + Arrays.toString(result));
        System.out.println("[PASS] Compare + Select (ReLU) OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: All comparison directions work")
    void testAllComparisonDirections() {
        System.out.println("[TEST] HIP Execution: All comparison directions");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {2.0f, 2.0f, 2.0f, 2.0f};

        // Test EQ
        float[] eq = executeCompare(a, b, "EQ");
        assertArrayEquals(new float[]{0.0f, 1.0f, 0.0f, 0.0f}, eq, EPSILON);
        System.out.println("  [OK] EQ: " + Arrays.toString(eq));

        // Test NE
        float[] ne = executeCompare(a, b, "NE");
        assertArrayEquals(new float[]{1.0f, 0.0f, 1.0f, 1.0f}, ne, EPSILON);
        System.out.println("  [OK] NE: " + Arrays.toString(ne));

        // Test LT
        float[] lt = executeCompare(a, b, "LT");
        assertArrayEquals(new float[]{1.0f, 0.0f, 0.0f, 0.0f}, lt, EPSILON);
        System.out.println("  [OK] LT: " + Arrays.toString(lt));

        // Test LE
        float[] le = executeCompare(a, b, "LE");
        assertArrayEquals(new float[]{1.0f, 1.0f, 0.0f, 0.0f}, le, EPSILON);
        System.out.println("  [OK] LE: " + Arrays.toString(le));

        // Test GT
        float[] gt = executeCompare(a, b, "GT");
        assertArrayEquals(new float[]{0.0f, 0.0f, 1.0f, 1.0f}, gt, EPSILON);
        System.out.println("  [OK] GT: " + Arrays.toString(gt));

        // Test GE
        float[] ge = executeCompare(a, b, "GE");
        assertArrayEquals(new float[]{0.0f, 1.0f, 1.0f, 1.0f}, ge, EPSILON);
        System.out.println("  [OK] GE: " + Arrays.toString(ge));

        System.out.println("[PASS] All comparison directions OK");
    }

    // ==================== Helper Methods ====================

    private float[] executeCompare(float[] a, float[] b, String direction) {
        String source = HipKernels.generateCompareF32(direction, HipKernels.SALT_NONE);
        String functionName = "compare_" + direction.toLowerCase() + "_f32";

        int n = a.length;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("compare_" + direction + "_module", source);
        long function = context.getFunction(module, functionName);

        long dA = context.allocate(byteSize);
        long dB = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorA = Tensor.fromFloatArray(a, n);
                 Tensor tensorB = Tensor.fromFloatArray(b, n)) {
                context.copyToDevice(dA, tensorA.data());
                context.copyToDevice(dB, tensorB.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dA, dB, dOut},
                n
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dA);
            context.free(dB);
            context.free(dOut);
        }
    }

    private float[] executeSelect(float[] pred, float[] onTrue, float[] onFalse) {
        String source = HipKernels.generateSelectF32(HipKernels.SALT_NONE);
        String functionName = "select_f32";

        int n = pred.length;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("select_module", source);
        long function = context.getFunction(module, functionName);

        long dPred = context.allocate(byteSize);
        long dOnTrue = context.allocate(byteSize);
        long dOnFalse = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorPred = Tensor.fromFloatArray(pred, n);
                 Tensor tensorOnTrue = Tensor.fromFloatArray(onTrue, n);
                 Tensor tensorOnFalse = Tensor.fromFloatArray(onFalse, n)) {
                context.copyToDevice(dPred, tensorPred.data());
                context.copyToDevice(dOnTrue, tensorOnTrue.data());
                context.copyToDevice(dOnFalse, tensorOnFalse.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dPred, dOnTrue, dOnFalse, dOut},
                n
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dPred);
            context.free(dOnTrue);
            context.free(dOnFalse);
            context.free(dOut);
        }
    }

    private float[] executeClamp(float[] min, float[] operand, float[] max) {
        String source = HipKernels.generateClampF32(HipKernels.SALT_NONE);
        String functionName = "clamp_f32";

        int n = operand.length;
        long byteSize = n * 4L;

        long module = context.compileAndLoadModule("clamp_module", source);
        long function = context.getFunction(module, functionName);

        long dMin = context.allocate(byteSize);
        long dOperand = context.allocate(byteSize);
        long dMax = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            try (Tensor tensorMin = Tensor.fromFloatArray(min, n);
                 Tensor tensorOperand = Tensor.fromFloatArray(operand, n);
                 Tensor tensorMax = Tensor.fromFloatArray(max, n)) {
                context.copyToDevice(dMin, tensorMin.data());
                context.copyToDevice(dOperand, tensorOperand.data());
                context.copyToDevice(dMax, tensorMax.data());
            }

            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dMin, dOperand, dMax, dOut},
                n
            );

            context.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                context.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            context.free(dMin);
            context.free(dOperand);
            context.free(dMax);
            context.free(dOut);
        }
    }
}
