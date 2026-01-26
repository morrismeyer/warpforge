package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.backend.amd.hip.HipRuntime;
import io.surfworks.warpforge.backend.amd.hip.HiprtcRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.lang.foreign.Arena;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Hardware execution tests for HIP kernels on AMD GPUs.
 *
 * <p>These tests compile HIP C++ source via HIPRTC and execute on actual AMD hardware.
 * All tests are tagged with @Tag("amd") and only run on machines with ROCm installed.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("HIP Kernel Execution Tests (AMD Hardware)")
class HipKernelExecutionTest {

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

    // ==================== Binary Elementwise Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Add kernel executes correctly on AMD GPU")
    void testAddExecution() {
        System.out.println("[TEST] HIP Execution: Add");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float[] expected = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};

        float[] result = executeBinaryOp("add", a, b);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("  Input A:   " + java.util.Arrays.toString(a));
        System.out.println("  Input B:   " + java.util.Arrays.toString(b));
        System.out.println("  Result:    " + java.util.Arrays.toString(result));
        System.out.println("[PASS] Add execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Multiply kernel executes correctly on AMD GPU")
    void testMultiplyExecution() {
        System.out.println("[TEST] HIP Execution: Multiply");
        createContext();

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {2.0f, 3.0f, 4.0f, 5.0f};
        float[] expected = {2.0f, 6.0f, 12.0f, 20.0f};

        float[] result = executeBinaryOp("multiply", a, b);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Multiply execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Subtract kernel executes correctly on AMD GPU")
    void testSubtractExecution() {
        System.out.println("[TEST] HIP Execution: Subtract");
        createContext();

        float[] a = {10.0f, 20.0f, 30.0f, 40.0f};
        float[] b = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] expected = {9.0f, 18.0f, 27.0f, 36.0f};

        float[] result = executeBinaryOp("subtract", a, b);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Subtract execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Divide kernel executes correctly on AMD GPU")
    void testDivideExecution() {
        System.out.println("[TEST] HIP Execution: Divide");
        createContext();

        float[] a = {10.0f, 20.0f, 30.0f, 40.0f};
        float[] b = {2.0f, 4.0f, 5.0f, 8.0f};
        float[] expected = {5.0f, 5.0f, 6.0f, 5.0f};

        float[] result = executeBinaryOp("divide", a, b);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Divide execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Maximum kernel executes correctly on AMD GPU")
    void testMaximumExecution() {
        System.out.println("[TEST] HIP Execution: Maximum");
        createContext();

        float[] a = {1.0f, 5.0f, 3.0f, 7.0f};
        float[] b = {2.0f, 4.0f, 6.0f, 8.0f};
        float[] expected = {2.0f, 5.0f, 6.0f, 8.0f};

        float[] result = executeBinaryOp("maximum", a, b);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Maximum execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Minimum kernel executes correctly on AMD GPU")
    void testMinimumExecution() {
        System.out.println("[TEST] HIP Execution: Minimum");
        createContext();

        float[] a = {1.0f, 5.0f, 3.0f, 7.0f};
        float[] b = {2.0f, 4.0f, 6.0f, 8.0f};
        float[] expected = {1.0f, 4.0f, 3.0f, 7.0f};

        float[] result = executeBinaryOp("minimum", a, b);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Minimum execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Power kernel executes correctly on AMD GPU")
    void testPowerExecution() {
        System.out.println("[TEST] HIP Execution: Power");
        createContext();

        float[] a = {2.0f, 3.0f, 4.0f, 2.0f};
        float[] b = {2.0f, 2.0f, 0.5f, 10.0f};
        float[] expected = {4.0f, 9.0f, 2.0f, 1024.0f};

        float[] result = executeBinaryOp("power", a, b);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Power execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Remainder kernel executes correctly on AMD GPU")
    void testRemainderExecution() {
        System.out.println("[TEST] HIP Execution: Remainder");
        createContext();

        float[] a = {10.0f, 7.0f, -10.0f, 5.5f};
        float[] b = {3.0f, 2.0f, 3.0f, 2.0f};
        float[] expected = {1.0f, 1.0f, -1.0f, 1.5f};

        float[] result = executeBinaryOp("remainder", a, b);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Remainder execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Atan2 kernel executes correctly on AMD GPU")
    void testAtan2Execution() {
        System.out.println("[TEST] HIP Execution: Atan2");
        createContext();

        // atan2(y, x) - angle in radians
        float[] y = {0.0f, 1.0f, 1.0f, -1.0f};
        float[] x = {1.0f, 0.0f, 1.0f, 1.0f};
        float[] expected = {0.0f, (float)(Math.PI / 2), (float)(Math.PI / 4), (float)(-Math.PI / 4)};

        float[] result = executeBinaryOp("atan2", y, x);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Atan2 execution OK");
    }

    // ==================== Unary Elementwise Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Negate kernel executes correctly on AMD GPU")
    void testNegateExecution() {
        System.out.println("[TEST] HIP Execution: Negate");
        createContext();

        float[] input = {1.0f, -2.0f, 3.0f, -4.0f};
        float[] expected = {-1.0f, 2.0f, -3.0f, 4.0f};

        float[] result = executeUnaryOp("negate", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Negate execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Abs kernel executes correctly on AMD GPU")
    void testAbsExecution() {
        System.out.println("[TEST] HIP Execution: Abs");
        createContext();

        float[] input = {-1.0f, 2.0f, -3.0f, 4.0f};
        float[] expected = {1.0f, 2.0f, 3.0f, 4.0f};

        float[] result = executeUnaryOp("abs", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Abs execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Sqrt kernel executes correctly on AMD GPU")
    void testSqrtExecution() {
        System.out.println("[TEST] HIP Execution: Sqrt");
        createContext();

        float[] input = {1.0f, 4.0f, 9.0f, 16.0f};
        float[] expected = {1.0f, 2.0f, 3.0f, 4.0f};

        float[] result = executeUnaryOp("sqrt", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Sqrt execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Exp kernel executes correctly on AMD GPU")
    void testExpExecution() {
        System.out.println("[TEST] HIP Execution: Exp");
        createContext();

        float[] input = {0.0f, 1.0f, 2.0f};
        float[] expected = {1.0f, (float)Math.E, (float)(Math.E * Math.E)};

        float[] result = executeUnaryOp("exp", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Exp execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Log kernel executes correctly on AMD GPU")
    void testLogExecution() {
        System.out.println("[TEST] HIP Execution: Log");
        createContext();

        float[] input = {1.0f, (float)Math.E, (float)(Math.E * Math.E)};
        float[] expected = {0.0f, 1.0f, 2.0f};

        float[] result = executeUnaryOp("log", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Log execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Tanh kernel executes correctly on AMD GPU")
    void testTanhExecution() {
        System.out.println("[TEST] HIP Execution: Tanh");
        createContext();

        float[] input = {0.0f, 1.0f, -1.0f};
        float[] expected = {0.0f, (float)Math.tanh(1.0), (float)Math.tanh(-1.0)};

        float[] result = executeUnaryOp("tanh", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Tanh execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Sin kernel executes correctly on AMD GPU")
    void testSinExecution() {
        System.out.println("[TEST] HIP Execution: Sin");
        createContext();

        float[] input = {0.0f, (float)(Math.PI / 2), (float)Math.PI};
        float[] expected = {0.0f, 1.0f, 0.0f};

        float[] result = executeUnaryOp("sin", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Sin execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Cos kernel executes correctly on AMD GPU")
    void testCosExecution() {
        System.out.println("[TEST] HIP Execution: Cos");
        createContext();

        float[] input = {0.0f, (float)(Math.PI / 2), (float)Math.PI};
        float[] expected = {1.0f, 0.0f, -1.0f};

        float[] result = executeUnaryOp("cos", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Cos execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Floor kernel executes correctly on AMD GPU")
    void testFloorExecution() {
        System.out.println("[TEST] HIP Execution: Floor");
        createContext();

        float[] input = {1.5f, 2.9f, -1.5f, -2.9f};
        float[] expected = {1.0f, 2.0f, -2.0f, -3.0f};

        float[] result = executeUnaryOp("floor", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Floor execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Ceil kernel executes correctly on AMD GPU")
    void testCeilExecution() {
        System.out.println("[TEST] HIP Execution: Ceil");
        createContext();

        float[] input = {1.5f, 2.1f, -1.5f, -2.1f};
        float[] expected = {2.0f, 3.0f, -1.0f, -2.0f};

        float[] result = executeUnaryOp("ceil", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Ceil execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Rsqrt kernel executes correctly on AMD GPU")
    void testRsqrtExecution() {
        System.out.println("[TEST] HIP Execution: Rsqrt");
        createContext();

        float[] input = {1.0f, 4.0f, 9.0f, 16.0f};
        float[] expected = {1.0f, 0.5f, 1.0f/3.0f, 0.25f};

        float[] result = executeUnaryOp("rsqrt", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Rsqrt execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Sign kernel executes correctly on AMD GPU")
    void testSignExecution() {
        System.out.println("[TEST] HIP Execution: Sign");
        createContext();

        float[] input = {-5.0f, 0.0f, 3.0f, -0.001f};
        float[] expected = {-1.0f, 0.0f, 1.0f, -1.0f};

        float[] result = executeUnaryOp("sign", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] Sign execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Tan kernel executes correctly on AMD GPU")
    void testTanExecution() {
        System.out.println("[TEST] HIP Execution: Tan");
        createContext();

        float[] input = {0.0f, (float)(Math.PI / 4), (float)(-Math.PI / 4)};
        float[] expected = {0.0f, 1.0f, -1.0f};

        float[] result = executeUnaryOp("tan", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Tan execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Logistic (sigmoid) kernel executes correctly on AMD GPU")
    void testLogisticExecution() {
        System.out.println("[TEST] HIP Execution: Logistic");
        createContext();

        float[] input = {0.0f, 1.0f, -1.0f, 10.0f};
        float[] expected = {0.5f, 1.0f / (1.0f + (float)Math.exp(-1.0)),
                           1.0f / (1.0f + (float)Math.exp(1.0)),
                           1.0f / (1.0f + (float)Math.exp(-10.0))};

        float[] result = executeUnaryOp("logistic", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Logistic execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Expm1 kernel executes correctly on AMD GPU")
    void testExpm1Execution() {
        System.out.println("[TEST] HIP Execution: Expm1");
        createContext();

        float[] input = {0.0f, 1.0f, -1.0f};
        float[] expected = {0.0f, (float)(Math.E - 1), (float)(1.0/Math.E - 1)};

        float[] result = executeUnaryOp("expm1", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Expm1 execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Log1p kernel executes correctly on AMD GPU")
    void testLog1pExecution() {
        System.out.println("[TEST] HIP Execution: Log1p");
        createContext();

        float[] input = {0.0f, (float)(Math.E - 1), 1.0f};
        float[] expected = {0.0f, 1.0f, (float)Math.log(2.0)};

        float[] result = executeUnaryOp("log1p", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Log1p execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: Cbrt kernel executes correctly on AMD GPU")
    void testCbrtExecution() {
        System.out.println("[TEST] HIP Execution: Cbrt");
        createContext();

        float[] input = {1.0f, 8.0f, 27.0f, -8.0f};
        float[] expected = {1.0f, 2.0f, 3.0f, -2.0f};

        float[] result = executeUnaryOp("cbrt", input);
        assertArrayEquals(expected, result, 1e-4f);

        System.out.println("[PASS] Cbrt execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: IsFinite kernel executes correctly on AMD GPU")
    void testIsFiniteExecution() {
        System.out.println("[TEST] HIP Execution: IsFinite");
        createContext();

        float[] input = {1.0f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        // isFinite returns 1.0 for finite, 0.0 for infinite/NaN
        float[] expected = {1.0f, 0.0f, 0.0f, 0.0f};

        float[] result = executeUnaryOp("isFinite", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] IsFinite execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: RoundNearestEven kernel executes correctly on AMD GPU")
    void testRoundNearestEvenExecution() {
        System.out.println("[TEST] HIP Execution: RoundNearestEven");
        createContext();

        // Round to nearest even (banker's rounding): 0.5 rounds to 0, 1.5 rounds to 2
        float[] input = {0.5f, 1.5f, 2.5f, 3.5f, -0.5f, -1.5f};
        float[] expected = {0.0f, 2.0f, 2.0f, 4.0f, 0.0f, -2.0f};

        float[] result = executeUnaryOp("roundNearestEven", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] RoundNearestEven execution OK");
    }

    @Test
    @Tag("amd")
    @DisplayName("HIP: RoundNearestAfz kernel executes correctly on AMD GPU")
    void testRoundNearestAfzExecution() {
        System.out.println("[TEST] HIP Execution: RoundNearestAfz");
        createContext();

        // Round nearest away from zero: 0.5 rounds to 1, -0.5 rounds to -1
        float[] input = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f};
        float[] expected = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f};

        float[] result = executeUnaryOp("roundNearestAfz", input);
        assertArrayEquals(expected, result, EPSILON);

        System.out.println("[PASS] RoundNearestAfz execution OK");
    }

    // ==================== Large Tensor Test ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Add handles large tensors (1M elements)")
    void testAddLargeTensor() {
        System.out.println("[TEST] HIP Execution: Add with 1M elements");
        createContext();

        int n = 1_000_000;
        float[] a = new float[n];
        float[] b = new float[n];
        float[] expected = new float[n];

        for (int i = 0; i < n; i++) {
            a[i] = i * 0.001f;
            b[i] = (n - i) * 0.001f;
            expected[i] = n * 0.001f;
        }

        float[] result = executeBinaryOp("add", a, b);

        // Check a few samples
        // Use larger tolerance for values ~1000 due to floating-point representation of 0.001f
        // The GPU computes a[i] + b[i] directly, while expected uses n * 0.001f
        // Both are correct within single-precision limits
        float largeTensorEpsilon = 1e-3f;
        assertEquals(expected[0], result[0], largeTensorEpsilon);
        assertEquals(expected[n / 2], result[n / 2], largeTensorEpsilon);
        assertEquals(expected[n - 1], result[n - 1], largeTensorEpsilon);

        System.out.println("  Verified 1M element addition");
        System.out.println("[PASS] Large tensor add OK");
    }

    // ==================== Helper Methods ====================

    private float[] executeBinaryOp(String opName, float[] a, float[] b) {
        String source = switch (opName) {
            case "add" -> HipKernels.generateAddF32(HipKernels.SALT_NONE);
            case "multiply" -> HipKernels.generateMultiplyF32(HipKernels.SALT_NONE);
            case "subtract" -> HipKernels.generateSubtractF32(HipKernels.SALT_NONE);
            case "divide" -> HipKernels.generateDivideF32(HipKernels.SALT_NONE);
            case "maximum" -> HipKernels.generateMaximumF32(HipKernels.SALT_NONE);
            case "minimum" -> HipKernels.generateMinimumF32(HipKernels.SALT_NONE);
            case "power" -> HipKernels.generatePowerF32(HipKernels.SALT_NONE);
            case "remainder" -> HipKernels.generateRemainderF32(HipKernels.SALT_NONE);
            case "atan2" -> HipKernels.generateAtan2F32(HipKernels.SALT_NONE);
            default -> throw new IllegalArgumentException("Unknown binary op: " + opName);
        };
        String functionName = opName + "_f32";

        int n = a.length;
        long byteSize = n * 4L;

        // Compile and load module
        long module = context.compileAndLoadModule(opName + "_module", source);
        long function = context.getFunction(module, functionName);

        // Allocate device memory
        long dA = context.allocate(byteSize);
        long dB = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            // Copy inputs to device
            try (Tensor tensorA = Tensor.fromFloatArray(a, n);
                 Tensor tensorB = Tensor.fromFloatArray(b, n)) {
                context.copyToDevice(dA, tensorA.data());
                context.copyToDevice(dB, tensorB.data());
            }

            // Launch kernel
            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dA, dB, dOut},
                n
            );

            // Synchronize
            context.synchronize();

            // Copy result back
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

    private float[] executeUnaryOp(String opName, float[] input) {
        String source = switch (opName) {
            case "negate" -> HipKernels.generateNegateF32(HipKernels.SALT_NONE);
            case "abs" -> HipKernels.generateAbsF32(HipKernels.SALT_NONE);
            case "exp" -> HipKernels.generateExpF32(HipKernels.SALT_NONE);
            case "log" -> HipKernels.generateLogF32(HipKernels.SALT_NONE);
            case "sqrt" -> HipKernels.generateSqrtF32(HipKernels.SALT_NONE);
            case "tanh" -> HipKernels.generateTanhF32(HipKernels.SALT_NONE);
            case "rsqrt" -> HipKernels.generateRsqrtF32(HipKernels.SALT_NONE);
            case "sin" -> HipKernels.generateSinF32(HipKernels.SALT_NONE);
            case "cos" -> HipKernels.generateCosF32(HipKernels.SALT_NONE);
            case "ceil" -> HipKernels.generateCeilF32(HipKernels.SALT_NONE);
            case "floor" -> HipKernels.generateFloorF32(HipKernels.SALT_NONE);
            case "sign" -> HipKernels.generateSignF32(HipKernels.SALT_NONE);
            case "tan" -> HipKernels.generateTanF32(HipKernels.SALT_NONE);
            case "logistic" -> HipKernels.generateLogisticF32(HipKernels.SALT_NONE);
            case "expm1" -> HipKernels.generateExpm1F32(HipKernels.SALT_NONE);
            case "log1p" -> HipKernels.generateLog1pF32(HipKernels.SALT_NONE);
            case "cbrt" -> HipKernels.generateCbrtF32(HipKernels.SALT_NONE);
            case "isFinite" -> HipKernels.generateIsFiniteF32(HipKernels.SALT_NONE);
            case "roundNearestEven" -> HipKernels.generateRoundNearestEvenF32(HipKernels.SALT_NONE);
            case "roundNearestAfz" -> HipKernels.generateRoundNearestAfzF32(HipKernels.SALT_NONE);
            default -> throw new IllegalArgumentException("Unknown unary op: " + opName);
        };
        // Convert camelCase opName to snake_case function name for special cases
        String functionName = switch (opName) {
            case "roundNearestEven" -> "round_nearest_even_f32";
            case "roundNearestAfz" -> "round_nearest_afz_f32";
            case "isFinite" -> "is_finite_f32";
            default -> opName + "_f32";
        };

        int n = input.length;
        long byteSize = n * 4L;

        // Compile and load module
        long module = context.compileAndLoadModule(opName + "_module", source);
        long function = context.getFunction(module, functionName);

        // Allocate device memory
        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            // Copy input to device
            try (Tensor tensorIn = Tensor.fromFloatArray(input, n)) {
                context.copyToDevice(dIn, tensorIn.data());
            }

            // Launch kernel
            context.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dIn, dOut},
                n
            );

            // Synchronize
            context.synchronize();

            // Copy result back
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

    // ==================== Environment Info Test ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP: Environment Info")
    void testEnvironmentInfo() {
        System.out.println("========================================");
        System.out.println("HIP Kernel Execution Environment");
        System.out.println("========================================");

        createContext();
        assertNotNull(context, "HipContext should be created successfully");

        System.out.println("  Device: AMD GPU (device 0)");
        System.out.println("  HIPRTC: " + (HiprtcRuntime.isAvailable() ? "Available" : "Not available"));
        System.out.println("  HIPRTC Code Version: " + HiprtcRuntime.VERSION);

        // Test HIPRTC version API call (simple FFM test)
        if (HiprtcRuntime.isAvailable()) {
            try (Arena arena = Arena.ofConfined()) {
                int[] version = HiprtcRuntime.getVersion(arena);
                System.out.println("  HIPRTC API Version: " + version[0] + "." + version[1]);
            } catch (Throwable t) {
                System.out.println("  HIPRTC API Version: FAILED - " + t.getClass().getSimpleName() + ": " + t.getMessage());
                throw new RuntimeException("HIPRTC getVersion FFM call failed", t);
            }
        }

        System.out.println("  Context: Created successfully");
        System.out.println("========================================");

        // Verify HIPRTC is available for compilation
        assertTrue(HiprtcRuntime.isAvailable(), "HIPRTC must be available for kernel compilation");
    }
}
