package io.surfworks.warpforge.backend.amd;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.backend.amd.hip.HipRuntime;
import io.surfworks.warpforge.backend.amd.hip.HiprtcRuntime;
import io.surfworks.warpforge.backend.cpu.CpuBackend;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration tests comparing CPU and AMD backend outputs.
 *
 * <p>The CPU backend is the source of truth for correctness.
 * These tests verify that HIP kernel execution produces numerically
 * equivalent results within floating-point tolerance.
 *
 * <p>This validates the custom HIP implementation against the
 * reference Java implementation.
 */
@Tag("amd")
@DisplayName("CPU vs AMD Backend Comparison")
class CpuAmdComparisonTest {

    private static final float TOLERANCE = 1e-5f;
    private static final long SEED = 42L;
    private static final int BLOCK_SIZE = HipKernels.ELEMENTWISE_BLOCK_SIZE;

    private CpuBackend cpuBackend;
    private HipContext hipContext;

    @BeforeEach
    void setUp() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");
        assumeTrue(HiprtcRuntime.isAvailable(), "HIPRTC not available");

        cpuBackend = new CpuBackend();

        try {
            hipContext = HipContext.create(0);
        } catch (Exception e) {
            assumeTrue(false, "HIP context creation failed: " + e.getMessage());
        }
    }

    @AfterEach
    void tearDown() {
        if (cpuBackend != null) {
            cpuBackend.close();
        }
        if (hipContext != null) {
            hipContext.close();
        }
    }

    private int gridSize(int n) {
        return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    // ==================== Add Operation Tests ====================

    @Test
    @DisplayName("Add: Small tensor - exact match")
    void testAddSmallTensor() {
        System.out.println("[TEST] CPU vs AMD: Add (small tensor)");
        float[] aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] bData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

        compareAddResults(aData, bData, 8);
        System.out.println("[PASS] Add small tensor OK");
    }

    @Test
    @DisplayName("Add: Medium tensor (10K elements)")
    void testAddMediumTensor() {
        System.out.println("[TEST] CPU vs AMD: Add (10K elements)");
        int n = 10_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareAddResults(aData, bData, n);
        System.out.println("[PASS] Add medium tensor OK");
    }

    @Test
    @DisplayName("Add: Large tensor (1M elements)")
    void testAddLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Add (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareAddResults(aData, bData, n);
        System.out.println("[PASS] Add large tensor OK");
    }

    @Test
    @DisplayName("Add: Edge case - zeros")
    void testAddZeros() {
        System.out.println("[TEST] CPU vs AMD: Add with zeros");
        float[] aData = new float[1024]; // All zeros
        float[] bData = generateRandomFloats(1024, SEED);

        compareAddResults(aData, bData, 1024);
        System.out.println("[PASS] Add zeros OK");
    }

    @Test
    @DisplayName("Add: Edge case - negative numbers")
    void testAddNegatives() {
        System.out.println("[TEST] CPU vs AMD: Add with negatives");
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = -aData[i]; // Result should be all zeros
        }

        compareAddResults(aData, bData, n);
        System.out.println("[PASS] Add negatives OK");
    }

    // ==================== Multiply Operation Tests ====================

    @Test
    @DisplayName("Multiply: Small tensor - exact match")
    void testMultiplySmallTensor() {
        System.out.println("[TEST] CPU vs AMD: Multiply (small tensor)");
        float[] aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] bData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

        compareMultiplyResults(aData, bData, 8);
        System.out.println("[PASS] Multiply small tensor OK");
    }

    @Test
    @DisplayName("Multiply: Large tensor (1M elements)")
    void testMultiplyLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Multiply (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareMultiplyResults(aData, bData, n);
        System.out.println("[PASS] Multiply large tensor OK");
    }

    @Test
    @DisplayName("Multiply: Edge case - ones (identity)")
    void testMultiplyOnes() {
        System.out.println("[TEST] CPU vs AMD: Multiply by ones");
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = 1.0f;
        }

        float[] cpuResult = executeCpuMultiply(aData, bData, n);
        float[] amdResult = executeHipMultiply(aData, bData, n);

        // Multiplying by 1 should return the original values
        assertArrayEquals(aData, cpuResult, TOLERANCE);
        assertArrayEquals(aData, amdResult, TOLERANCE);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE);
        System.out.println("[PASS] Multiply by ones OK");
    }

    // ==================== Subtract Operation Tests ====================

    @Test
    @DisplayName("Subtract: CPU vs AMD - 1M elements")
    void testSubtractLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Subtract (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareSubtractResults(aData, bData, n);
        System.out.println("[PASS] Subtract large tensor OK");
    }

    @Test
    @DisplayName("Subtract: Same values yields zeros")
    void testSubtractSameValues() {
        System.out.println("[TEST] CPU vs AMD: Subtract (a - a = 0)");
        int n = 1024;
        float[] data = generateRandomFloats(n, SEED);

        float[] cpuResult = executeCpuSubtract(data, data.clone(), n);
        float[] amdResult = executeHipSubtract(data, data.clone(), n);

        for (int i = 0; i < n; i++) {
            assertEquals(0f, cpuResult[i], TOLERANCE, "CPU result at " + i + " should be 0");
            assertEquals(0f, amdResult[i], TOLERANCE, "AMD result at " + i + " should be 0");
        }
        System.out.println("[PASS] Subtract same values OK");
    }

    // ==================== Negate Operation Tests ====================

    @Test
    @DisplayName("Negate: CPU vs AMD - 1M elements")
    void testNegateLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Negate (1M elements)");
        int n = 1_000_000;
        float[] data = generateRandomFloats(n, SEED);

        compareNegateResults(data, n);
        System.out.println("[PASS] Negate large tensor OK");
    }

    // ==================== Abs Operation Tests ====================

    @Test
    @DisplayName("Abs: CPU vs AMD - 1M elements")
    void testAbsLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Abs (1M elements)");
        int n = 1_000_000;
        float[] data = generateRandomFloats(n, SEED);

        compareAbsResults(data, n);
        System.out.println("[PASS] Abs large tensor OK");
    }

    // ==================== Exp Operation Tests ====================

    @Test
    @DisplayName("Exp: CPU vs AMD - bounded values")
    void testExpBoundedValues() {
        System.out.println("[TEST] CPU vs AMD: Exp (bounded to avoid overflow)");
        int n = 10_000;
        float[] data = generateBoundedRandomFloats(n, SEED, -5.0f, 5.0f);

        // Exp uses approximation, so higher tolerance
        compareExpResults(data, n, 1e-3f);
        System.out.println("[PASS] Exp bounded values OK");
    }

    // ==================== Sqrt Operation Tests ====================

    @Test
    @DisplayName("Sqrt: CPU vs AMD - positive values")
    void testSqrtPositiveValues() {
        System.out.println("[TEST] CPU vs AMD: Sqrt (positive values)");
        int n = 10_000;
        float[] data = generateBoundedRandomFloats(n, SEED, 0.0f, 100.0f);

        compareSqrtResults(data, n, 1e-4f);
        System.out.println("[PASS] Sqrt positive values OK");
    }

    // ==================== Tanh Operation Tests ====================

    @Test
    @DisplayName("Tanh: CPU vs AMD - 10K elements")
    void testTanhMediumTensor() {
        System.out.println("[TEST] CPU vs AMD: Tanh (10K elements)");
        int n = 10_000;
        float[] data = generateRandomFloats(n, SEED);

        // Tanh uses approximation, so higher tolerance
        compareTanhResults(data, n, 1e-3f);
        System.out.println("[PASS] Tanh medium tensor OK");
    }

    // ==================== Maximum Operation Tests ====================

    @Test
    @DisplayName("Maximum: CPU vs AMD - 1M elements")
    void testMaximumLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Maximum (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareMaximumResults(aData, bData, n);
        System.out.println("[PASS] Maximum large tensor OK");
    }

    // ==================== Minimum Operation Tests ====================

    @Test
    @DisplayName("Minimum: CPU vs AMD - 1M elements")
    void testMinimumLargeTensor() {
        System.out.println("[TEST] CPU vs AMD: Minimum (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareMinimumResults(aData, bData, n);
        System.out.println("[PASS] Minimum large tensor OK");
    }

    // ==================== Combined Summary Test ====================

    @Test
    @DisplayName("All Elementwise Ops: Summary test")
    void testAllElementwiseOpsSummary() {
        System.out.println("\n========================================");
        System.out.println("CPU vs AMD: Elementwise Operations Summary");
        System.out.println("========================================");

        int n = 10_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        // Binary ops
        System.out.print("  Add: ");
        compareAddResults(aData, bData, n);
        System.out.println("[OK]");

        System.out.print("  Multiply: ");
        compareMultiplyResults(aData, bData, n);
        System.out.println("[OK]");

        System.out.print("  Subtract: ");
        compareSubtractResults(aData, bData, n);
        System.out.println("[OK]");

        System.out.print("  Maximum: ");
        compareMaximumResults(aData, bData, n);
        System.out.println("[OK]");

        System.out.print("  Minimum: ");
        compareMinimumResults(aData, bData, n);
        System.out.println("[OK]");

        // Unary ops
        System.out.print("  Negate: ");
        compareNegateResults(aData, n);
        System.out.println("[OK]");

        System.out.print("  Abs: ");
        compareAbsResults(aData, n);
        System.out.println("[OK]");

        // Ops that need bounded input
        float[] expData = generateBoundedRandomFloats(n, SEED, -5.0f, 5.0f);
        System.out.print("  Exp: ");
        compareExpResults(expData, n, 1e-3f);
        System.out.println("[OK]");

        float[] sqrtData = generateBoundedRandomFloats(n, SEED, 0.0f, 100.0f);
        System.out.print("  Sqrt: ");
        compareSqrtResults(sqrtData, n, 1e-4f);
        System.out.println("[OK]");

        System.out.print("  Tanh: ");
        compareTanhResults(aData, n, 1e-3f);
        System.out.println("[OK]");

        System.out.println("========================================");
        System.out.println("All 10 elementwise operations PASSED");
        System.out.println("========================================\n");
    }

    // ==================== Binary Op Comparison Helpers ====================

    private void compareAddResults(float[] aData, float[] bData, int n) {
        float[] cpuResult = executeCpuAdd(aData, bData, n);
        float[] amdResult = executeHipAdd(aData, bData, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    private void compareMultiplyResults(float[] aData, float[] bData, int n) {
        float[] cpuResult = executeCpuMultiply(aData, bData, n);
        float[] amdResult = executeHipMultiply(aData, bData, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    private void compareSubtractResults(float[] aData, float[] bData, int n) {
        float[] cpuResult = executeCpuSubtract(aData, bData, n);
        float[] amdResult = executeHipSubtract(aData, bData, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    private void compareMaximumResults(float[] aData, float[] bData, int n) {
        float[] cpuResult = executeCpuMaximum(aData, bData, n);
        float[] amdResult = executeHipMaximum(aData, bData, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    private void compareMinimumResults(float[] aData, float[] bData, int n) {
        float[] cpuResult = executeCpuMinimum(aData, bData, n);
        float[] amdResult = executeHipMinimum(aData, bData, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    // ==================== Unary Op Comparison Helpers ====================

    private void compareNegateResults(float[] data, int n) {
        float[] cpuResult = executeCpuNegate(data, n);
        float[] amdResult = executeHipNegate(data, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    private void compareAbsResults(float[] data, int n) {
        float[] cpuResult = executeCpuAbs(data, n);
        float[] amdResult = executeHipAbs(data, n);
        assertArrayEquals(cpuResult, amdResult, TOLERANCE, "AMD output doesn't match CPU reference");
    }

    private void compareExpResults(float[] data, int n, float tolerance) {
        float[] cpuResult = executeCpuExp(data, n);
        float[] amdResult = executeHipExp(data, n);
        assertArrayEquals(cpuResult, amdResult, tolerance, "AMD output doesn't match CPU reference");
    }

    private void compareSqrtResults(float[] data, int n, float tolerance) {
        float[] cpuResult = executeCpuSqrt(data, n);
        float[] amdResult = executeHipSqrt(data, n);
        assertArrayEquals(cpuResult, amdResult, tolerance, "AMD output doesn't match CPU reference");
    }

    private void compareTanhResults(float[] data, int n, float tolerance) {
        float[] cpuResult = executeCpuTanh(data, n);
        float[] amdResult = executeHipTanh(data, n);
        assertArrayEquals(cpuResult, amdResult, tolerance, "AMD output doesn't match CPU reference");
    }

    // ==================== CPU Backend Execution Helpers ====================

    private float[] executeCpuAdd(float[] aData, float[] bData, int n) {
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {
            StableHloAst.AddOp op = createAddOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(a, b));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuMultiply(float[] aData, float[] bData, int n) {
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {
            StableHloAst.MultiplyOp op = createMultiplyOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(a, b));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuSubtract(float[] aData, float[] bData, int n) {
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {
            StableHloAst.SubtractOp op = createSubtractOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(a, b));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuMaximum(float[] aData, float[] bData, int n) {
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {
            StableHloAst.MaximumOp op = createMaximumOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(a, b));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuMinimum(float[] aData, float[] bData, int n) {
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {
            StableHloAst.MinimumOp op = createMinimumOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(a, b));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuNegate(float[] data, int n) {
        try (Tensor input = Tensor.fromFloatArray(data, n)) {
            StableHloAst.NegateOp op = createNegateOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(input));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuAbs(float[] data, int n) {
        try (Tensor input = Tensor.fromFloatArray(data, n)) {
            StableHloAst.AbsOp op = createAbsOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(input));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuExp(float[] data, int n) {
        try (Tensor input = Tensor.fromFloatArray(data, n)) {
            StableHloAst.ExpOp op = createExpOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(input));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuSqrt(float[] data, int n) {
        try (Tensor input = Tensor.fromFloatArray(data, n)) {
            StableHloAst.SqrtOp op = createSqrtOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(input));
            return result.get(0).toFloatArray();
        }
    }

    private float[] executeCpuTanh(float[] data, int n) {
        try (Tensor input = Tensor.fromFloatArray(data, n)) {
            StableHloAst.TanhOp op = createTanhOp(n);
            List<Tensor> result = cpuBackend.execute(op, List.of(input));
            return result.get(0).toFloatArray();
        }
    }

    // ==================== HIP Kernel Execution Helpers ====================

    private float[] executeHipAdd(float[] aData, float[] bData, int n) {
        return executeHipBinaryOp(aData, bData, n, "add");
    }

    private float[] executeHipMultiply(float[] aData, float[] bData, int n) {
        return executeHipBinaryOp(aData, bData, n, "mul");
    }

    private float[] executeHipSubtract(float[] aData, float[] bData, int n) {
        return executeHipBinaryOp(aData, bData, n, "sub");
    }

    private float[] executeHipMaximum(float[] aData, float[] bData, int n) {
        return executeHipBinaryOp(aData, bData, n, "max");
    }

    private float[] executeHipMinimum(float[] aData, float[] bData, int n) {
        return executeHipBinaryOp(aData, bData, n, "min");
    }

    private float[] executeHipNegate(float[] data, int n) {
        return executeHipUnaryOp(data, n, "neg");
    }

    private float[] executeHipAbs(float[] data, int n) {
        return executeHipUnaryOp(data, n, "abs");
    }

    private float[] executeHipExp(float[] data, int n) {
        return executeHipUnaryOp(data, n, "exp");
    }

    private float[] executeHipSqrt(float[] data, int n) {
        return executeHipUnaryOp(data, n, "sqrt");
    }

    private float[] executeHipTanh(float[] data, int n) {
        return executeHipUnaryOp(data, n, "tanh");
    }

    private float[] executeHipBinaryOp(float[] aData, float[] bData, int n, String opName) {
        String source = switch (opName) {
            case "add" -> HipKernels.generateAddF32(HipKernels.SALT_NONE);
            case "mul" -> HipKernels.generateMultiplyF32(HipKernels.SALT_NONE);
            case "sub" -> HipKernels.generateSubtractF32(HipKernels.SALT_NONE);
            case "max" -> HipKernels.generateMaximumF32(HipKernels.SALT_NONE);
            case "min" -> HipKernels.generateMinimumF32(HipKernels.SALT_NONE);
            default -> throw new IllegalArgumentException("Unknown binary op: " + opName);
        };
        String functionName = switch (opName) {
            case "add" -> "add_f32";
            case "mul" -> "multiply_f32";
            case "sub" -> "subtract_f32";
            case "max" -> "maximum_f32";
            case "min" -> "minimum_f32";
            default -> throw new IllegalArgumentException("Unknown binary op: " + opName);
        };

        long byteSize = n * 4L;

        long module = hipContext.compileAndLoadModule(opName + "_module", source);
        long function = hipContext.getFunction(module, functionName);

        long dA = hipContext.allocate(byteSize);
        long dB = hipContext.allocate(byteSize);
        long dOut = hipContext.allocate(byteSize);

        try {
            try (Tensor tensorA = Tensor.fromFloatArray(aData, n);
                 Tensor tensorB = Tensor.fromFloatArray(bData, n)) {
                hipContext.copyToDevice(dA, tensorA.data());
                hipContext.copyToDevice(dB, tensorB.data());
            }

            hipContext.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dA, dB, dOut},
                n
            );

            hipContext.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                hipContext.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            hipContext.free(dA);
            hipContext.free(dB);
            hipContext.free(dOut);
        }
    }

    private float[] executeHipUnaryOp(float[] data, int n, String opName) {
        String source = switch (opName) {
            case "neg" -> HipKernels.generateNegateF32(HipKernels.SALT_NONE);
            case "abs" -> HipKernels.generateAbsF32(HipKernels.SALT_NONE);
            case "exp" -> HipKernels.generateExpF32(HipKernels.SALT_NONE);
            case "sqrt" -> HipKernels.generateSqrtF32(HipKernels.SALT_NONE);
            case "tanh" -> HipKernels.generateTanhF32(HipKernels.SALT_NONE);
            default -> throw new IllegalArgumentException("Unknown unary op: " + opName);
        };
        String functionName = switch (opName) {
            case "neg" -> "negate_f32";
            case "abs" -> "abs_f32";
            case "exp" -> "exp_f32";
            case "sqrt" -> "sqrt_f32";
            case "tanh" -> "tanh_f32";
            default -> throw new IllegalArgumentException("Unknown unary op: " + opName);
        };

        long byteSize = n * 4L;

        long module = hipContext.compileAndLoadModule(opName + "_module", source);
        long function = hipContext.getFunction(module, functionName);

        long dIn = hipContext.allocate(byteSize);
        long dOut = hipContext.allocate(byteSize);

        try {
            try (Tensor tensorIn = Tensor.fromFloatArray(data, n)) {
                hipContext.copyToDevice(dIn, tensorIn.data());
            }

            hipContext.launchKernelWithIntParams(
                function,
                new int[]{gridSize(n)}, new int[]{BLOCK_SIZE},
                0,
                new long[]{dIn, dOut},
                n
            );

            hipContext.synchronize();

            float[] result = new float[n];
            try (Tensor resultTensor = Tensor.fromFloatArray(result, n)) {
                hipContext.copyToHost(resultTensor.data(), dOut, byteSize);
                return resultTensor.toFloatArray();
            }

        } finally {
            hipContext.free(dIn);
            hipContext.free(dOut);
        }
    }

    // ==================== StableHLO Op Creation Helpers ====================

    private StableHloAst.AddOp createAddOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.AddOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private StableHloAst.MultiplyOp createMultiplyOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.MultiplyOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private StableHloAst.SubtractOp createSubtractOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.SubtractOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private StableHloAst.MaximumOp createMaximumOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.MaximumOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private StableHloAst.MinimumOp createMinimumOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.MinimumOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private StableHloAst.NegateOp createNegateOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.NegateOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            resultType
        );
    }

    private StableHloAst.AbsOp createAbsOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.AbsOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            resultType
        );
    }

    private StableHloAst.ExpOp createExpOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.ExpOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            resultType
        );
    }

    private StableHloAst.SqrtOp createSqrtOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.SqrtOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            resultType
        );
    }

    private StableHloAst.TanhOp createTanhOp(int n) {
        List<Integer> shape = List.of(n);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(shape, StableHloAst.ScalarType.F32);
        return new StableHloAst.TanhOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            resultType
        );
    }

    // ==================== Random Data Generation ====================

    private float[] generateRandomFloats(int n, long seed) {
        Random rng = new Random(seed);
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = rng.nextFloat() * 200 - 100; // Range: [-100, 100]
        }
        return data;
    }

    private float[] generateBoundedRandomFloats(int n, long seed, float min, float max) {
        Random rng = new Random(seed);
        float[] data = new float[n];
        float range = max - min;
        for (int i = 0; i < n; i++) {
            data[i] = min + rng.nextFloat() * range;
        }
        return data;
    }
}
