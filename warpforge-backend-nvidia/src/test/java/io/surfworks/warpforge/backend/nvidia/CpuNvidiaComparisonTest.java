package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.CpuBackend;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration tests comparing CPU and NVIDIA backend outputs.
 *
 * <p>The CPU backend is the source of truth for correctness.
 * These tests verify that the NVIDIA backend produces numerically
 * equivalent results within floating-point tolerance.
 *
 * <p>This validates the custom PTX implementation against the
 * reference Java implementation, NOT against cuBLAS (which would
 * introduce Heisenbug potential due to different FP rounding).
 */
@Tag("nvidia")
@DisplayName("CPU vs NVIDIA Backend Comparison")
class CpuNvidiaComparisonTest {

    private static final float TOLERANCE = 1e-5f;
    private static final long SEED = 42L;

    private CpuBackend cpuBackend;
    private NvidiaBackend nvidiaBackend;

    @BeforeEach
    void setUp() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");

        cpuBackend = new CpuBackend();
        nvidiaBackend = new NvidiaBackend(0, CudaKernels.SALT_NONE);

        assumeTrue(nvidiaBackend.hasCudaContext(), "CUDA context not available");
    }

    @AfterEach
    void tearDown() {
        if (cpuBackend != null) {
            cpuBackend.close();
        }
        if (nvidiaBackend != null) {
            nvidiaBackend.close();
        }
    }

    // ==================== Add Operation Tests ====================

    @Test
    @DisplayName("Add: Small tensor - exact match")
    void testAddSmallTensor() {
        float[] aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] bData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

        compareAddResults(aData, bData, 8);
    }

    @Test
    @DisplayName("Add: Medium tensor (10K elements)")
    void testAddMediumTensor() {
        int n = 10_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareAddResults(aData, bData, n);
    }

    @Test
    @DisplayName("Add: Large tensor (1M elements)")
    void testAddLargeTensor() {
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareAddResults(aData, bData, n);
    }

    @Test
    @DisplayName("Add: 2D tensor shape preserved")
    void testAdd2DShape() {
        float[] aData = generateRandomFloats(256, SEED);
        float[] bData = generateRandomFloats(256, SEED + 1);

        try (Tensor a = Tensor.fromFloatArray(aData, 16, 16);
             Tensor b = Tensor.fromFloatArray(bData, 16, 16)) {

            StableHloAst.AddOp addOp = createAddOp(16, 16);

            List<Tensor> cpuResult = cpuBackend.execute(addOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(addOp, List.of(a, b));

            assertArrayEquals(new int[]{16, 16}, cpuResult.get(0).shape());
            assertArrayEquals(new int[]{16, 16}, nvidiaResult.get(0).shape());
            assertArrayEquals(
                cpuResult.get(0).toFloatArray(),
                nvidiaResult.get(0).toFloatArray(),
                TOLERANCE
            );
        }
    }

    @Test
    @DisplayName("Add: 3D tensor shape preserved")
    void testAdd3DShape() {
        float[] aData = generateRandomFloats(8 * 16 * 32, SEED);
        float[] bData = generateRandomFloats(8 * 16 * 32, SEED + 1);

        try (Tensor a = Tensor.fromFloatArray(aData, 8, 16, 32);
             Tensor b = Tensor.fromFloatArray(bData, 8, 16, 32)) {

            StableHloAst.AddOp addOp = createAddOp(8, 16, 32);

            List<Tensor> cpuResult = cpuBackend.execute(addOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(addOp, List.of(a, b));

            assertArrayEquals(new int[]{8, 16, 32}, cpuResult.get(0).shape());
            assertArrayEquals(new int[]{8, 16, 32}, nvidiaResult.get(0).shape());
            assertArrayEquals(
                cpuResult.get(0).toFloatArray(),
                nvidiaResult.get(0).toFloatArray(),
                TOLERANCE
            );
        }
    }

    @Test
    @DisplayName("Add: Edge case - zeros")
    void testAddZeros() {
        float[] aData = new float[1024]; // All zeros
        float[] bData = generateRandomFloats(1024, SEED);

        compareAddResults(aData, bData, 1024);
    }

    @Test
    @DisplayName("Add: Edge case - negative numbers")
    void testAddNegatives() {
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = -aData[i]; // Result should be all zeros
        }

        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.AddOp addOp = createAddOp(n);

            List<Tensor> cpuResult = cpuBackend.execute(addOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(addOp, List.of(a, b));

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            for (int i = 0; i < n; i++) {
                assertEquals(cpuData[i], nvidiaData[i], TOLERANCE,
                    "Mismatch at index " + i);
            }
        }
    }

    @Test
    @DisplayName("Add: Edge case - very small numbers")
    void testAddSmallNumbers() {
        int n = 1024;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            aData[i] = 1e-30f * (i + 1);
            bData[i] = 1e-30f * (i + 1);
        }

        compareAddResults(aData, bData, n);
    }

    @Test
    @DisplayName("Add: Edge case - very large numbers")
    void testAddLargeNumbers() {
        int n = 1024;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            aData[i] = 1e30f + i;
            bData[i] = 1e30f + i;
        }

        // Use relative tolerance for large numbers
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.AddOp addOp = createAddOp(n);

            List<Tensor> cpuResult = cpuBackend.execute(addOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(addOp, List.of(a, b));

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            for (int i = 0; i < n; i++) {
                float relError = Math.abs(cpuData[i] - nvidiaData[i]) / Math.abs(cpuData[i]);
                assertEquals(0.0f, relError, 1e-6f,
                    "Relative error too large at index " + i);
            }
        }
    }

    @Test
    @DisplayName("Add with SALT_TIMING produces same results as SALT_NONE")
    void testAddTimingInstrumentationPreservesResults() {
        try (NvidiaBackend timedBackend = new NvidiaBackend(0, CudaKernels.SALT_TIMING)) {
            assumeTrue(timedBackend.hasCudaContext(), "CUDA context not available");

            int n = 10_000;
            float[] aData = generateRandomFloats(n, SEED);
            float[] bData = generateRandomFloats(n, SEED + 1);

            try (Tensor a = Tensor.fromFloatArray(aData, n);
                 Tensor b = Tensor.fromFloatArray(bData, n)) {

                StableHloAst.AddOp addOp = createAddOp(n);

                // Execute on CPU (reference)
                List<Tensor> cpuResult = cpuBackend.execute(addOp, List.of(a, b));

                // Execute on NVIDIA without timing
                List<Tensor> nvidiaNoTiming = nvidiaBackend.execute(addOp, List.of(a, b));

                // Execute on NVIDIA with timing
                List<Tensor> nvidiaWithTiming = timedBackend.execute(addOp, List.of(a, b));

                // All three should match
                float[] cpuData = cpuResult.get(0).toFloatArray();
                float[] nvidiaNoTimingData = nvidiaNoTiming.get(0).toFloatArray();
                float[] nvidiaWithTimingData = nvidiaWithTiming.get(0).toFloatArray();

                assertArrayEquals(cpuData, nvidiaNoTimingData, TOLERANCE,
                    "NVIDIA (no timing) doesn't match CPU");
                assertArrayEquals(cpuData, nvidiaWithTimingData, TOLERANCE,
                    "NVIDIA (with timing) doesn't match CPU");
                assertArrayEquals(nvidiaNoTimingData, nvidiaWithTimingData, 0.0f,
                    "NVIDIA timing variant doesn't match non-timing variant exactly");
            }
        }
    }

    // ==================== Multiply Operation Tests ====================

    @Test
    @DisplayName("Multiply: Small tensor - exact match")
    void testMultiplySmallTensor() {
        float[] aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] bData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

        compareMultiplyResults(aData, bData, 8);
    }

    @Test
    @DisplayName("Multiply: Medium tensor (10K elements)")
    void testMultiplyMediumTensor() {
        int n = 10_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareMultiplyResults(aData, bData, n);
    }

    @Test
    @DisplayName("Multiply: Large tensor (1M elements)")
    void testMultiplyLargeTensor() {
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareMultiplyResults(aData, bData, n);
    }

    @Test
    @DisplayName("Multiply: 2D tensor shape preserved")
    void testMultiply2DShape() {
        float[] aData = generateRandomFloats(256, SEED);
        float[] bData = generateRandomFloats(256, SEED + 1);

        try (Tensor a = Tensor.fromFloatArray(aData, 16, 16);
             Tensor b = Tensor.fromFloatArray(bData, 16, 16)) {

            StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(16, 16);

            List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(multiplyOp, List.of(a, b));

            assertArrayEquals(new int[]{16, 16}, cpuResult.get(0).shape());
            assertArrayEquals(new int[]{16, 16}, nvidiaResult.get(0).shape());
            assertArrayEquals(
                cpuResult.get(0).toFloatArray(),
                nvidiaResult.get(0).toFloatArray(),
                TOLERANCE
            );
        }
    }

    @Test
    @DisplayName("Multiply: 3D tensor shape preserved")
    void testMultiply3DShape() {
        float[] aData = generateRandomFloats(8 * 16 * 32, SEED);
        float[] bData = generateRandomFloats(8 * 16 * 32, SEED + 1);

        try (Tensor a = Tensor.fromFloatArray(aData, 8, 16, 32);
             Tensor b = Tensor.fromFloatArray(bData, 8, 16, 32)) {

            StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(8, 16, 32);

            List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(multiplyOp, List.of(a, b));

            assertArrayEquals(new int[]{8, 16, 32}, cpuResult.get(0).shape());
            assertArrayEquals(new int[]{8, 16, 32}, nvidiaResult.get(0).shape());
            assertArrayEquals(
                cpuResult.get(0).toFloatArray(),
                nvidiaResult.get(0).toFloatArray(),
                TOLERANCE
            );
        }
    }

    @Test
    @DisplayName("Multiply: Edge case - zeros")
    void testMultiplyZeros() {
        float[] aData = new float[1024]; // All zeros
        float[] bData = generateRandomFloats(1024, SEED);

        compareMultiplyResults(aData, bData, 1024);
    }

    @Test
    @DisplayName("Multiply: Edge case - ones (identity)")
    void testMultiplyOnes() {
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = 1.0f;
        }

        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(n);

            List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(multiplyOp, List.of(a, b));

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            // Multiplying by 1 should return the original values
            assertArrayEquals(aData, cpuData, TOLERANCE);
            assertArrayEquals(aData, nvidiaData, TOLERANCE);
        }
    }

    @Test
    @DisplayName("Multiply: Edge case - negative numbers")
    void testMultiplyNegatives() {
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = -1.0f; // Negate all values
        }

        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(n);

            List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(multiplyOp, List.of(a, b));

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            for (int i = 0; i < n; i++) {
                assertEquals(cpuData[i], nvidiaData[i], TOLERANCE,
                    "Mismatch at index " + i);
                // Result should be negated original
                assertEquals(-aData[i], cpuData[i], TOLERANCE,
                    "CPU result at index " + i + " should be negated");
            }
        }
    }

    @Test
    @DisplayName("Multiply: Edge case - very small numbers")
    void testMultiplySmallNumbers() {
        int n = 1024;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            aData[i] = 1e-15f * (i + 1);
            bData[i] = 1e-15f * (i + 1);
        }

        compareMultiplyResults(aData, bData, n);
    }

    @Test
    @DisplayName("Multiply: Edge case - mixed small and large")
    void testMultiplyMixedMagnitudes() {
        int n = 1024;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            aData[i] = 1e10f + i;
            bData[i] = 1e-10f;
        }

        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(n);

            List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(multiplyOp, List.of(a, b));

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            for (int i = 0; i < n; i++) {
                float relError = Math.abs(cpuData[i] - nvidiaData[i]) / Math.max(Math.abs(cpuData[i]), 1e-10f);
                assertEquals(0.0f, relError, 1e-5f,
                    "Relative error too large at index " + i);
            }
        }
    }

    @Test
    @DisplayName("Multiply with SALT_TIMING produces same results as SALT_NONE")
    void testMultiplyTimingInstrumentationPreservesResults() {
        try (NvidiaBackend timedBackend = new NvidiaBackend(0, CudaKernels.SALT_TIMING)) {
            assumeTrue(timedBackend.hasCudaContext(), "CUDA context not available");

            int n = 10_000;
            float[] aData = generateRandomFloats(n, SEED);
            float[] bData = generateRandomFloats(n, SEED + 1);

            try (Tensor a = Tensor.fromFloatArray(aData, n);
                 Tensor b = Tensor.fromFloatArray(bData, n)) {

                StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(n);

                // Execute on CPU (reference)
                List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));

                // Execute on NVIDIA without timing
                List<Tensor> nvidiaNoTiming = nvidiaBackend.execute(multiplyOp, List.of(a, b));

                // Execute on NVIDIA with timing
                List<Tensor> nvidiaWithTiming = timedBackend.execute(multiplyOp, List.of(a, b));

                // All three should match
                float[] cpuData = cpuResult.get(0).toFloatArray();
                float[] nvidiaNoTimingData = nvidiaNoTiming.get(0).toFloatArray();
                float[] nvidiaWithTimingData = nvidiaWithTiming.get(0).toFloatArray();

                assertArrayEquals(cpuData, nvidiaNoTimingData, TOLERANCE,
                    "NVIDIA (no timing) doesn't match CPU");
                assertArrayEquals(cpuData, nvidiaWithTimingData, TOLERANCE,
                    "NVIDIA (with timing) doesn't match CPU");
                assertArrayEquals(nvidiaNoTimingData, nvidiaWithTimingData, 0.0f,
                    "NVIDIA timing variant doesn't match non-timing variant exactly");
            }
        }
    }

    // ==================== Subtract Operation Tests ====================

    @Test
    @DisplayName("Subtract: CPU vs NVIDIA - 1M elements")
    void testSubtractLargeTensor() {
        System.out.println("[TEST] CPU vs NVIDIA: Subtract (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareBinaryOpResults(aData, bData, n, StableHloAst.SubtractOp.class, "Subtract");
        System.out.println("[PASS] Subtract: CPU vs NVIDIA match");
    }

    @Test
    @DisplayName("Subtract: Edge cases")
    void testSubtractEdgeCases() {
        System.out.println("[TEST] CPU vs NVIDIA: Subtract edge cases");
        int n = 1024;

        // Same values -> zeros
        float[] aData = generateRandomFloats(n, SEED);
        compareBinaryOpResults(aData, aData.clone(), n, StableHloAst.SubtractOp.class, "Subtract (a-a)");

        // Negatives
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = -aData[i];
        }
        compareBinaryOpResults(aData, bData, n, StableHloAst.SubtractOp.class, "Subtract (a-(-a))");

        System.out.println("[PASS] Subtract edge cases OK");
    }

    // ==================== Divide Operation Tests ====================

    @Test
    @DisplayName("Divide: CPU vs NVIDIA - 1M elements")
    void testDivideLargeTensor() {
        System.out.println("[TEST] CPU vs NVIDIA: Divide (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        // Avoid division by zero - ensure all b values are non-zero
        for (int i = 0; i < n; i++) {
            if (Math.abs(bData[i]) < 0.01f) {
                bData[i] = 1.0f;
            }
        }

        // Use looser tolerance for divide (div.approx.f32)
        compareBinaryOpResultsWithTolerance(aData, bData, n, StableHloAst.DivideOp.class, "Divide", 1e-3f);
        System.out.println("[PASS] Divide: CPU vs NVIDIA match");
    }

    @Test
    @DisplayName("Divide: Identity (divide by 1)")
    void testDivideByOne() {
        System.out.println("[TEST] CPU vs NVIDIA: Divide by 1");
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            bData[i] = 1.0f;
        }

        compareBinaryOpResultsWithTolerance(aData, bData, n, StableHloAst.DivideOp.class, "Divide by 1", 1e-4f);
        System.out.println("[PASS] Divide by 1 OK");
    }

    // ==================== Maximum Operation Tests ====================

    @Test
    @DisplayName("Maximum: CPU vs NVIDIA - 1M elements")
    void testMaximumLargeTensor() {
        System.out.println("[TEST] CPU vs NVIDIA: Maximum (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareBinaryOpResults(aData, bData, n, StableHloAst.MaximumOp.class, "Maximum");
        System.out.println("[PASS] Maximum: CPU vs NVIDIA match");
    }

    @Test
    @DisplayName("Maximum: Same values")
    void testMaximumSameValues() {
        System.out.println("[TEST] CPU vs NVIDIA: Maximum with same values");
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);

        compareBinaryOpResults(aData, aData.clone(), n, StableHloAst.MaximumOp.class, "Maximum (same)");
        System.out.println("[PASS] Maximum same values OK");
    }

    @Test
    @DisplayName("Maximum: Negative numbers")
    void testMaximumNegatives() {
        System.out.println("[TEST] CPU vs NVIDIA: Maximum with negatives");
        int n = 1024;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            aData[i] = -100.0f + i * 0.1f;
            bData[i] = -50.0f - i * 0.1f;
        }

        compareBinaryOpResults(aData, bData, n, StableHloAst.MaximumOp.class, "Maximum (neg)");
        System.out.println("[PASS] Maximum negatives OK");
    }

    // ==================== Minimum Operation Tests ====================

    @Test
    @DisplayName("Minimum: CPU vs NVIDIA - 1M elements")
    void testMinimumLargeTensor() {
        System.out.println("[TEST] CPU vs NVIDIA: Minimum (1M elements)");
        int n = 1_000_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        compareBinaryOpResults(aData, bData, n, StableHloAst.MinimumOp.class, "Minimum");
        System.out.println("[PASS] Minimum: CPU vs NVIDIA match");
    }

    @Test
    @DisplayName("Minimum: Same values")
    void testMinimumSameValues() {
        System.out.println("[TEST] CPU vs NVIDIA: Minimum with same values");
        int n = 1024;
        float[] aData = generateRandomFloats(n, SEED);

        compareBinaryOpResults(aData, aData.clone(), n, StableHloAst.MinimumOp.class, "Minimum (same)");
        System.out.println("[PASS] Minimum same values OK");
    }

    @Test
    @DisplayName("Minimum: Negative numbers")
    void testMinimumNegatives() {
        System.out.println("[TEST] CPU vs NVIDIA: Minimum with negatives");
        int n = 1024;
        float[] aData = new float[n];
        float[] bData = new float[n];
        for (int i = 0; i < n; i++) {
            aData[i] = -100.0f + i * 0.1f;
            bData[i] = -50.0f - i * 0.1f;
        }

        compareBinaryOpResults(aData, bData, n, StableHloAst.MinimumOp.class, "Minimum (neg)");
        System.out.println("[PASS] Minimum negatives OK");
    }

    // ==================== Combined Summary Test ====================

    @Test
    @DisplayName("All Binary Elementwise: Summary test")
    void testAllBinaryElementwiseSummary() {
        System.out.println("\n========================================");
        System.out.println("Binary Elementwise Operations Summary");
        System.out.println("========================================");

        int n = 10_000;
        float[] aData = generateRandomFloats(n, SEED);
        float[] bData = generateRandomFloats(n, SEED + 1);

        // Ensure no zeros for divide
        for (int i = 0; i < n; i++) {
            if (Math.abs(bData[i]) < 0.01f) {
                bData[i] = 1.0f;
            }
        }

        String[] ops = {"Add", "Multiply", "Subtract", "Divide", "Maximum", "Minimum"};
        Class<?>[] opClasses = {
            StableHloAst.AddOp.class,
            StableHloAst.MultiplyOp.class,
            StableHloAst.SubtractOp.class,
            StableHloAst.DivideOp.class,
            StableHloAst.MaximumOp.class,
            StableHloAst.MinimumOp.class
        };

        for (int i = 0; i < ops.length; i++) {
            try {
                float tolerance = ops[i].equals("Divide") ? 1e-3f : TOLERANCE;
                @SuppressWarnings("unchecked")
                Class<? extends StableHloAst.Operation> opClass =
                    (Class<? extends StableHloAst.Operation>) opClasses[i];
                compareBinaryOpResultsWithTolerance(aData, bData, n, opClass, ops[i], tolerance);
                System.out.println("  [OK] " + ops[i] + ": CPU == NVIDIA");
            } catch (Exception e) {
                System.out.println("  [FAIL] " + ops[i] + ": " + e.getMessage());
                throw e;
            }
        }

        System.out.println("========================================");
        System.out.println("All 6 binary elementwise operations PASSED");
        System.out.println("========================================\n");
    }

    // ==================== Helper Methods ====================

    private void compareBinaryOpResults(float[] aData, float[] bData, int n,
                                         Class<? extends StableHloAst.Operation> opClass,
                                         String opName) {
        compareBinaryOpResultsWithTolerance(aData, bData, n, opClass, opName, TOLERANCE);
    }

    private void compareBinaryOpResultsWithTolerance(float[] aData, float[] bData, int n,
                                                      Class<? extends StableHloAst.Operation> opClass,
                                                      String opName, float tolerance) {
        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.Operation op = createBinaryOp(opClass, n);

            List<Tensor> cpuResult = cpuBackend.execute(op, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(op, List.of(a, b));

            assertEquals(1, cpuResult.size(), opName + ": CPU should return 1 tensor");
            assertEquals(1, nvidiaResult.size(), opName + ": NVIDIA should return 1 tensor");

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            assertEquals(cpuData.length, nvidiaData.length, opName + ": Output sizes don't match");
            assertArrayEquals(cpuData, nvidiaData, tolerance,
                opName + ": NVIDIA output doesn't match CPU reference");
        }
    }

    private StableHloAst.Operation createBinaryOp(Class<? extends StableHloAst.Operation> opClass, int... shape) {
        List<Integer> shapeList = new java.util.ArrayList<>();
        for (int dim : shape) {
            shapeList.add(dim);
        }

        StableHloAst.TensorType resultType = new StableHloAst.TensorType(
            shapeList,
            StableHloAst.ScalarType.F32
        );

        StableHloAst.Value lhs = new StableHloAst.Value("0", resultType);
        StableHloAst.Value rhs = new StableHloAst.Value("1", resultType);
        StableHloAst.Value result = new StableHloAst.Value("2", resultType);

        if (opClass == StableHloAst.AddOp.class) {
            return new StableHloAst.AddOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.MultiplyOp.class) {
            return new StableHloAst.MultiplyOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.SubtractOp.class) {
            return new StableHloAst.SubtractOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.DivideOp.class) {
            return new StableHloAst.DivideOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.MaximumOp.class) {
            return new StableHloAst.MaximumOp(lhs, rhs, result, resultType);
        } else if (opClass == StableHloAst.MinimumOp.class) {
            return new StableHloAst.MinimumOp(lhs, rhs, result, resultType);
        } else {
            throw new IllegalArgumentException("Unknown operation class: " + opClass);
        }
    }

    private void compareAddResults(float[] aData, float[] bData, int... shape) {
        int n = aData.length;

        try (Tensor a = Tensor.fromFloatArray(aData, shape);
             Tensor b = Tensor.fromFloatArray(bData, shape)) {

            StableHloAst.AddOp addOp = createAddOp(shape);

            List<Tensor> cpuResult = cpuBackend.execute(addOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(addOp, List.of(a, b));

            assertEquals(1, cpuResult.size());
            assertEquals(1, nvidiaResult.size());

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            assertEquals(cpuData.length, nvidiaData.length, "Output sizes don't match");
            assertArrayEquals(cpuData, nvidiaData, TOLERANCE,
                "NVIDIA output doesn't match CPU reference");
        }
    }

    private void compareMultiplyResults(float[] aData, float[] bData, int... shape) {
        int n = aData.length;

        try (Tensor a = Tensor.fromFloatArray(aData, shape);
             Tensor b = Tensor.fromFloatArray(bData, shape)) {

            StableHloAst.MultiplyOp multiplyOp = createMultiplyOp(shape);

            List<Tensor> cpuResult = cpuBackend.execute(multiplyOp, List.of(a, b));
            List<Tensor> nvidiaResult = nvidiaBackend.execute(multiplyOp, List.of(a, b));

            assertEquals(1, cpuResult.size());
            assertEquals(1, nvidiaResult.size());

            float[] cpuData = cpuResult.get(0).toFloatArray();
            float[] nvidiaData = nvidiaResult.get(0).toFloatArray();

            assertEquals(cpuData.length, nvidiaData.length, "Output sizes don't match");
            assertArrayEquals(cpuData, nvidiaData, TOLERANCE,
                "NVIDIA output doesn't match CPU reference");
        }
    }

    private StableHloAst.AddOp createAddOp(int... shape) {
        List<Integer> shapeList = new java.util.ArrayList<>();
        for (int dim : shape) {
            shapeList.add(dim);
        }

        StableHloAst.TensorType resultType = new StableHloAst.TensorType(
            shapeList,
            StableHloAst.ScalarType.F32
        );

        return new StableHloAst.AddOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private StableHloAst.MultiplyOp createMultiplyOp(int... shape) {
        List<Integer> shapeList = new java.util.ArrayList<>();
        for (int dim : shape) {
            shapeList.add(dim);
        }

        StableHloAst.TensorType resultType = new StableHloAst.TensorType(
            shapeList,
            StableHloAst.ScalarType.F32
        );

        return new StableHloAst.MultiplyOp(
            new StableHloAst.Value("0", resultType),
            new StableHloAst.Value("1", resultType),
            new StableHloAst.Value("2", resultType),
            resultType
        );
    }

    private float[] generateRandomFloats(int n, long seed) {
        Random rng = new Random(seed);
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = rng.nextFloat() * 200 - 100; // Range: [-100, 100]
        }
        return data;
    }
}
