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

    // ==================== Helper Methods ====================

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

    private float[] generateRandomFloats(int n, long seed) {
        Random rng = new Random(seed);
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = rng.nextFloat() * 200 - 100; // Range: [-100, 100]
        }
        return data;
    }
}
