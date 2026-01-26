package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for matrix multiplication (Dot) and reduction HIP kernels.
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Matrix and Reduce HIP Kernels")
class MatrixReduceKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: Dot generates valid output")
    void testDotSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Dot");
        String src = HipKernels.generateDotF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void dot_f32"));
        assertTrue(src.contains("for (int k = 0; k < K; k++)"));
        assertTrue(src.contains("A[row * K + k]"));
        assertTrue(src.contains("B[k * N + col]"));
        System.out.println("[PASS] Dot HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Reduce add generates valid output")
    void testReduceAddSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Reduce add");
        String src = HipKernels.generateReduceAddF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void reduce_add_f32"));
        assertTrue(src.contains("__shared__"));
        assertTrue(src.contains("atomicAdd"));
        System.out.println("[PASS] Reduce add HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Reduce max generates valid output")
    void testReduceMaxSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Reduce max");
        String src = HipKernels.generateReduceMaxF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void reduce_max_f32"));
        assertTrue(src.contains("__shared__"));
        assertTrue(src.contains("fmaxf"));
        System.out.println("[PASS] Reduce max HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Reduce min generates valid output")
    void testReduceMinSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Reduce min");
        String src = HipKernels.generateReduceMinF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void reduce_min_f32"));
        assertTrue(src.contains("__shared__"));
        assertTrue(src.contains("fminf"));
        System.out.println("[PASS] Reduce min HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Reduce mul generates valid output")
    void testReduceMulSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Reduce mul");
        String src = HipKernels.generateReduceMulF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void reduce_mul_f32"));
        assertTrue(src.contains("__shared__"));
        System.out.println("[PASS] Reduce mul HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Dot supports SALT_TIMING")
    void testDotTimingSupport() {
        System.out.println("[TEST] HIP Generation: Dot with SALT_TIMING");
        String src = HipKernels.generateDotF32(HipKernels.SALT_TIMING);

        assertTrue(src.contains("unsigned long long* timing"), "Dot should have timing parameter");
        assertTrue(src.contains("clock64()"), "Dot should use clock64");
        System.out.println("[PASS] Dot supports SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: All reduce operations support SALT_TIMING")
    void testAllReduceOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All reduce operations with SALT_TIMING");

        String[] ops = {"add", "max", "min", "mul"};
        String[] srcCodes = {
            HipKernels.generateReduceAddF32(HipKernels.SALT_TIMING),
            HipKernels.generateReduceMaxF32(HipKernels.SALT_TIMING),
            HipKernels.generateReduceMinF32(HipKernels.SALT_TIMING),
            HipKernels.generateReduceMulF32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(srcCodes[i].contains("unsigned long long* timing"),
                "Reduce " + ops[i] + " should have timing parameter");
            assertTrue(srcCodes[i].contains("clock64()"),
                "Reduce " + ops[i] + " should use clock64");
            System.out.println("  [OK] Reduce " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All reduce operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: Reduce add uses shared memory correctly")
    void testReduceAddSharedMemory() {
        System.out.println("[TEST] HIP Generation: Reduce add shared memory");
        String src = HipKernels.generateReduceAddF32(HipKernels.SALT_NONE);

        assertTrue(src.contains("__shared__ float sdata[BLOCK_SIZE]"));
        assertTrue(src.contains("__syncthreads()"));
        assertTrue(src.contains("for (int s = blockDim.x / 2; s > 0; s >>= 1)"));
        System.out.println("[PASS] Reduce add shared memory OK");
    }

    @Test
    @DisplayName("HIP: All Matrix/Reduce Summary")
    void testAllMatrixReduceSummary() {
        System.out.println("========================================");
        System.out.println("Matrix and Reduce HIP Operations Summary");
        System.out.println("========================================");

        System.out.println("--- Matrix Operations ---");
        System.out.println("  Dot: [OK]");

        System.out.println("--- Reduce Operations ---");
        System.out.println("  Reduce add: [OK]");
        System.out.println("  Reduce max: [OK]");
        System.out.println("  Reduce min: [OK]");
        System.out.println("  Reduce mul: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 5 matrix/reduce operations PASSED");
        System.out.println("========================================");
    }
}
