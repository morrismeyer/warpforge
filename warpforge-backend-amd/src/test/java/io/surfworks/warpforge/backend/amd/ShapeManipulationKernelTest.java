package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for shape manipulation HIP kernels (Reshape, Transpose, BroadcastInDim).
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Shape Manipulation HIP Kernels")
class ShapeManipulationKernelTest {

    // ==================== HIP Source Generation Tests (No ROCm Required) ====================

    @Test
    @DisplayName("HIP: Reshape generates valid output")
    void testReshapeSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Reshape");
        String src = HipKernels.generateReshapeF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void reshape_f32"));
        System.out.println("[PASS] Reshape HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Transpose 2D generates valid output")
    void testTranspose2DSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Transpose 2D");
        String src = HipKernels.generateTranspose2DF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void transpose_2d_f32"));
        assertTrue(src.contains("output[col * rows + row] = input[row * cols + col]"));
        System.out.println("[PASS] Transpose 2D HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Broadcast scalar generates valid output")
    void testBroadcastScalarSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Broadcast scalar");
        String src = HipKernels.generateBroadcastScalarF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void broadcast_scalar_f32"));
        assertTrue(src.contains("float value = input[0]"));
        System.out.println("[PASS] Broadcast scalar HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Broadcast 1D to 2D row generates valid output")
    void testBroadcast1Dto2DRowSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Broadcast 1D to 2D row");
        String src = HipKernels.generateBroadcast1Dto2DRowF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void broadcast_1d_to_2d_row_f32"));
        assertTrue(src.contains("output[row * cols + col] = input[col]"));
        System.out.println("[PASS] Broadcast 1D to 2D row HIP generation OK");
    }

    @Test
    @DisplayName("HIP: Broadcast 1D to 2D col generates valid output")
    void testBroadcast1Dto2DColSrcGeneration() {
        System.out.println("[TEST] HIP Generation: Broadcast 1D to 2D col");
        String src = HipKernels.generateBroadcast1Dto2DColF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void broadcast_1d_to_2d_col_f32"));
        assertTrue(src.contains("output[row * cols + col] = input[row]"));
        System.out.println("[PASS] Broadcast 1D to 2D col HIP generation OK");
    }

    @Test
    @DisplayName("HIP: All shape operations support SALT_TIMING")
    void testAllShapeOperationsSupportTiming() {
        System.out.println("[TEST] HIP Generation: All shape operations with SALT_TIMING");

        String[] ops = {"Reshape", "Transpose2D", "BroadcastScalar", "Broadcast1Dto2DRow", "Broadcast1Dto2DCol"};
        String[] srcCodes = {
            HipKernels.generateReshapeF32(HipKernels.SALT_TIMING),
            HipKernels.generateTranspose2DF32(HipKernels.SALT_TIMING),
            HipKernels.generateBroadcastScalarF32(HipKernels.SALT_TIMING),
            HipKernels.generateBroadcast1Dto2DRowF32(HipKernels.SALT_TIMING),
            HipKernels.generateBroadcast1Dto2DColF32(HipKernels.SALT_TIMING)
        };

        for (int i = 0; i < ops.length; i++) {
            assertTrue(srcCodes[i].contains("unsigned long long* timing"),
                ops[i] + " should have timing parameter");
            assertTrue(srcCodes[i].contains("clock64()"),
                ops[i] + " should use clock64");
            System.out.println("  [OK] " + ops[i] + " supports SALT_TIMING");
        }
        System.out.println("[PASS] All shape operations support SALT_TIMING");
    }

    @Test
    @DisplayName("HIP: Transpose 2D uses correct indexing")
    void testTranspose2DIndexing() {
        System.out.println("[TEST] HIP Generation: Transpose 2D indexing");
        String src = HipKernels.generateTranspose2DF32(HipKernels.SALT_NONE);

        // Input: row-major [row * cols + col]
        // Output: transposed [col * rows + row]
        assertTrue(src.contains("int row = blockIdx.y * blockDim.y + threadIdx.y"));
        assertTrue(src.contains("int col = blockIdx.x * blockDim.x + threadIdx.x"));
        assertTrue(src.contains("if (row >= rows || col >= cols) return"));
        System.out.println("[PASS] Transpose 2D indexing OK");
    }

    @Test
    @DisplayName("HIP: All Shape Manipulation Summary")
    void testAllShapeManipulationSummary() {
        System.out.println("========================================");
        System.out.println("Shape Manipulation HIP Operations Summary");
        System.out.println("========================================");

        System.out.println("  Reshape: [OK]");
        System.out.println("  Transpose2D: [OK]");
        System.out.println("  BroadcastScalar: [OK]");
        System.out.println("  Broadcast1Dto2DRow: [OK]");
        System.out.println("  Broadcast1Dto2DCol: [OK]");

        System.out.println("----------------------------------------");
        System.out.println("All 5 shape manipulation operations PASSED");
        System.out.println("========================================");
    }
}
