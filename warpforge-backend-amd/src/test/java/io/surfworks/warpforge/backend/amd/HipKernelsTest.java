package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for HIP kernel source generation.
 *
 * <p>These tests verify that HIP C++ source code is correctly generated
 * with appropriate instrumentation for each salt level.
 *
 * <p>No hardware is required for these tests.
 */
@DisplayName("HIP Kernel Generation Tests")
class HipKernelsTest {

    // ==================== Add Kernel Generation ====================

    @Test
    @DisplayName("Add kernel generation produces valid HIP C++ for SALT_NONE")
    void testAddKernelNoSalt() {
        String src = HipKernels.generateAddF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("#include <hip/hip_runtime.h>"));
        assertTrue(src.contains("extern \"C\" __global__ void add_f32"));
        assertTrue(src.contains("const float* __restrict__ a"));
        assertTrue(src.contains("const float* __restrict__ b"));
        assertTrue(src.contains("float* __restrict__ out"));
        assertTrue(src.contains("int n"));
        assertTrue(src.contains("out[i] = a[i] + b[i]"));

        // Should NOT contain timing parameter
        assertTrue(!src.contains("unsigned long long* timing"));
        assertTrue(!src.contains("[SALT_TIMING]"));
    }

    @Test
    @DisplayName("Add kernel generation includes timing for SALT_TIMING")
    void testAddKernelWithTiming() {
        String src = HipKernels.generateAddF32(HipKernels.SALT_TIMING);

        assertNotNull(src);
        assertTrue(src.contains("extern \"C\" __global__ void add_f32"));

        // Should contain timing instrumentation
        assertTrue(src.contains("unsigned long long* timing"));
        assertTrue(src.contains("clock64()"));
        assertTrue(src.contains("atomicAdd(timing"));
        assertTrue(src.contains("[SALT_TIMING]"));
    }

    @Test
    @DisplayName("Add kernel includes salt level in comment")
    void testAddKernelSaltComment() {
        String srcNone = HipKernels.generateAddF32(HipKernels.SALT_NONE);
        String srcTiming = HipKernels.generateAddF32(HipKernels.SALT_TIMING);
        String srcTrace = HipKernels.generateAddF32(HipKernels.SALT_TRACE);

        assertTrue(srcNone.contains("Salt level: 0"));
        assertTrue(srcTiming.contains("Salt level: 1"));
        assertTrue(srcTrace.contains("Salt level: 2"));
    }

    // ==================== Multiply Kernel Generation ====================

    @Test
    @DisplayName("Multiply kernel generation produces valid HIP C++ for SALT_NONE")
    void testMultiplyKernelNoSalt() {
        String src = HipKernels.generateMultiplyF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("#include <hip/hip_runtime.h>"));
        assertTrue(src.contains("extern \"C\" __global__ void multiply_f32"));
        assertTrue(src.contains("out[i] = a[i] * b[i]"));

        // Should NOT contain timing
        assertTrue(!src.contains("unsigned long long* timing"));
    }

    @Test
    @DisplayName("Multiply kernel generation includes timing for SALT_TIMING")
    void testMultiplyKernelWithTiming() {
        String src = HipKernels.generateMultiplyF32(HipKernels.SALT_TIMING);

        assertNotNull(src);
        assertTrue(src.contains("unsigned long long* timing"));
        assertTrue(src.contains("clock64()"));
        assertTrue(src.contains("atomicAdd(timing"));
    }

    // ==================== Dot Kernel Generation ====================

    @Test
    @DisplayName("Dot kernel generation produces valid HIP C++ for SALT_NONE")
    void testDotKernelNoSalt() {
        String src = HipKernels.generateDotF32(HipKernels.SALT_NONE);

        assertNotNull(src);
        assertTrue(src.contains("#include <hip/hip_runtime.h>"));
        assertTrue(src.contains("extern \"C\" __global__ void dot_f32"));
        assertTrue(src.contains("const float* __restrict__ A"));
        assertTrue(src.contains("const float* __restrict__ B"));
        assertTrue(src.contains("float* __restrict__ C"));
        assertTrue(src.contains("int M, int N, int K"));

        // Should have the naive matrix multiply loop
        assertTrue(src.contains("for (int k = 0; k < K; k++)"));
        assertTrue(src.contains("A[row * K + k] * B[k * N + col]"));

        // Should NOT contain timing
        assertTrue(!src.contains("unsigned long long* timing"));
    }

    @Test
    @DisplayName("Dot kernel generation includes timing for SALT_TIMING")
    void testDotKernelWithTiming() {
        String src = HipKernels.generateDotF32(HipKernels.SALT_TIMING);

        assertNotNull(src);
        assertTrue(src.contains("unsigned long long* timing"));
        assertTrue(src.contains("clock64()"));
        assertTrue(src.contains("atomicAdd(timing"));
    }

    @Test
    @DisplayName("Dot kernel notes it is CORRECTNESS tier")
    void testDotKernelCorrectnessComment() {
        String src = HipKernels.generateDotF32(HipKernels.SALT_NONE);

        assertTrue(src.contains("CORRECTNESS tier"));
        assertTrue(src.contains("use rocBLAS for PRODUCTION"));
    }

    // ==================== Grid Size Calculation ====================

    @Test
    @DisplayName("Grid size calculation is correct for 1D")
    void testGridSizeCalculation() {
        assertEquals(1, HipKernels.calculateGridSize(1));
        assertEquals(1, HipKernels.calculateGridSize(256));
        assertEquals(2, HipKernels.calculateGridSize(257));
        assertEquals(4, HipKernels.calculateGridSize(1000));
        assertEquals(4, HipKernels.calculateGridSize(1024));
        assertEquals(5, HipKernels.calculateGridSize(1025));
    }

    @Test
    @DisplayName("Grid size calculation is correct for 2D")
    void testGridSize2DCalculation() {
        int[] grid1 = HipKernels.calculateGridSize2D(16, 16);
        assertEquals(1, grid1[0]);
        assertEquals(1, grid1[1]);

        int[] grid2 = HipKernels.calculateGridSize2D(32, 64);
        assertEquals(4, grid2[0]);  // 64 / 16 = 4
        assertEquals(2, grid2[1]);  // 32 / 16 = 2

        int[] grid3 = HipKernels.calculateGridSize2D(100, 200);
        assertEquals(13, grid3[0]); // ceil(200/16) = 13
        assertEquals(7, grid3[1]);  // ceil(100/16) = 7
    }

    // ==================== Salt Constants ====================

    @Test
    @DisplayName("Salt constants are correctly defined")
    void testSaltConstants() {
        assertEquals(0, HipKernels.SALT_NONE);
        assertEquals(1, HipKernels.SALT_TIMING);
        assertEquals(2, HipKernels.SALT_TRACE);
    }

    @Test
    @DisplayName("Block sizes are reasonable")
    void testBlockSizes() {
        assertEquals(256, HipKernels.ELEMENTWISE_BLOCK_SIZE);
        assertEquals(16, HipKernels.DOT_BLOCK_SIZE);
    }
}
