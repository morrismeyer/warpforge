package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipKernels;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for HIP CustomCall kernels (transformer operations).
 *
 * <p>Tests HIP C++ source generation for:
 * <ul>
 *   <li>GELU activation</li>
 *   <li>SiLU activation</li>
 *   <li>Softmax normalization</li>
 *   <li>LayerNorm normalization</li>
 *   <li>Embedding lookup</li>
 * </ul>
 *
 * <p>Note: Actual execution requires HIPRTC integration. These tests verify
 * the HIP C++ source generation is correct.
 */
@DisplayName("CustomCall HIP Kernels (Transformer Ops)")
class HipCustomCallKernelTest {

    // ==================== HIP C++ Source Generation Tests ====================

    @Test
    @DisplayName("HIP C++: GELU generates valid source")
    void testGeluSourceGeneration() {
        System.out.println("[TEST] HIP C++ Generation: GELU");
        String source = HipKernels.generateGeluF32(HipKernels.SALT_NONE);

        assertNotNull(source);
        assertTrue(source.contains("gelu_f32"), "Should have gelu_f32 kernel");
        assertTrue(source.contains("__global__"), "Should be a global kernel");
        assertTrue(source.contains("tanhf"), "Should use tanh approximation");
        assertTrue(source.contains("SQRT_2_OVER_PI"), "Should have GELU coefficient");
        System.out.println("[PASS] GELU HIP C++ generation OK");
    }

    @Test
    @DisplayName("HIP C++: SiLU generates valid source")
    void testSiluSourceGeneration() {
        System.out.println("[TEST] HIP C++ Generation: SiLU");
        String source = HipKernels.generateSiluF32(HipKernels.SALT_NONE);

        assertNotNull(source);
        assertTrue(source.contains("silu_f32"), "Should have silu_f32 kernel");
        assertTrue(source.contains("__global__"), "Should be a global kernel");
        assertTrue(source.contains("expf"), "Should use exp for sigmoid");
        System.out.println("[PASS] SiLU HIP C++ generation OK");
    }

    @Test
    @DisplayName("HIP C++: Softmax generates valid source")
    void testSoftmaxSourceGeneration() {
        System.out.println("[TEST] HIP C++ Generation: Softmax");
        String source = HipKernels.generateSoftmaxF32(HipKernels.SALT_NONE);

        assertNotNull(source);
        assertTrue(source.contains("softmax_f32"), "Should have softmax_f32 kernel");
        assertTrue(source.contains("__shared__"), "Should use shared memory");
        assertTrue(source.contains("fmaxf"), "Should compute max for stability");
        assertTrue(source.contains("__syncthreads"), "Should use barriers for reduction");
        System.out.println("[PASS] Softmax HIP C++ generation OK");
    }

    @Test
    @DisplayName("HIP C++: LayerNorm generates valid source")
    void testLayerNormSourceGeneration() {
        System.out.println("[TEST] HIP C++ Generation: LayerNorm");
        String source = HipKernels.generateLayerNormF32(HipKernels.SALT_NONE);

        assertNotNull(source);
        assertTrue(source.contains("layer_norm_f32"), "Should have layer_norm_f32 kernel");
        assertTrue(source.contains("__shared__"), "Should use shared memory");
        assertTrue(source.contains("rsqrtf"), "Should use rsqrt for normalization");
        assertTrue(source.contains("gamma"), "Should accept gamma parameter");
        assertTrue(source.contains("beta"), "Should accept beta parameter");
        System.out.println("[PASS] LayerNorm HIP C++ generation OK");
    }

    @Test
    @DisplayName("HIP C++: Embedding generates valid source")
    void testEmbeddingSourceGeneration() {
        System.out.println("[TEST] HIP C++ Generation: Embedding");
        String source = HipKernels.generateEmbeddingF32(HipKernels.SALT_NONE);

        assertNotNull(source);
        assertTrue(source.contains("embedding_f32"), "Should have embedding_f32 kernel");
        assertTrue(source.contains("indices"), "Should accept indices parameter");
        assertTrue(source.contains("table"), "Should accept table parameter");
        assertTrue(source.contains("long long"), "Should use int64 indices");
        System.out.println("[PASS] Embedding HIP C++ generation OK");
    }

    // ==================== Salt Level Tests ====================

    @Test
    @DisplayName("HIP C++: GELU supports SALT_TIMING")
    void testGeluWithTiming() {
        System.out.println("[TEST] HIP C++ with SALT_TIMING: GELU");
        String source = HipKernels.generateGeluF32(HipKernels.SALT_TIMING);

        assertNotNull(source);
        assertTrue(source.contains("gelu_f32"), "Should have gelu_f32 kernel");
        assertTrue(source.contains("get_timer"), "Should have timing helper");
        System.out.println("[PASS] GELU with SALT_TIMING OK");
    }
}
