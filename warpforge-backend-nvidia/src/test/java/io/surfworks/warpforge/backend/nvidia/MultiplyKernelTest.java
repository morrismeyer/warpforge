package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for the NVIDIA Multiply kernel implementation.
 *
 * <p>Tests without @Tag("nvidia") run in stub mode on any machine.
 * Tests with @Tag("nvidia") require actual CUDA hardware.
 */
@DisplayName("NVIDIA Multiply Kernel Tests")
class MultiplyKernelTest {

    private NvidiaBackend backend;

    @BeforeEach
    void setUp() {
        // Create backend - will use CUDA context if available, stub mode otherwise
        backend = new NvidiaBackend(0, CudaKernels.SALT_NONE);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== PTX Generation Tests (No CUDA Required) ====================

    @Test
    @DisplayName("PTX generation produces valid output for SALT_NONE")
    void testPtxGenerationNoSalt() {
        String ptx = CudaKernels.generateMultiplyF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry multiply_f32"));
        assertTrue(ptx.contains("mul.f32"));
        assertTrue(ptx.contains(".param .u64 a_ptr"));
        assertTrue(ptx.contains(".param .u64 b_ptr"));
        assertTrue(ptx.contains(".param .u64 out_ptr"));
        assertTrue(ptx.contains(".param .u32 n"));

        // Should NOT contain timing parameter
        assertTrue(!ptx.contains("timing_ptr"));
    }

    @Test
    @DisplayName("PTX generation includes timing instrumentation for SALT_TIMING")
    void testPtxGenerationWithTiming() {
        String ptx = CudaKernels.generateMultiplyF32(CudaKernels.SALT_TIMING);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry multiply_f32"));
        assertTrue(ptx.contains("mul.f32"));

        // Should contain timing parameter and instrumentation
        assertTrue(ptx.contains("timing_ptr"));
        assertTrue(ptx.contains("%globaltimer"));
        assertTrue(ptx.contains("atom.global.add.u64"));
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("Multiply kernel executes correctly on CUDA hardware")
    void testMultiplyKernelExecution() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // Create test data
        float[] aData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] bData = {0.5f, 1.5f, 2.0f, 2.5f, 3.0f, 0.1f, 0.2f, 0.25f};
        float[] expected = {0.5f, 3.0f, 6.0f, 10.0f, 15.0f, 0.6f, 1.4f, 2.0f};

        try (Tensor a = Tensor.fromFloatArray(aData, 8);
             Tensor b = Tensor.fromFloatArray(bData, 8)) {

            // Create MultiplyOp
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(8),
                StableHloAst.ScalarType.F32
            );
            StableHloAst.MultiplyOp multiplyOp = new StableHloAst.MultiplyOp(
                new StableHloAst.Value("0", resultType),
                new StableHloAst.Value("1", resultType),
                new StableHloAst.Value("2", resultType),
                resultType
            );

            // Execute
            List<Tensor> results = backend.execute(multiplyOp, List.of(a, b));

            // Verify
            assertNotNull(results);
            assertEquals(1, results.size());

            Tensor result = results.get(0);
            assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
        }
    }

    @Test
    @Tag("nvidia")
    @DisplayName("Multiply kernel handles large tensors")
    void testMultiplyKernelLargeTensor() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        int n = 1_000_000;
        float[] aData = new float[n];
        float[] bData = new float[n];
        float[] expected = new float[n];

        for (int i = 0; i < n; i++) {
            aData[i] = (i + 1) * 0.001f;
            bData[i] = 2.0f;
            expected[i] = (i + 1) * 0.002f;
        }

        try (Tensor a = Tensor.fromFloatArray(aData, n);
             Tensor b = Tensor.fromFloatArray(bData, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );
            StableHloAst.MultiplyOp multiplyOp = new StableHloAst.MultiplyOp(
                new StableHloAst.Value("0", resultType),
                new StableHloAst.Value("1", resultType),
                new StableHloAst.Value("2", resultType),
                resultType
            );

            List<Tensor> results = backend.execute(multiplyOp, List.of(a, b));

            Tensor result = results.get(0);
            float[] actual = result.toFloatArray();

            // Check a sample of values
            assertEquals(expected[0], actual[0], 1e-5f);
            assertEquals(expected[n/2], actual[n/2], 1e-4f);
            assertEquals(expected[n-1], actual[n-1], 1e-3f);
        }
    }

    @Test
    @Tag("nvidia")
    @DisplayName("Multiply kernel with SALT_TIMING executes correctly")
    void testMultiplyKernelWithTiming() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");

        // Create backend with timing instrumentation
        try (NvidiaBackend timedBackend = new NvidiaBackend(0, CudaKernels.SALT_TIMING)) {
            assumeTrue(timedBackend.hasCudaContext(), "CUDA context not available");

            float[] aData = {2.0f, 3.0f, 4.0f, 5.0f};
            float[] bData = {10.0f, 20.0f, 30.0f, 40.0f};
            float[] expected = {20.0f, 60.0f, 120.0f, 200.0f};

            try (Tensor a = Tensor.fromFloatArray(aData, 4);
                 Tensor b = Tensor.fromFloatArray(bData, 4)) {

                StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                    List.of(4),
                    StableHloAst.ScalarType.F32
                );
                StableHloAst.MultiplyOp multiplyOp = new StableHloAst.MultiplyOp(
                    new StableHloAst.Value("0", resultType),
                    new StableHloAst.Value("1", resultType),
                    new StableHloAst.Value("2", resultType),
                    resultType
                );

                List<Tensor> results = timedBackend.execute(multiplyOp, List.of(a, b));

                // Results should still be correct with timing enabled
                Tensor result = results.get(0);
                assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
            }
        }
    }

    @Test
    @Tag("nvidia")
    @DisplayName("Multiply kernel handles 2D tensor shapes")
    void testMultiplyKernel2D() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] aData = {1, 2, 3, 4, 5, 6};
        float[] bData = {2, 2, 2, 3, 3, 3};
        float[] expected = {2, 4, 6, 12, 15, 18};

        try (Tensor a = Tensor.fromFloatArray(aData, 2, 3);
             Tensor b = Tensor.fromFloatArray(bData, 2, 3)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(2, 3),
                StableHloAst.ScalarType.F32
            );
            StableHloAst.MultiplyOp multiplyOp = new StableHloAst.MultiplyOp(
                new StableHloAst.Value("0", resultType),
                new StableHloAst.Value("1", resultType),
                new StableHloAst.Value("2", resultType),
                resultType
            );

            List<Tensor> results = backend.execute(multiplyOp, List.of(a, b));

            Tensor result = results.get(0);
            assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
            assertArrayEquals(new int[]{2, 3}, result.shape());
        }
    }

    @Test
    @Tag("nvidia")
    @DisplayName("Multiply kernel handles zeros correctly")
    void testMultiplyByZero() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] aData = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] bData = {0.0f, 0.0f, 0.0f, 0.0f};
        float[] expected = {0.0f, 0.0f, 0.0f, 0.0f};

        try (Tensor a = Tensor.fromFloatArray(aData, 4);
             Tensor b = Tensor.fromFloatArray(bData, 4)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(4),
                StableHloAst.ScalarType.F32
            );
            StableHloAst.MultiplyOp multiplyOp = new StableHloAst.MultiplyOp(
                new StableHloAst.Value("0", resultType),
                new StableHloAst.Value("1", resultType),
                new StableHloAst.Value("2", resultType),
                resultType
            );

            List<Tensor> results = backend.execute(multiplyOp, List.of(a, b));

            Tensor result = results.get(0);
            assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
        }
    }

    @Test
    @Tag("nvidia")
    @DisplayName("Multiply kernel handles negative numbers")
    void testMultiplyNegatives() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] aData = {-1.0f, 2.0f, -3.0f, 4.0f};
        float[] bData = {2.0f, -3.0f, -4.0f, 5.0f};
        float[] expected = {-2.0f, -6.0f, 12.0f, 20.0f};

        try (Tensor a = Tensor.fromFloatArray(aData, 4);
             Tensor b = Tensor.fromFloatArray(bData, 4)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(4),
                StableHloAst.ScalarType.F32
            );
            StableHloAst.MultiplyOp multiplyOp = new StableHloAst.MultiplyOp(
                new StableHloAst.Value("0", resultType),
                new StableHloAst.Value("1", resultType),
                new StableHloAst.Value("2", resultType),
                resultType
            );

            List<Tensor> results = backend.execute(multiplyOp, List.of(a, b));

            Tensor result = results.get(0);
            assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
        }
    }
}
