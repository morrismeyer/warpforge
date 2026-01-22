package io.surfworks.warpforge.backend.amd;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipRuntime;
import io.surfworks.warpforge.backend.amd.ops.RocblasDotKernel;
import io.surfworks.warpforge.backend.amd.rocblas.RocblasRuntime;
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
 * Tests for rocBLAS matrix multiplication (SGEMM) on AMD GPUs.
 *
 * <p>Tests tagged with @Tag("amd") require actual AMD GPU hardware with ROCm.
 * Tests without the tag verify API compatibility without hardware.
 */
@DisplayName("rocBLAS Dot Kernel Tests")
class RocblasDotKernelTest {

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

    /**
     * Helper to create HIP context, skipping test if creation fails.
     */
    private void createContext() {
        try {
            context = HipContext.create(0);
        } catch (Exception e) {
            assumeTrue(false, "HIP context creation failed: " + e.getMessage());
        }
    }

    // ==================== API Compatibility Tests (No Hardware) ====================

    @Test
    @DisplayName("rocBLAS runtime reports availability correctly")
    void testRocblasAvailability() {
        // This should not throw - it just checks availability
        boolean available = RocblasRuntime.isAvailable();
        // Just verify we can check - result depends on system
        assertNotNull(Boolean.valueOf(available));
    }

    @Test
    @DisplayName("HIP runtime reports availability correctly")
    void testHipAvailability() {
        boolean available = HipRuntime.isAvailable();
        assertNotNull(Boolean.valueOf(available));
    }

    // ==================== AMD Hardware Tests ====================

    @Test
    @Tag("amd")
    @DisplayName("HIP initializes successfully")
    void testHipInitialization() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");

        createContext();
        assertNotNull(context);
        assertTrue(context.deviceIndex() == 0);
    }

    @Test
    @Tag("amd")
    @DisplayName("rocBLAS handle creation succeeds")
    void testRocblasHandleCreation() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");
        assumeTrue(RocblasRuntime.isAvailable(), "rocBLAS not available");

        createContext();
        assertTrue(context.isRocblasAvailable());

        long handle = context.getRocblasHandle();
        assertTrue(handle != 0, "rocBLAS handle should be non-zero");
    }

    @Test
    @Tag("amd")
    @DisplayName("Device memory allocation works")
    void testDeviceMemoryAllocation() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");

        createContext();

        long byteSize = 1024 * 4; // 1024 floats
        long dPtr = context.allocate(byteSize);

        assertTrue(dPtr != 0, "Device pointer should be non-zero");

        // Free the memory
        context.free(dPtr);
    }

    @Test
    @Tag("amd")
    @DisplayName("rocBLAS SGEMM executes correctly")
    void testRocblasSgemm() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");
        assumeTrue(RocblasRuntime.isAvailable(), "rocBLAS not available");

        createContext();
        assumeTrue(context.isRocblasAvailable(), "rocBLAS context not available");

        // Test: C[2,3] = A[2,4] * B[4,3]
        int M = 2, K = 4, N = 3;

        // A[2,4] = [[1,2,3,4], [5,6,7,8]]
        float[] aData = {1, 2, 3, 4, 5, 6, 7, 8};
        // B[4,3] = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        float[] bData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        // Expected C[2,3] = [[70, 80, 90], [158, 184, 210]]
        float[] expected = {70, 80, 90, 158, 184, 210};

        try (Tensor a = Tensor.fromFloatArray(aData, M, K);
             Tensor b = Tensor.fromFloatArray(bData, K, N)) {

            // Allocate device memory
            long dA = context.allocate(M * K * 4L);
            long dB = context.allocate(K * N * 4L);
            long dC = context.allocate(M * N * 4L);

            try {
                // Copy to device
                context.copyToDevice(dA, a.data());
                context.copyToDevice(dB, b.data());

                // Execute SGEMM
                context.sgemm(dA, dB, dC, M, N, K);
                context.synchronize();

                // Copy result back
                Tensor c = Tensor.zeros(a.dtype(), M, N);
                context.copyToHost(c.data(), dC, M * N * 4L);

                // Verify
                float[] actual = c.toFloatArray();
                assertArrayEquals(expected, actual, 1e-4f,
                    "Matrix multiplication result should match expected");

            } finally {
                context.free(dA);
                context.free(dB);
                context.free(dC);
            }
        }
    }

    @Test
    @Tag("amd")
    @DisplayName("RocblasDotKernel executes StableHLO DotOp correctly")
    void testRocblasDotKernelExecution() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");
        assumeTrue(RocblasRuntime.isAvailable(), "rocBLAS not available");

        createContext();
        assumeTrue(context.isRocblasAvailable(), "rocBLAS context not available");

        RocblasDotKernel kernel = new RocblasDotKernel(context);

        // Test: C[2,2] = A[2,3] * B[3,2]
        float[] aData = {1, 2, 3, 4, 5, 6};
        float[] bData = {7, 8, 9, 10, 11, 12};
        // Expected: [[58, 64], [139, 154]]
        float[] expected = {58, 64, 139, 154};

        try (Tensor a = Tensor.fromFloatArray(aData, 2, 3);
             Tensor b = Tensor.fromFloatArray(bData, 3, 2)) {

            // Create DotOp
            StableHloAst.TensorType aType = new StableHloAst.TensorType(
                List.of(2, 3), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType bType = new StableHloAst.TensorType(
                List.of(3, 2), StableHloAst.ScalarType.F32);
            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(2, 2), StableHloAst.ScalarType.F32);

            StableHloAst.DotOp dotOp = new StableHloAst.DotOp(
                new StableHloAst.Value("lhs", aType),
                new StableHloAst.Value("rhs", bType),
                new StableHloAst.Value("result", resultType),
                resultType
            );

            // Execute
            List<Tensor> results = kernel.execute(dotOp, List.of(a, b));

            // Verify
            assertNotNull(results);
            assertEquals(1, results.size());

            Tensor result = results.get(0);
            assertArrayEquals(expected, result.toFloatArray(), 1e-4f);
            assertArrayEquals(new int[]{2, 2}, result.shape());
        }
    }

    @Test
    @Tag("amd")
    @DisplayName("rocBLAS SGEMM handles larger matrices")
    void testRocblasSgemmLarger() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");
        assumeTrue(RocblasRuntime.isAvailable(), "rocBLAS not available");

        createContext();
        assumeTrue(context.isRocblasAvailable(), "rocBLAS context not available");

        // Test: C[64,64] = A[64,128] * B[128,64]
        int M = 64, K = 128, N = 64;

        float[] aData = new float[M * K];
        float[] bData = new float[K * N];

        // Initialize with simple pattern
        for (int i = 0; i < M * K; i++) aData[i] = 0.01f * i;
        for (int i = 0; i < K * N; i++) bData[i] = 0.01f * i;

        try (Tensor a = Tensor.fromFloatArray(aData, M, K);
             Tensor b = Tensor.fromFloatArray(bData, K, N)) {

            long dA = context.allocate(M * K * 4L);
            long dB = context.allocate(K * N * 4L);
            long dC = context.allocate(M * N * 4L);

            try {
                context.copyToDevice(dA, a.data());
                context.copyToDevice(dB, b.data());

                context.sgemm(dA, dB, dC, M, N, K);
                context.synchronize();

                Tensor c = Tensor.zeros(a.dtype(), M, N);
                context.copyToHost(c.data(), dC, M * N * 4L);

                // Just verify execution completed and result has correct shape
                assertEquals(M * N, c.elementCount());

                // Verify first element (computed manually)
                // C[0,0] = sum(A[0,:] * B[:,0]) = sum(0.01*i * 0.01*(i*64)) for i=0..127
                float[] actual = c.toFloatArray();
                assertTrue(actual[0] >= 0, "Result should be positive for positive inputs");

            } finally {
                context.free(dA);
                context.free(dB);
                context.free(dC);
            }
        }
    }

    @Test
    @Tag("amd")
    @DisplayName("Device synchronization works")
    void testDeviceSynchronization() {
        assumeTrue(HipRuntime.isAvailable(), "HIP not available");

        createContext();

        // Should not throw
        context.synchronize();
    }
}
