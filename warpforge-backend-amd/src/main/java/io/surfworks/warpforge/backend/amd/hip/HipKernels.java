package io.surfworks.warpforge.backend.amd.hip;

/**
 * HIP kernel generation with salt-based instrumentation.
 *
 * <p>All kernels are generated from templates with optional instrumentation
 * injected based on the salt level. This mirrors the approach used in
 * {@code CudaKernels} for NVIDIA GPUs.
 *
 * <h2>Salt Levels</h2>
 * <ul>
 *   <li>{@link #SALT_NONE} - Production kernel, no instrumentation</li>
 *   <li>{@link #SALT_TIMING} - Adds cycle counters (known overhead)</li>
 *   <li>{@link #SALT_TRACE} - Adds memory access logging (higher overhead)</li>
 * </ul>
 *
 * <h2>Implementation Note</h2>
 * <p>HIP kernels are generated as HIP C++ source code and compiled at runtime
 * via HIPRTC (HIP Runtime Compilation). Unlike CUDA's PTX which is a stable
 * intermediate representation, HIP uses standard C++ source compiled to HSACO
 * (AMD's GPU binary format).
 *
 * <p>For the PRODUCTION tier, we use rocBLAS. This kernel generation is for
 * the OPTIMIZED_OBSERVABLE tier where we need salt instrumentation.
 *
 * @see io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels NVIDIA equivalent
 */
public final class HipKernels {

    private HipKernels() {}

    // ==================== Salt Levels ====================

    /** No instrumentation - production kernel */
    public static final int SALT_NONE = 0;

    /** Timing instrumentation - cycle counters with known overhead */
    public static final int SALT_TIMING = 1;

    /** Full trace - memory access patterns (high overhead) */
    public static final int SALT_TRACE = 2;

    // ==================== Kernel Configuration ====================

    /** Standard block size for elementwise operations */
    public static final int ELEMENTWISE_BLOCK_SIZE = 256;

    /** Block size for matrix operations */
    public static final int DOT_BLOCK_SIZE = 16;

    // ==================== HIP C++ Kernel Generation ====================

    /**
     * Generate HIP C++ source for element-wise float32 addition.
     *
     * <p>Kernel signature: add_f32(float* a, float* b, float* out, int n, long* timing)
     * <p>The timing parameter is only used when salt > SALT_NONE.
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateAddF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // add_f32: Element-wise addition of float32 arrays
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>

            """.formatted(salt));

        // Timing helpers
        if (salt >= SALT_TIMING) {
            src.append("""
            // Timing helper using AMD's s_memrealtime or clock64
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        // Kernel definition
        src.append("""
            extern "C" __global__ void add_f32(
                const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ out,
                int n
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        // Calculate global index
        src.append("""
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n) return;

            """);

        // Timing start
        if (salt >= SALT_TIMING) {
            src.append("""
                // [SALT_TIMING] Capture start time
                unsigned long long t0 = get_timer();

            """);
        }

        // Core computation
        src.append("""
                // Compute: out[i] = a[i] + b[i]
                out[i] = a[i] + b[i];

            """);

        // Timing end and accumulate
        if (salt >= SALT_TIMING) {
            src.append("""
                // [SALT_TIMING] Capture end time and accumulate
                unsigned long long t1 = get_timer();
                atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    /**
     * Generate HIP C++ source for element-wise float32 multiplication.
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateMultiplyF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // multiply_f32: Element-wise multiplication of float32 arrays
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>

            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void multiply_f32(
                const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ out,
                int n
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n) return;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                // Compute: out[i] = a[i] * b[i]
                out[i] = a[i] * b[i];

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t1 = get_timer();
                atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    /**
     * Generate HIP C++ source for naive matrix multiplication.
     *
     * <p>This is the CORRECTNESS tier implementation - not optimized.
     * For PRODUCTION tier, use rocBLAS SGEMM.
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateDotF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // dot_f32: Naive matrix multiplication C[M,N] = A[M,K] * B[K,N]
            // Salt level: %d
            // NOTE: This is CORRECTNESS tier - use rocBLAS for PRODUCTION
            //
            #include <hip/hip_runtime.h>

            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void dot_f32(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= M || col >= N) return;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                // Naive dot product for this output element
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = sum;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t1 = get_timer();
                atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    // ==================== Utility Methods ====================

    /**
     * Calculate grid size for a 1D kernel launch.
     *
     * @param n Number of elements
     * @return Number of blocks needed
     */
    public static int calculateGridSize(int n) {
        return (n + ELEMENTWISE_BLOCK_SIZE - 1) / ELEMENTWISE_BLOCK_SIZE;
    }

    /**
     * Calculate grid dimensions for a 2D kernel launch (matrix operations).
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Grid dimensions [x, y]
     */
    public static int[] calculateGridSize2D(int rows, int cols) {
        int gridX = (cols + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE;
        int gridY = (rows + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE;
        return new int[]{gridX, gridY};
    }
}
