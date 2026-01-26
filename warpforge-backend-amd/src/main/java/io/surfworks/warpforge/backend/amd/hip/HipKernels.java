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

    // ==================== CustomCall Kernel Generation ====================

    /**
     * Generate HIP C++ source for GELU activation (Gaussian Error Linear Unit).
     *
     * <p>Uses the tanh approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateGeluF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // gelu_f32: GELU activation using tanh approximation
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            #define SQRT_2_OVER_PI 0.7978845608028654f
            #define GELU_COEFF 0.044715f

            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void gelu_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int n
            ) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n) return;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                float x = input[i];
                float x3 = x * x * x;
                float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
                output[i] = 0.5f * x * (1.0f + tanhf(inner));

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                // Timing accumulation would go here
            """);
        }

        src.append("}\n");

        return src.toString();
    }

    /**
     * Generate HIP C++ source for SiLU activation (Sigmoid Linear Unit / Swish).
     *
     * <p>SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateSiluF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // silu_f32: SiLU (Swish) activation
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            """.formatted(salt));

        src.append("""
            extern "C" __global__ void silu_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int n
            ) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n) return;

                float x = input[i];
                float sigmoid = 1.0f / (1.0f + expf(-x));
                output[i] = x * sigmoid;
            }
            """);

        return src.toString();
    }

    /**
     * Generate HIP C++ source for softmax over the last dimension.
     *
     * <p>Uses shared memory for the max and sum reductions within each row.
     * Each block processes one row of the input.
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateSoftmaxF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // softmax_f32: Row-wise softmax using shared memory reduction
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            #define BLOCK_SIZE 256

            """.formatted(salt));

        src.append("""
            extern "C" __global__ void softmax_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int rows,
                int cols
            ) {
                __shared__ float sdata[BLOCK_SIZE];

                int row = blockIdx.x;
                int tid = threadIdx.x;

                if (row >= rows) return;

                const float* row_in = input + row * cols;
                float* row_out = output + row * cols;

                // Step 1: Find max for numerical stability
                float local_max = -INFINITY;
                for (int i = tid; i < cols; i += blockDim.x) {
                    local_max = fmaxf(local_max, row_in[i]);
                }
                sdata[tid] = local_max;
                __syncthreads();

                // Reduce to find global max
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
                    }
                    __syncthreads();
                }
                float row_max = sdata[0];
                __syncthreads();

                // Step 2: Compute exp(x - max) and sum
                float local_sum = 0.0f;
                for (int i = tid; i < cols; i += blockDim.x) {
                    float exp_val = expf(row_in[i] - row_max);
                    row_out[i] = exp_val;  // Store temporarily
                    local_sum += exp_val;
                }
                sdata[tid] = local_sum;
                __syncthreads();

                // Reduce sum
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }
                float row_sum = sdata[0];
                __syncthreads();

                // Step 3: Normalize
                for (int i = tid; i < cols; i += blockDim.x) {
                    row_out[i] /= row_sum;
                }
            }
            """);

        return src.toString();
    }

    /**
     * Generate HIP C++ source for LayerNorm normalization.
     *
     * <p>LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
     * Each block processes one row.
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateLayerNormF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // layer_norm_f32: Layer normalization with gamma/beta
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            #define BLOCK_SIZE 256

            """.formatted(salt));

        src.append("""
            extern "C" __global__ void layer_norm_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                const float* __restrict__ gamma,
                const float* __restrict__ beta,
                int rows,
                int cols,
                float eps
            ) {
                __shared__ float sdata[BLOCK_SIZE];

                int row = blockIdx.x;
                int tid = threadIdx.x;

                if (row >= rows) return;

                const float* row_in = input + row * cols;
                float* row_out = output + row * cols;

                // Step 1: Compute mean
                float local_sum = 0.0f;
                for (int i = tid; i < cols; i += blockDim.x) {
                    local_sum += row_in[i];
                }
                sdata[tid] = local_sum;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }
                float mean = sdata[0] / cols;
                __syncthreads();

                // Step 2: Compute variance
                float local_var = 0.0f;
                for (int i = tid; i < cols; i += blockDim.x) {
                    float diff = row_in[i] - mean;
                    local_var += diff * diff;
                }
                sdata[tid] = local_var;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        sdata[tid] += sdata[tid + s];
                    }
                    __syncthreads();
                }
                float var = sdata[0] / cols;
                float rstd = rsqrtf(var + eps);
                __syncthreads();

                // Step 3: Normalize with gamma and beta
                for (int i = tid; i < cols; i += blockDim.x) {
                    float normalized = (row_in[i] - mean) * rstd;
                    row_out[i] = gamma[i] * normalized + beta[i];
                }
            }
            """);

        return src.toString();
    }

    /**
     * Generate HIP C++ source for embedding table lookup.
     *
     * <p>Looks up embeddings from a table using int64 indices.
     *
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateEmbeddingF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // embedding_f32: Embedding table lookup
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>

            """.formatted(salt));

        src.append("""
            extern "C" __global__ void embedding_f32(
                const long long* __restrict__ indices,
                const float* __restrict__ table,
                float* __restrict__ output,
                int num_indices,
                int embed_dim
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= num_indices) return;

                long long table_idx = indices[idx];
                const float* src = table + table_idx * embed_dim;
                float* dst = output + idx * embed_dim;

                // Copy embedding vector
                for (int i = 0; i < embed_dim; i++) {
                    dst[i] = src[i];
                }
            }
            """);

        return src.toString();
    }

    // ==================== Binary Elementwise Operations ====================

    /**
     * Generate HIP C++ source for element-wise float32 subtraction.
     */
    public static String generateSubtractF32(int salt) {
        return generateBinaryElementwiseF32("subtract", "a[i] - b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 division.
     */
    public static String generateDivideF32(int salt) {
        return generateBinaryElementwiseF32("divide", "a[i] / b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 maximum.
     */
    public static String generateMaximumF32(int salt) {
        return generateBinaryElementwiseF32("maximum", "fmaxf(a[i], b[i])", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 minimum.
     */
    public static String generateMinimumF32(int salt) {
        return generateBinaryElementwiseF32("minimum", "fminf(a[i], b[i])", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 power.
     */
    public static String generatePowerF32(int salt) {
        return generateBinaryElementwiseF32("power", "powf(a[i], b[i])", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 remainder.
     */
    public static String generateRemainderF32(int salt) {
        return generateBinaryElementwiseF32("remainder", "fmodf(a[i], b[i])", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 atan2.
     */
    public static String generateAtan2F32(int salt) {
        return generateBinaryElementwiseF32("atan2", "atan2f(a[i], b[i])", salt);
    }

    private static String generateBinaryElementwiseF32(String name, String op, int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // %s_f32: Element-wise %s of float32 arrays
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            """.formatted(name, name, salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void %s_f32(
                const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ out,
                int n
            """.formatted(name));

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

        src.append("        out[i] = %s;\n\n".formatted(op));

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t1 = get_timer();
                atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    // ==================== Unary Elementwise Operations ====================

    /**
     * Generate HIP C++ source for element-wise float32 negation.
     */
    public static String generateNegateF32(int salt) {
        return generateUnaryElementwiseF32("negate", "-x", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 absolute value.
     */
    public static String generateAbsF32(int salt) {
        return generateUnaryElementwiseF32("abs", "fabsf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 exponential.
     */
    public static String generateExpF32(int salt) {
        return generateUnaryElementwiseF32("exp", "expf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 natural logarithm.
     */
    public static String generateLogF32(int salt) {
        return generateUnaryElementwiseF32("log", "logf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 square root.
     */
    public static String generateSqrtF32(int salt) {
        return generateUnaryElementwiseF32("sqrt", "sqrtf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 hyperbolic tangent.
     */
    public static String generateTanhF32(int salt) {
        return generateUnaryElementwiseF32("tanh", "tanhf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 reciprocal square root.
     */
    public static String generateRsqrtF32(int salt) {
        return generateUnaryElementwiseF32("rsqrt", "rsqrtf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 sine.
     */
    public static String generateSinF32(int salt) {
        return generateUnaryElementwiseF32("sin", "sinf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 cosine.
     */
    public static String generateCosF32(int salt) {
        return generateUnaryElementwiseF32("cos", "cosf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 ceiling.
     */
    public static String generateCeilF32(int salt) {
        return generateUnaryElementwiseF32("ceil", "ceilf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 floor.
     */
    public static String generateFloorF32(int salt) {
        return generateUnaryElementwiseF32("floor", "floorf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 sign.
     */
    public static String generateSignF32(int salt) {
        return generateUnaryElementwiseF32("sign", "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 tangent.
     */
    public static String generateTanF32(int salt) {
        return generateUnaryElementwiseF32("tan", "tanf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 logistic (sigmoid).
     */
    public static String generateLogisticF32(int salt) {
        return generateUnaryElementwiseF32("logistic", "1.0f / (1.0f + expf(-x))", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 expm1 (exp(x) - 1).
     */
    public static String generateExpm1F32(int salt) {
        return generateUnaryElementwiseF32("expm1", "expm1f(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 log1p (log(1 + x)).
     */
    public static String generateLog1pF32(int salt) {
        return generateUnaryElementwiseF32("log1p", "log1pf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 cube root.
     */
    public static String generateCbrtF32(int salt) {
        return generateUnaryElementwiseF32("cbrt", "cbrtf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 is_finite check.
     */
    public static String generateIsFiniteF32(int salt) {
        return generateUnaryElementwiseF32("is_finite", "isfinite(x) ? 1.0f : 0.0f", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 round nearest even.
     */
    public static String generateRoundNearestEvenF32(int salt) {
        return generateUnaryElementwiseF32("round_nearest_even", "rintf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 round nearest away from zero.
     */
    public static String generateRoundNearestAfzF32(int salt) {
        return generateUnaryElementwiseF32("round_nearest_afz", "roundf(x)", salt);
    }

    /**
     * Generate HIP C++ source for element-wise float32 logical NOT.
     */
    public static String generateNotF32(int salt) {
        return generateUnaryElementwiseF32("not", "(x == 0.0f) ? 1.0f : 0.0f", salt);
    }

    private static String generateUnaryElementwiseF32(String name, String op, int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // %s_f32: Element-wise %s of float32 array
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            """.formatted(name, name, salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void %s_f32(
                const float* __restrict__ input,
                float* __restrict__ out,
                int n
            """.formatted(name));

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n) return;

                float x = input[i];

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("        out[i] = %s;\n\n".formatted(op));

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t1 = get_timer();
                atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    // ==================== Compare Operations ====================

    /**
     * Generate HIP C++ source for element-wise float32 comparison.
     *
     * @param direction One of: "EQ", "NE", "LT", "LE", "GT", "GE"
     * @param salt Instrumentation level
     * @return HIP C++ source code
     */
    public static String generateCompareF32(String direction, int salt) {
        String op = switch (direction) {
            case "EQ" -> "a[i] == b[i]";
            case "NE" -> "a[i] != b[i]";
            case "LT" -> "a[i] < b[i]";
            case "LE" -> "a[i] <= b[i]";
            case "GT" -> "a[i] > b[i]";
            case "GE" -> "a[i] >= b[i]";
            default -> throw new IllegalArgumentException("Unknown comparison direction: " + direction);
        };

        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // compare_%s_f32: Element-wise %s comparison of float32 arrays
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>

            """.formatted(direction.toLowerCase(), direction, salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void compare_%s_f32(
                const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ out,
                int n
            """.formatted(direction.toLowerCase()));

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

        src.append("        out[i] = (%s) ? 1.0f : 0.0f;\n\n".formatted(op));

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
     * Generate HIP C++ source for element-wise float32 select.
     */
    public static String generateSelectF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // select_f32: Element-wise select of float32 arrays
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
            extern "C" __global__ void select_f32(
                const float* __restrict__ pred_ptr,
                const float* __restrict__ on_true_ptr,
                const float* __restrict__ on_false_ptr,
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
                out[i] = (pred_ptr[i] != 0.0f) ? on_true_ptr[i] : on_false_ptr[i];

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
     * Generate HIP C++ source for element-wise float32 clamp.
     */
    public static String generateClampF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // clamp_f32: Element-wise clamp of float32 arrays
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void clamp_f32(
                const float* __restrict__ min_ptr,
                const float* __restrict__ operand_ptr,
                const float* __restrict__ max_ptr,
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
                out[i] = fmaxf(min_ptr[i], fminf(operand_ptr[i], max_ptr[i]));

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

    // ==================== Integer Bitwise Operations ====================

    /**
     * Generate HIP C++ source for element-wise int32 AND.
     */
    public static String generateAndI32(int salt) {
        return generateIntegerBinaryOp("and", "a[i] & b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise int32 OR.
     */
    public static String generateOrI32(int salt) {
        return generateIntegerBinaryOp("or", "a[i] | b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise int32 XOR.
     */
    public static String generateXorI32(int salt) {
        return generateIntegerBinaryOp("xor", "a[i] ^ b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise int32 left shift.
     */
    public static String generateShiftLeftI32(int salt) {
        return generateIntegerBinaryOp("shift_left", "a[i] << b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise int32 arithmetic right shift.
     */
    public static String generateShiftRightArithmeticI32(int salt) {
        return generateIntegerBinaryOp("shift_right_arithmetic", "a[i] >> b[i]", salt);
    }

    /**
     * Generate HIP C++ source for element-wise int32 logical right shift.
     */
    public static String generateShiftRightLogicalI32(int salt) {
        return generateIntegerBinaryOp("shift_right_logical", "((unsigned int)a[i]) >> b[i]", salt);
    }

    private static String generateIntegerBinaryOp(String name, String op, int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // %s_i32: Element-wise %s of int32 arrays
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>

            """.formatted(name, name, salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void %s_i32(
                const int* __restrict__ a,
                const int* __restrict__ b,
                int* __restrict__ out,
                int n
            """.formatted(name));

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

        src.append("        out[i] = %s;\n\n".formatted(op));

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t1 = get_timer();
                atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    // ==================== Reduce Operations ====================

    /**
     * Generate HIP C++ source for reduce add.
     */
    public static String generateReduceAddF32(int salt) {
        return generateReduceOp("add", "0.0f", "+", salt);
    }

    /**
     * Generate HIP C++ source for reduce max.
     */
    public static String generateReduceMaxF32(int salt) {
        return generateReduceOp("max", "-INFINITY", "fmaxf", salt);
    }

    /**
     * Generate HIP C++ source for reduce min.
     */
    public static String generateReduceMinF32(int salt) {
        return generateReduceOp("min", "INFINITY", "fminf", salt);
    }

    /**
     * Generate HIP C++ source for reduce mul.
     */
    public static String generateReduceMulF32(int salt) {
        return generateReduceOp("mul", "1.0f", "*", salt);
    }

    private static String generateReduceOp(String name, String identity, String op, int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // reduce_%s_f32: Reduce %s of float32 array
            // Salt level: %d
            //
            #include <hip/hip_runtime.h>
            #include <math.h>

            #define BLOCK_SIZE 256

            """.formatted(name, name, salt));

        if (salt >= SALT_TIMING) {
            src.append("""
            __device__ __forceinline__ unsigned long long get_timer() {
                return clock64();
            }

            """);
        }

        src.append("""
            extern "C" __global__ void reduce_%s_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int n
            """.formatted(name));

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                __shared__ float sdata[BLOCK_SIZE];

                int tid = threadIdx.x;
                int i = blockIdx.x * blockDim.x + threadIdx.x;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        // Load into shared memory
        src.append("        sdata[tid] = (i < n) ? input[i] : %s;\n".formatted(identity));
        src.append("        __syncthreads();\n\n");

        // Reduction in shared memory
        src.append("        for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n");
        src.append("            if (tid < s) {\n");
        if (op.equals("+") || op.equals("*")) {
            src.append("                sdata[tid] = sdata[tid] %s sdata[tid + s];\n".formatted(op));
        } else {
            src.append("                sdata[tid] = %s(sdata[tid], sdata[tid + s]);\n".formatted(op));
        }
        src.append("            }\n");
        src.append("            __syncthreads();\n");
        src.append("        }\n\n");

        src.append("        if (tid == 0) {\n");
        if (op.equals("+")) {
            src.append("            atomicAdd(output, sdata[0]);\n");
        } else {
            src.append("            output[blockIdx.x] = sdata[0];\n");
        }
        src.append("        }\n");

        if (salt >= SALT_TIMING) {
            src.append("""

                unsigned long long t1 = get_timer();
                if (tid == 0) atomicAdd(timing, t1 - t0);

            """);
        }

        src.append("}\n");

        return src.toString();
    }

    // ==================== Shape Manipulation Operations ====================

    /**
     * Generate HIP C++ source for reshape (copy).
     */
    public static String generateReshapeF32(int salt) {
        return generateUnaryElementwiseF32("reshape", "x", salt);
    }

    /**
     * Generate HIP C++ source for 2D transpose.
     */
    public static String generateTranspose2DF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // transpose_2d_f32: 2D matrix transpose
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
            extern "C" __global__ void transpose_2d_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int rows,
                int cols
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= rows || col >= cols) return;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                output[col * rows + row] = input[row * cols + col];

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
     * Generate HIP C++ source for broadcast scalar.
     */
    public static String generateBroadcastScalarF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // broadcast_scalar_f32: Broadcast scalar to array
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
            extern "C" __global__ void broadcast_scalar_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int n
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= n) return;

                float value = input[0];

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                output[i] = value;

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
     * Generate HIP C++ source for broadcast 1D to 2D along rows.
     */
    public static String generateBroadcast1Dto2DRowF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // broadcast_1d_to_2d_row_f32: Broadcast 1D array to 2D along rows
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
            extern "C" __global__ void broadcast_1d_to_2d_row_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int rows,
                int cols
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= rows || col >= cols) return;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                output[row * cols + col] = input[col];

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
     * Generate HIP C++ source for broadcast 1D to 2D along columns.
     */
    public static String generateBroadcast1Dto2DColF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // broadcast_1d_to_2d_col_f32: Broadcast 1D array to 2D along columns
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
            extern "C" __global__ void broadcast_1d_to_2d_col_f32(
                const float* __restrict__ input,
                float* __restrict__ output,
                int rows,
                int cols
            """);

        if (salt >= SALT_TIMING) {
            src.append("    , unsigned long long* timing\n");
        }

        src.append(") {\n");

        src.append("""
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;

                if (row >= rows || col >= cols) return;

            """);

        if (salt >= SALT_TIMING) {
            src.append("""
                unsigned long long t0 = get_timer();

            """);
        }

        src.append("""
                output[row * cols + col] = input[row];

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

    // ==================== Type Conversion Operations ====================

    /**
     * Generate HIP C++ source for F32 to I32 conversion.
     */
    public static String generateConvertF32toI32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // convert_f32_to_i32: Convert float32 to int32
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
            extern "C" __global__ void convert_f32_to_i32(
                const float* __restrict__ input,
                int* __restrict__ output,
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
                output[i] = (int)input[i];

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
     * Generate HIP C++ source for I32 to F32 conversion.
     */
    public static String generateConvertI32toF32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // convert_i32_to_f32: Convert int32 to float32
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
            extern "C" __global__ void convert_i32_to_f32(
                const int* __restrict__ input,
                float* __restrict__ output,
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
                output[i] = (float)input[i];

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
     * Generate HIP C++ source for I32 to I32 identity conversion.
     */
    public static String generateConvertI32toI32(int salt) {
        StringBuilder src = new StringBuilder();

        src.append("""
            //
            // convert_i32_to_i32: Identity conversion for int32
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
            extern "C" __global__ void convert_i32_to_i32(
                const int* __restrict__ input,
                int* __restrict__ output,
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
                output[i] = input[i];

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
