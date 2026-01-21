package io.surfworks.warpforge.backend.nvidia.cuda;

/**
 * PTX kernel generation with salt-based instrumentation.
 *
 * <p>All kernels are generated from templates with optional instrumentation
 * injected based on the salt level. The instrumentation is part of the same
 * code path - not a separate implementation - ensuring that performance
 * measurements reflect actual production behavior with known, quantifiable overhead.
 *
 * <h2>Salt Levels</h2>
 * <ul>
 *   <li>{@link #SALT_NONE} - Production kernel, no instrumentation</li>
 *   <li>{@link #SALT_TIMING} - Adds cycle counters (known overhead: ~8 instructions)</li>
 *   <li>{@link #SALT_TRACE} - Adds memory access logging (higher overhead)</li>
 * </ul>
 *
 * <h2>Design Principle</h2>
 * <p>There is ONE implementation of each operation. Instrumentation variants
 * are generated from the same template with conditional sections enabled.
 * This avoids Heisenbug scenarios where switching between "debug" and "production"
 * paths would measure entirely different code.
 */
public final class CudaKernels {

    private CudaKernels() {}

    // ==================== Salt Levels ====================

    /** No instrumentation - production kernel */
    public static final int SALT_NONE = 0;

    /** Timing instrumentation - cycle counters with known overhead */
    public static final int SALT_TIMING = 1;

    /** Full trace - memory access patterns, warp divergence (high overhead) */
    public static final int SALT_TRACE = 2;

    // ==================== Kernel Configuration ====================

    /** Standard block size for elementwise operations */
    public static final int ELEMENTWISE_BLOCK_SIZE = 256;

    /** Module name for elementwise operations */
    public static final String ELEMENTWISE_MODULE = "elementwise_ops";

    // ==================== PTX Generation ====================

    /**
     * Generate PTX for element-wise float32 addition.
     *
     * <p>Kernel signature: add_f32(float* a, float* b, float* out, int n, long* timing)
     * <p>The timing parameter is only used when salt > SALT_NONE.
     *
     * @param salt Instrumentation level
     * @return PTX source code
     */
    public static String generateAddF32(int salt) {
        StringBuilder ptx = new StringBuilder();

        // PTX header
        ptx.append("""
            //
            // add_f32: Element-wise addition of float32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            """.formatted(salt));

        // Entry point with parameters
        ptx.append("""
            .visible .entry add_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            """);

        // Add timing accumulator parameter if instrumented
        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append(")\n{\n");

        // Register declarations
        ptx.append("""
                .reg .pred  %p<2>;
                .reg .f32   %f<4>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<10>;
            """);

        // Additional registers for timing
        if (salt >= SALT_TIMING) {
            ptx.append("""
                .reg .b64   %rd_t0, %rd_t1, %rd_delta;
            """);
        }

        // Calculate global thread index
        ptx.append("""

                // Calculate global thread index: i = blockIdx.x * blockDim.x + threadIdx.x
                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                // Load n and check bounds
                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                // Load base pointers
                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [out_ptr];

                // Calculate byte offset: offset = i * 4 (sizeof float)
                cvt.s64.s32     %rd4, %r4;
                shl.b64         %rd5, %rd4, 2;

                // Calculate element addresses
                add.s64         %rd6, %rd1, %rd5;
                add.s64         %rd7, %rd2, %rd5;
                add.s64         %rd8, %rd3, %rd5;

                // Load a[i] and b[i]
                ld.global.f32   %f1, [%rd6];
                ld.global.f32   %f2, [%rd7];

            """);

        // Timing: capture start
        if (salt >= SALT_TIMING) {
            ptx.append("""
                // [SALT_TIMING] Capture start time
                mov.u64         %rd_t0, %globaltimer;

            """);
        }

        // Core computation
        ptx.append("""
                // Compute: out[i] = a[i] + b[i]
                add.f32         %f3, %f1, %f2;

            """);

        // Timing: capture end and accumulate
        if (salt >= SALT_TIMING) {
            ptx.append("""
                // [SALT_TIMING] Capture end time and accumulate
                mov.u64         %rd_t1, %globaltimer;
                sub.u64         %rd_delta, %rd_t1, %rd_t0;
                ld.param.u64    %rd9, [timing_ptr];
                atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        // Store result
        ptx.append("""
                // Store result
                st.global.f32   [%rd8], %f3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for element-wise float32 multiplication.
     */
    public static String generateMultiplyF32(int salt) {
        // Same structure as add, just different core operation
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // multiply_f32: Element-wise multiplication of float32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry multiply_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            """.formatted(salt));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<4>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<10>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [out_ptr];

                cvt.s64.s32     %rd4, %r4;
                shl.b64         %rd5, %rd4, 2;

                add.s64         %rd6, %rd1, %rd5;
                add.s64         %rd7, %rd2, %rd5;
                add.s64         %rd8, %rd3, %rd5;

                ld.global.f32   %f1, [%rd6];
                ld.global.f32   %f2, [%rd7];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        ptx.append("    mul.f32         %f3, %f1, %f2;\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd9, [timing_ptr];
                    atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        ptx.append("""
                st.global.f32   [%rd8], %f3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    /**
     * Generate PTX for element-wise float32 subtraction.
     */
    public static String generateSubtractF32(int salt) {
        return generateBinaryElementwiseF32("subtract", "sub.f32", "a[i] - b[i]", salt);
    }

    /**
     * Generate PTX for element-wise float32 division.
     */
    public static String generateDivideF32(int salt) {
        return generateBinaryElementwiseF32("divide", "div.approx.f32", "a[i] / b[i]", salt);
    }

    /**
     * Generate PTX for element-wise float32 maximum.
     */
    public static String generateMaximumF32(int salt) {
        return generateBinaryElementwiseF32("maximum", "max.f32", "max(a[i], b[i])", salt);
    }

    /**
     * Generate PTX for element-wise float32 minimum.
     */
    public static String generateMinimumF32(int salt) {
        return generateBinaryElementwiseF32("minimum", "min.f32", "min(a[i], b[i])", salt);
    }

    /**
     * Generate PTX for a binary elementwise float32 operation.
     *
     * @param opName Operation name for comments and entry point
     * @param ptxInstruction The PTX instruction (e.g., "add.f32", "mul.f32")
     * @param comment Description of the operation
     * @param salt Instrumentation level
     * @return PTX source code
     */
    private static String generateBinaryElementwiseF32(String opName, String ptxInstruction,
                                                        String comment, int salt) {
        StringBuilder ptx = new StringBuilder();

        ptx.append("""
            //
            // %s_f32: Element-wise %s of float32 arrays
            // Salt level: %d
            //
            .version 7.0
            .target sm_50
            .address_size 64

            .visible .entry %s_f32(
                .param .u64 a_ptr,
                .param .u64 b_ptr,
                .param .u64 out_ptr,
                .param .u32 n
            """.formatted(opName, comment, salt, opName));

        if (salt >= SALT_TIMING) {
            ptx.append("    ,.param .u64 timing_ptr\n");
        }

        ptx.append("""
            )
            {
                .reg .pred  %p<2>;
                .reg .f32   %f<4>;
                .reg .b32   %r<6>;
                .reg .b64   %rd<10>;
            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    .reg .b64   %rd_t0, %rd_t1, %rd_delta;\n");
        }

        ptx.append("""

                mov.u32         %r1, %ctaid.x;
                mov.u32         %r2, %ntid.x;
                mov.u32         %r3, %tid.x;
                mad.lo.s32      %r4, %r1, %r2, %r3;

                ld.param.u32    %r5, [n];
                setp.ge.s32     %p1, %r4, %r5;
                @%p1 bra        EXIT;

                ld.param.u64    %rd1, [a_ptr];
                ld.param.u64    %rd2, [b_ptr];
                ld.param.u64    %rd3, [out_ptr];

                cvt.s64.s32     %rd4, %r4;
                shl.b64         %rd5, %rd4, 2;

                add.s64         %rd6, %rd1, %rd5;
                add.s64         %rd7, %rd2, %rd5;
                add.s64         %rd8, %rd3, %rd5;

                ld.global.f32   %f1, [%rd6];
                ld.global.f32   %f2, [%rd7];

            """);

        if (salt >= SALT_TIMING) {
            ptx.append("    mov.u64         %rd_t0, %globaltimer;\n\n");
        }

        // Core operation
        ptx.append("    ").append(ptxInstruction).append("         %f3, %f1, %f2;\n\n");

        if (salt >= SALT_TIMING) {
            ptx.append("""
                    mov.u64         %rd_t1, %globaltimer;
                    sub.u64         %rd_delta, %rd_t1, %rd_t0;
                    ld.param.u64    %rd9, [timing_ptr];
                    atom.global.add.u64 [%rd9], %rd_delta;

            """);
        }

        ptx.append("""
                st.global.f32   [%rd8], %f3;

            EXIT:
                ret;
            }
            """);

        return ptx.toString();
    }

    // ==================== Utility Methods ====================

    /**
     * Calculate grid size for the given element count.
     */
    public static int calculateGridSize(int elementCount, int blockSize) {
        return (elementCount + blockSize - 1) / blockSize;
    }

    /**
     * Calculate grid size using default block size.
     */
    public static int calculateGridSize(int elementCount) {
        return calculateGridSize(elementCount, ELEMENTWISE_BLOCK_SIZE);
    }

    /**
     * Get the function name for an operation.
     */
    public static String getFunctionName(String opName) {
        return switch (opName) {
            case "add" -> "add_f32";
            case "multiply" -> "multiply_f32";
            case "subtract" -> "subtract_f32";
            case "divide" -> "divide_f32";
            case "maximum" -> "maximum_f32";
            case "minimum" -> "minimum_f32";
            default -> throw new IllegalArgumentException("Unknown operation: " + opName);
        };
    }
}
